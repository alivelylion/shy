"""Portable Windows entry point for See-through inference and animation layers."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def configure() -> None:
    os.chdir(ROOT)
    sys.path[:0]=[str(ROOT),str(ROOT/'common')]
    for key,value in {
        'PYTHONNOUSERSITE':'1','PYTHONUTF8':'1',
        'PYTHONPATH':os.pathsep.join([str(ROOT),str(ROOT/'common')]),
        'HF_HOME':str(ROOT/'workspace/.cache/huggingface'),
        'TORCH_HOME':str(ROOT/'workspace/.cache/torch'),
        'YOLO_CONFIG_DIR':str(ROOT/'workspace/.cache/ultralytics'),
        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','YOLO_OFFLINE':'true',
        'OPENBLAS_NUM_THREADS':'8','OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8',
    }.items():os.environ[key]=value
    os.environ['PATH']=os.pathsep.join([str(ROOT/'.venv'),str(ROOT/'.venv/Library/bin'),os.environ.get('PATH','')])
    for key in ('HF_HOME','TORCH_HOME','YOLO_CONFIG_DIR'):Path(os.environ[key]).mkdir(parents=True,exist_ok=True)
    # The author's LaMa loader uses torch.hub's local checkpoint cache.
    lama=ROOT/'workspace/models/lama/lama_large_512px.ckpt'
    cached=Path(os.environ['TORCH_HOME'])/'hub/checkpoints'/lama.name
    if lama.is_file() and not cached.is_file():
        cached.parent.mkdir(parents=True,exist_ok=True)
        try:os.link(lama,cached)
        except FileExistsError:pass
        except OSError:
            temporary=cached.with_name(f'{cached.name}.{os.getpid()}.tmp')
            try:
                shutil.copy2(lama,temporary)
                if not cached.is_file():
                    try:os.replace(temporary,cached)
                    except OSError:
                        if not cached.is_file():raise
            finally:
                if temporary.exists():temporary.unlink()
    if not (ROOT/'assets').exists() and (ROOT/'common/assets').is_dir():
        shutil.copytree(ROOT/'common/assets',ROOT/'assets')


def run_script(name: str, *arguments: str) -> None:
    subprocess.run([sys.executable,'-u',str(ROOT/'scripts'/name),*arguments],cwd=ROOT,check=True)


def check(require_gpu: bool=True) -> None:
    if sys.platform!='win32' or struct.calcsize('P')!=8:
        raise RuntimeError('此离线包的 Python 环境仅支持 Windows 64 位。')
    print(f'项目位置：{ROOT}\nPython：{sys.version.split()[0]} / {sys.executable}',flush=True)
    for name in ('torch','torchvision','diffusers','transformers','cv2','numpy','scipy',
                 'PIL','psd_tools','aggdraw','ultralytics','utils.inference_utils',
                 'utils.animation_detail_repairs','annotators.lama_inpainter'):
        module=importlib.import_module(name)
        location=Path(module.__file__).resolve()
        if name.startswith(('utils.','annotators.')) and not location.is_relative_to(ROOT):
            raise RuntimeError(f'导入了包外代码：{name}: {location}')
    required=[
        'workspace/models/layerdiff_v0.0.1/model_index.json',
        'workspace/models/layerdiff_v0.0.1/unet/diffusion_pytorch_model.safetensors',
        'workspace/models/marigold_v0.0.1/model_index.json',
        'workspace/models/marigold_v0.0.1/unet/diffusion_pytorch_model.safetensors',
        'workspace/models/lama/lama_large_512px.ckpt',
        'workspace/models/pose/yolo11s-pose.pt',
    ]
    manifest=ROOT/'portable_manifest.json'
    if manifest.is_file():
        required+=list(json.loads(manifest.read_text(encoding='utf-8'))['model_files'])
    for relative in set(required):
        path=ROOT/relative
        if not path.is_file() or path.stat().st_size==0:raise FileNotFoundError(f'缺少模型文件：{path}')
    if manifest.is_file():
        for relative,size in json.loads(manifest.read_text(encoding='utf-8'))['model_files'].items():
            if (ROOT/relative).stat().st_size!=size:raise RuntimeError(f'模型文件长度不符：{relative}')
    from safetensors import safe_open
    tensors=0
    for path in (ROOT/'workspace/models').rglob('*.safetensors'):
        with safe_open(str(path),framework='pt',device='cpu') as model:tensors+=len(model.keys())
    import torch
    print(f'PyTorch：{torch.__version__}；内置 CUDA：{torch.version.cuda}；模型头检查：{tensors} 个张量',flush=True)
    if require_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA 不可用：请检查 NVIDIA 显卡和驱动。本包无需另装 Python 或 CUDA Toolkit。')
        torch.ones(8,device='cuda').sum().item()
        gpu=torch.cuda.get_device_properties(0)
        print(f'显卡：{gpu.name}；显存：{gpu.total_memory/1024**3:.1f} GiB',flush=True)
    print('CHECK_OK：运行库、源码和本地权重可用。',flush=True)


def compact_new_results(source: Path, output: Path) -> None:
    """Run generic refinement; reviewed sample-specific rules stay in examples."""
    from utils.animation_parts import refine_for_animation, load_refined, load_layers
    from utils.animation_finish import finish
    from utils.animation_clean import export_compact
    from utils.animation_integrity import foreground_mask, read_hints
    from utils.animation_special import preserve_visible_source
    items=sorted(p.parent for p in source.rglob('info.json') if (p.parent/'src_img.png').is_file())
    if not items:raise FileNotFoundError(f'未找到原始拆层结果：{source}')
    for item in items:
        intermediate=output.parent/'_refinement'/item.relative_to(source)
        refine_for_animation(item,intermediate,ROOT/'workspace/models/pose/yolo11s-pose.pt',previews=False)
        layers,info=load_refined(intermediate)
        src,originals=load_layers(item)
        remainders=[p for p in layers if p.role=='source_unclassified']
        if remainders:
            target,_=foreground_mask(src,originals,read_hints(item.name)[0])
            layers=preserve_visible_source(src,[p for p in layers if p not in remainders],target)
            info['source_edge_assignment']={
                'pixels':sum(int(p.mask.sum()) for p in remainders),
                'method':'nearest existing part; source pixels retained; anatomical assignment requires review'}
        layers,info=finish(item.name,layers,info,originals)
        export_compact(layers,output/item.relative_to(source),info)
        print(f'ANIMATION_DONE：{output/item.relative_to(source)}',flush=True)


def infer(input_dir: Path, output: Path, animation: bool) -> None:
    input_dir=input_dir.expanduser().resolve(strict=True);output=output.expanduser().resolve()
    if not input_dir.is_dir():raise ValueError('输入应为图片文件夹。')
    if output==input_dir or output.is_relative_to(input_dir):raise ValueError('输出文件夹应放在输入文件夹之外。')
    check()
    raw=output/'原始拆层'
    run_script('run_ours_full_batch.py','--input-dir',str(input_dir),'--output-dir',str(raw))
    if animation:compact_new_results(raw,output/'动画拆层')


def interactive() -> list[str]:
    print('See-through 动画拆层\n1. 检查运行环境\n2. 处理新图片文件夹\n3. 重建已修过的 25 组示例（需另附示例素材）\n0. 退出')
    choice=input('请选择：').strip()
    if choice=='0':return []
    if choice=='1':return ['check']
    if choice=='3':return ['examples']
    if choice=='2':
        source=input('图片文件夹（可粘贴路径）：').strip().strip('"')
        destination=input('输出文件夹（留空使用 workspace/新图片结果）：').strip().strip('"')
        return ['infer','--input',source,'--output',destination or str(ROOT/'workspace/新图片结果')]
    raise ValueError('无效选项。')


def main() -> int:
    configure()
    parser=argparse.ArgumentParser(description='Windows 推理和动画拆层；源码安装见 README.md。')
    commands=parser.add_subparsers(dest='command',required=True)
    checks=commands.add_parser('check',help='检查导入、权重文件和显卡')
    checks.add_argument('--cpu-only',action='store_true',help='仅核对文件/导入，跳过 CUDA 测试；不代表支持 CPU 完整推理')
    inference=commands.add_parser('infer',help='对新图片运行 Ours Full 和通用动画拆层')
    inference.add_argument('--input',type=Path,required=True)
    inference.add_argument('--output',type=Path,default=ROOT/'workspace/新图片结果')
    inference.add_argument('--raw-only',action='store_true',help='仅输出作者的原始 Ours Full 结果')
    examples=commands.add_parser('examples',help='从随包快照重建当前已修过的示例；会更新示例结果目录')
    examples.add_argument('--only',nargs='+')
    args_list=sys.argv[1:] if len(sys.argv)>1 else interactive()
    if not args_list:return 0
    args=parser.parse_args(args_list)
    if args.command=='check':check(not args.cpu_only)
    elif args.command=='infer':infer(args.input,args.output,not args.raw_only)
    else:
        if not (ROOT/'workspace/reference_all_20261102/before.zip').is_file():
            raise FileNotFoundError('examples 需要完整运行包中的示例快照和图片；GitHub 源码版请用 infer 处理新图片。')
        check()
        run_script('standardize_animation_delivery.py',*(['--only',*args.only] if args.only else ['--all']))
    return 0


if __name__=='__main__':
    bundled=next((p for p in (ROOT/'.venv/python.exe',ROOT/'.venv/Scripts/python.exe') if p.is_file()),None)
    if bundled is not None and Path(sys.executable).resolve()!=bundled.resolve():
        raise SystemExit(subprocess.call([str(bundled),'-u',str(Path(__file__).resolve()),*sys.argv[1:]],cwd=ROOT))
    try:raise SystemExit(main())
    except (RuntimeError,ValueError,FileNotFoundError,subprocess.CalledProcessError) as error:
        print(f'运行失败：{error}',file=sys.stderr)
        raise SystemExit(1)
