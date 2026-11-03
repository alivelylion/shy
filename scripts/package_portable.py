"""Build the self-contained Windows inference ZIP without reinstalling anything."""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'workspace/portable_build_20261103/SeeThrough'
ARCHIVE=ROOT/'SeeThrough_完整运行包.zip'
SKIP_DIRS={'__pycache__','.git','.pytest_cache','.mypy_cache','.ruff_cache','node_modules',
           'portable_smoke_input','portable_smoke_output','portable_smoke_compact',
           'portable_smoke_compact_v2','_refinement'}


def selected_files(source: Path, runtime: bool=False):
    for base,dirs,files in os.walk(source,followlinks=False):
        dirs[:]=[d for d in dirs if d not in SKIP_DIRS and d!='.cache'
                 and not (runtime and d in {'conda-meta','include','libs','Tools'})]
        for name in sorted(files):
            path=Path(base)/name
            if path.suffix.lower() in {'.pyc','.pyo','.pdb','.log'} or name=='direct_url.json':continue
            yield path


def copy_tree(source: Path, target: Path, hardlink: bool=False, runtime: bool=False) -> int:
    count=0
    for path in selected_files(source,runtime):
        destination=target/path.relative_to(source)
        destination.parent.mkdir(parents=True,exist_ok=True)
        if hardlink:
            try:os.link(path,destination)
            except OSError:shutil.copy2(path,destination)
        else:shutil.copy2(path,destination)
        count+=1
    return count


def prepare() -> None:
    if STAGE.exists():raise FileExistsError(f'Stage already exists: {STAGE}')
    STAGE.mkdir(parents=True)
    for name in ['common','annotators','inference','training','ui','scripts','tests',
                 '立绘测试','立绘测试结果_OursFull_20261027','立绘拆分结果']:
        count=copy_tree(ROOT/name,STAGE/name)
        print('STAGED',name,count,flush=True)
    for pattern in ['main.py','启动.bat','使用说明.txt','LICENSE','README*.md','requirements*.txt','download_assets.ps1']:
        for path in ROOT.glob(pattern):shutil.copy2(path,STAGE/path.name)
    copy_tree(ROOT/'common/assets',STAGE/'assets')
    for relative in ['workspace/animation_clean_hints.json','workspace/animation_integrity_hints.json',
                     'workspace/animation_joint_hints.json','workspace/reference_all_20261102/before.zip',
                     'workspace/finish_audit/before_finish.zip']:
        source=ROOT/relative
        if not source.is_file():raise FileNotFoundError(source)
        destination=STAGE/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination)
    for name in ['.venv','workspace/models']:
        count=copy_tree(ROOT/name,STAGE/name,hardlink=True,runtime=name=='.venv')
        print('STAGED',name,count,flush=True)
    # Do not ship launcher EXEs whose shebang points to the original directory.
    # All supported entry points call the relocatable python.exe directly.
    for path in (STAGE/'.venv/Scripts').glob('*.exe'):path.unlink()
    versions=sorted({f'{d.metadata["Name"]}=={d.version}' for d in metadata.distributions() if d.metadata['Name']})
    (STAGE/'requirements-portable-lock.txt').write_text(
        '# Inventory of the bundled Windows Python 3.13 runtime; no installation needed.\n'+'\n'.join(versions)+'\n',encoding='utf-8')
    models={p.relative_to(STAGE).as_posix():p.stat().st_size
            for p in (STAGE/'workspace/models').rglob('*') if p.is_file()}
    manifest={'platform':'Windows x64','python':'3.13.5','torch':'2.13.0+cu130',
              'inference_model':'Ours Full v0.0.1 (19 classes)','model_files':models,
              'entrypoint':'启动.bat / .venv/python.exe main.py',
              'scope':'generic inference/refinement and separately reviewed 25 examples'}
    (STAGE/'portable_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('STAGE_READY',str(STAGE),flush=True)


def archive() -> None:
    if ARCHIVE.exists():raise FileExistsError(ARCHIVE)
    if not (STAGE/'portable_manifest.json').is_file():raise FileNotFoundError('Prepare the stage first')
    files=list(selected_files(STAGE))
    # LaMa cache is recreated from the shipped checkpoint at startup.
    files=[p for p in files if '.cache' not in p.relative_to(STAGE).parts]
    total=sum(p.stat().st_size for p in files)
    temporary=ARCHIVE.with_suffix('.zip.partial')
    if temporary.exists():raise FileExistsError(temporary)
    print(f'ZIP_START files={len(files)} uncompressed_GiB={total/1024**3:.3f}',flush=True)
    completed=0;last=time.monotonic()
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as output:
        for path in files:
            # Large model tensors are already dense binary data. Store them
            # directly to keep packaging fast; compress source and runtime.
            size=path.stat().st_size
            compression=zipfile.ZIP_STORED if size>128*1024**2 or path.suffix.lower() in {'.zip','.png','.jpg','.jpeg','.psd','.pt','.ckpt','.safetensors'} else zipfile.ZIP_DEFLATED
            output.write(path,'SeeThrough/'+path.relative_to(STAGE).as_posix(),compress_type=compression,compresslevel=1)
            completed+=size
            if time.monotonic()-last>12:
                print(f'ZIP_PROGRESS {completed/1024**3:.2f}/{total/1024**3:.2f} GiB',flush=True);last=time.monotonic()
    print('ZIP_VERIFY_START',flush=True)
    with zipfile.ZipFile(temporary) as check:
        bad=check.testzip()
        if bad:raise RuntimeError(f'ZIP CRC mismatch: {bad}')
        names=set(check.namelist())
        if len(names)!=len(files):raise RuntimeError('ZIP entry count mismatch')
    temporary.replace(ARCHIVE)
    print(f'ZIP_OK {ARCHIVE} bytes={ARCHIVE.stat().st_size} GiB={ARCHIVE.stat().st_size/1024**3:.3f} entries={len(files)}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','archive'])
    args=parser.parse_args()
    prepare() if args.action=='prepare' else archive()
