# See-through 动画拆层修改版

基于 [shitagaki-lab/see-through](https://github.com/shitagaki-lab/see-through)（上游基准提交 `df019de`）修改的 Windows 推理与动画拆层源码。保留上游源码及 Apache-2.0 许可证；本仓库不是作者的官方发布。

修改内容包括：Ours Full 批量推理入口、姿态辅助的关节拆分、轮廓和接口处理、图层完整性检查，以及此前示例的专项修正规则。原项目介绍与训练文档见 [README_upstream.md](README_upstream.md)。

## 仓库包含什么

- `main.py`：环境检查、处理新图片、调用通用动画拆层。
- `common/utils/animation_*.py`：新增的拆层、清理、修正与检查模块。
- `scripts/`：批量推理、模型下载及辅助工具。
- `workspace/animation_*hints.json` 和 `common/assets/animation_layouts.json`：已有示例的人工规则。
- `tests/`：图层完整性及后处理检查。

源码支持输入新图片，不局限于此前的 25 组。但那 25 组中部分效果依赖人工轮廓和逐图规则，不能保证任意新图自动达到同样效果。推理结果仍需检查关节、遮挡及补全区域。

新图默认使用通用模式，不按文件名加载旧示例的关节、前景轮廓或专项修正；即使新素材与旧示例重名也适用。切换规则模式或替换图片内容后，会重新计算后处理，避免复用旧切法。检测不确定时保留图层并给出检查提示。衣服自动合并目前仅在确认单个人体姿态时启用，多人物和未识别姿态保留原衣物层。

后续以人物素材为主要改进对象。`scripts/refine_animation_layers.py` 默认也使用通用模式；`--example-rules` 只用于主动重建已有人工检查的示例。重名及缓存隔离测试属于代码行为验证，不代表已经验证所有新人物图的拆分质量。

**仓库不含 Python 环境、模型权重、个人测试图片、拆层结果及示例快照。** 下载 GitHub ZIP 后需要先准备依赖和权重；它不是 19 GB 的离线运行包。

## Windows 使用方法

已验证的本地环境为 Windows 64 位、Python 3.13.5、PyTorch 2.13.0+cu130、RTX 5070 Ti 16 GB。其他环境需要自行确认兼容性；16 GB 是本机运行记录，不是所有图片都保证成功的显存下限。

1. 下载并解压源码，或克隆仓库。在 PyCharm 中选择自己已有的 Python 解释器，不需要重新安装 Python。
2. 已有可用依赖时跳过安装。缺少依赖时，在你选择的环境中执行：

   ```powershell
   python -m pip install --extra-index-url https://download.pytorch.org/whl/cu130 -r requirements-tested-windows.txt
   ```

   该文件是实际运行环境的版本清单，尚未在第二台电脑全新安装验证。上游 `requirements.txt` 含旧版依赖和训练组件，不要同时用它覆盖已配好的推理环境。

3. 先查看模型大小，再下载模型：

   ```powershell
   python scripts/download_inference_models.py --list
   python scripts/download_inference_models.py
   ```

   下载目标均位于本项目 `workspace/models/`。脚本从作者 Hugging Face 仓库取得固定版本 LayerDiff v0.0.1 和 Marigold，另取 LaMa 与 Ultralytics YOLO11s Pose；核对文件校验值，支持保留未完成下载。本入口不下载论文测试集。网络需要能够访问 Hugging Face 及 GitHub。权重与依赖也占磁盘空间，请不要只按源码大小预留空间。

4. 检查环境，然后处理自己的图片文件夹：

   ```powershell
   python main.py check
   python main.py infer --input "E:\输入图片" --output "E:\拆层结果"
   ```

   输入输出应为不同文件夹。输出包含 `原始拆层` 和 `动画拆层`；仅运行作者 Ours Full 时，在最后追加 `--raw-only`。中间处理文件位于输出旁的 `_refinement`。

也可以运行 `启动.bat`。它优先使用项目中的 `.venv\python.exe` 或 `.venv\Scripts\python.exe`，否则使用 PATH 中的 Python；PyCharm 用户可直接运行 `main.py`。

模型准备好后，常规推理从本地加载模型。显卡驱动需要支持所用 PyTorch 的 CUDA 运行库；无需为本项目另装 CUDA Toolkit。

## 输出与范围

动画输出包括透明 PNG、PSD、静态合成及 `关节与图层.json`。JSON 是本项目的图层与关节说明，**不能直接导入 Spine**，也不等同于已绑定完成的骨骼动画。

`main.py examples` 和部分 `scripts/` 工具用于重建此前人工检查过的示例，需要另附原图、原始推理结果及 `workspace/reference_all_20261102/before.zip` 等快照。源码仓库未附这些用户素材；新图请使用 `infer`。

本入口运行 Ours Full 推理和新增后处理，不包含原论文所有消融权重、原始测试真值或完整定量评测。后处理改善不能当作作者模型本身的实验结论。

## 检查后处理代码

```powershell
python -m unittest discover -s tests -p "test_animation_*.py"
```

## 来源与许可

原作者、论文链接和其他依赖说明保留在 [README_upstream.md](README_upstream.md)。代码许可见 [LICENSE](LICENSE)，第三方代码原有声明保留。模型和第三方资源分别遵循其来源的许可；代码许可证不替代模型许可。
