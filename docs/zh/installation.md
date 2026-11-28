# 环境安装

使用 Python 3.10 与支持 CUDA 的 NVIDIA GPU。在项目根目录运行以下命令；模型权重需单独准备。

先克隆公开项目并进入根目录，再安装依赖：

```bash
git clone https://github.com/WangYuLin-SEU/HCCEPose.git
cd HCCEPose
```

RGB、RGB-D、训练与渲染示例均位于项目根目录。WAPR 使用 `s4_p3_test_mi10_bin_picking_RGBD_wapr.py`，预测后自动执行后处理。

## HccePose 环境

使用独立的 Python 3.10 环境。若尚未创建，可使用 Python 3.10 解释器执行以下命令，再安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows Git Bash 中改为激活 `.venv/Scripts/activate`。安装 HccePose、准备 WAPR 和运行推理时应始终使用同一个环境。

根据驱动选择适用的 [PyTorch CUDA 安装方式](https://pytorch.org/get-started/locally/)，再安装本包依赖。驱动支持 CUDA 12.8 时，可使用以下 PyTorch 2.8 示例：

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-inference.txt
python -m zipfile -e bop_toolkit.zip .
```

依赖文件固定 Ultralytics 为 8.3.70、NumPy 为 1.26.4、OpenCV 为 4.9.0.80，请保持这组版本；单独升级 OpenCV 可能导致可视化代码不兼容。工具包解压后应存在 `bop_toolkit/bop_toolkit_lib/`。接下来按[资源说明](assets.md)准备匹配的检测器和各物体的位姿权重。

## 可选 WAPR

RGB-D 示例使用[这个固定提交的 WAPR 源码](https://github.com/WangYuLin-SEU/WAPR/tree/4e591be318a828c91980c58ef8bbdc34e6599881)。请在本项目以外的工作目录中使用 Git、Python 3.10 和 Bash 执行以下命令。生成的 wheel 带有本地编译所需的源码。

```bash
git clone --no-checkout https://github.com/WangYuLin-SEU/WAPR.git WAPR-source
git -C WAPR-source checkout --detach 4e591be318a828c91980c58ef8bbdc34e6599881
python -m pip install "setuptools>=77" "wheel>=0.45"
(cd WAPR-source && WAPR_WHEEL_SOURCE_ONLY=1 python wheel_build/build_wheel.py)
python -m pip install WAPR-source/wheel_build/dist/wapr-0.0.11-py3-none-any.whl
```

如在另一台机器构建，将上述 wheel 复制到目标机，把最后一条安装命令中的路径改为目标机上的位置，在 **HccePose 使用的同一 Python 环境**中安装。目标机需备好含 `nvcc` 的 CUDA toolkit、C++ 编译器和支持 CUDA 的 PyTorch。Linux 上的 OpenGL 渲染器还需要 EGL/OpenGL 开发头文件。

回到 HCCEPose 项目根目录，在准备组件前选择资源目录。WAPR 默认使用用户缓存目录中的 `wapr/`；若希望把资源放在当前项目内，可设置绝对路径：

```bash
export WAPR_CACHE_DIR="$(pwd)/.cache/wapr"
```

准备和推理时始终使用同一个值。它独立于 HccePose 的 `data/` 和 Torch Hub 缓存。随后检查环境、准备核心组件，并只下载四份位姿权重：

```bash
python -m wapr.bootstrap --feature core --check
python - <<'PY'
from wapr.bootstrap import prepare_feature
from wapr.download_assets import check_and_fetch_pack

result = prepare_feature("core", allow_replacement=False)
if result.get("status") != "ready":
    raise RuntimeError(result)
check_and_fetch_pack("wapr_sapr_wbps")
PY
```

这组命令保留已有包版本；若报告依赖冲突，先调整环境再重试。该固定版本的完整命令 `python -m wapr.bootstrap --feature core` 还会下载 WAPR 教程样例，HCCEPose 的这个示例不需要这些数据。

以上构建命令使用 Bash；Windows 可使用 Git Bash，并按上游要求准备 MSVC 与 CUDA，不能直接将 Bash 子命令粘贴到 PowerShell。

Linux 默认使用 TensorRT。安装后的 wheel 在四份 FP16 位姿引擎均缺失时，会于首次调用尝试构建整套引擎。使用 PyTorch 2.8 时，请先用下面的命令显式构建：它的融合注意力快路径可能在 ONNX 导出时产生不支持的 `aten::_native_multi_head_attention` 算子。

```bash
python - <<'PY'
import torch
from wapr.export_engines import main as export_engines

torch.backends.mha.set_fastpath_enabled(False)
export_engines()
PY
```

该设置仅在本次导出进程中关闭融合注意力，不修改 WAPR 源码或推理配方。引擎仅部分存在时，也可用同一命令重建整套。引擎缓存应与目标 GPU 和软件栈配套。若先检查 PyTorch 推理路径，可在运行 HCCEPose 示例前显式执行 `export WAPR_BACKEND=torch`；取消该环境变量后恢复上游默认选择。引擎构建和首次下载属于准备过程，不计作稳定推理耗时。

需要在其他 GPU 上自动构建引擎时，在启动 Python 前设置 `CUDA_VISIBLE_DEVICES`，并保持入口脚本的 `device = '0'`，让它使用第一张可见 GPU。

首次配置需联网下载公开权重和依赖。请在目标 GPU 上用配套模型和实际输入验证完整流程。详见[该版本 WAPR 的安装说明](https://github.com/WangYuLin-SEU/WAPR/tree/4e591be318a828c91980c58ef8bbdc34e6599881#start)。

## 公开训练与渲染源码

物体准备、PBR 渲染、标签生成、检测器训练、位姿训练和 BOP 评测脚本位于项目根目录。使用与本文档配套的当前脚本，并先解压随附工具包：

```bash
python -m zipfile -e bop_toolkit.zip .
```

这些脚本包括模型准备入口 `s1_p1_obj_rename_center.py`、训练入口 `s4_p2_train_bf_pbr.py`；根目录的 `s4_p3_*.py` 用于应用推理。阅读各脚本顶部的输入要求，修改用户数据路径后再运行。

训练和渲染的依赖多于纯推理。PyMeshLab 用于网格预处理，KASAL 用于对称信息；BlenderProc 与 `bpy` 用于 PBR 渲染。需要渲染时在仓库根目录执行：

```bash
python -m zipfile -e blenderproc.zip .
python -m pip install bpy==3.6.0 --extra-index-url https://download.blender.org/pypi/
```

Ubuntu 的 EGL/OpenGL 开发依赖、显卡驱动和 CUDA 必须与选用的软件版本匹配。独立准备训练/渲染环境，不要将上述额外组件作为所有推理用户的必需依赖。各阶段的公开参数与目录见[项目结构](project-structure.md)。

## MegaPose 与网络加速

MegaPose 可能在首次调用时克隆公开源码，创建 `.envs/megapose/` Python 3.9 子环境并下载模型。请确保 Conda 可用、源码目录可写；不要把所有 MegaPose 依赖混装到主环境。用法见 [MegaPose 教程](rgbd-megapose.md)。

ONNX / TensorRT 选择会触发后端环境检查，可能安装依赖、导出模型或构建缓存。需要兼容的 CUDA、ONNX Runtime 和 TensorRT 运行库，详见[加速说明](acceleration.md)。纯 PyTorch RGB 推理不要求安装这些可选后端。
