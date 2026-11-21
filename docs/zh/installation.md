# 环境安装

使用 Python 3.10 与支持 CUDA 的 NVIDIA GPU。在项目根目录运行以下命令；模型权重需单独准备。

先克隆公开项目并进入根目录，再安装依赖：

```bash
git clone https://github.com/WangYuLin-SEU/HCCEPose.git
cd HCCEPose
```

应用入口位于 `examples/inference/`，公开训练与渲染阶段脚本仍保留在根目录。

## HccePose 环境

根据驱动选择适用的 [PyTorch CUDA 安装方式](https://pytorch.org/get-started/locally/)，再安装本包依赖：

```bash
python -m pip install -r requirements-inference.txt
python -m zipfile -e bop_toolkit.zip .
```

工具包解压后应存在 `bop_toolkit/bop_toolkit_lib/`。接下来按[资源说明](assets.md)准备匹配的检测器和各物体的位姿权重。

## 可选 WAPR

RGB-D 示例使用[这个固定提交的 WAPR 源码](https://github.com/WangYuLin-SEU/WAPR/tree/4e591be318a828c91980c58ef8bbdc34e6599881)。请在本项目以外的工作目录中使用 Git、Python 3.10 和 Bash 执行以下命令。生成的 wheel 带有本地编译所需的源码。

```bash
git clone --no-checkout https://github.com/WangYuLin-SEU/WAPR.git WAPR-source
git -C WAPR-source checkout --detach 4e591be318a828c91980c58ef8bbdc34e6599881
python -m pip install "setuptools>=77" "wheel>=0.45"
(cd WAPR-source && WAPR_WHEEL_SOURCE_ONLY=1 python wheel_build/build_wheel.py)
python -m pip install WAPR-source/wheel_build/dist/wapr-0.0.11-py3-none-any.whl
```

如在另一台机器构建，将上述 wheel 复制到目标机，把最后一条安装命令中的路径改为目标机上的位置，在 **HccePose 使用的同一 Python 环境**中安装。目标机需备好含 `nvcc` 的 CUDA toolkit、C++ 编译器和支持 CUDA 的 PyTorch。Linux 上的 OpenGL 渲染器还需要 EGL/OpenGL 开发头文件。随后准备位姿估计组件：

```bash
python -m wapr.bootstrap --feature core
```

WAPR 默认将资源放在用户缓存目录中的 `wapr/`，可选的 `WAPR_CACHE_DIR` 必须是绝对路径；配置组件和运行推理时应使用同一个值。该目录独立于 HccePose 的 `data/` 和 Torch Hub 缓存。以上构建命令使用 Bash；Windows 可使用 Git Bash，并按上游要求准备 MSVC 与 CUDA，不能直接将 Bash 子命令粘贴到 PowerShell。

首次配置需联网下载公开权重和依赖。请在目标 GPU 上用配套模型和实际输入验证完整流程。详见[该版本 WAPR 的安装说明](https://github.com/WangYuLin-SEU/WAPR/tree/4e591be318a828c91980c58ef8bbdc34e6599881#start)。

## 公开训练与渲染源码

物体准备、PBR 渲染、标签生成、检测器训练、位姿训练和 BOP 评测脚本已保留在项目根目录。教程源码链接标明原始公开设置；如需使用该历史版本，可另建独立副本：

```bash
git clone https://github.com/WangYuLin-SEU/HCCEPose.git HCCEPose-upstream
cd HCCEPose-upstream
git checkout --detach f75cc4d5db2fc298fc7b853f8c75273757872382
python -m zipfile -e bop_toolkit.zip .
```

这些脚本位于该目录根部，如 `s1_p1_obj_rename_center.py`、`s4_p2_train_bf_pbr.py`；它们不在当前项目的 `examples/inference/` 中。教程链接均指向公开固定版本。阅读各脚本顶部的输入要求，修改用户数据路径后再运行。

训练和渲染的依赖多于纯推理。PyMeshLab 用于网格预处理，KASAL 用于对称信息；BlenderProc 与 `bpy` 用于 PBR 渲染。需要渲染时在上游源码根目录执行：

```bash
python -m zipfile -e blenderproc.zip .
python -m pip install bpy==3.6.0 --extra-index-url https://download.blender.org/pypi/
```

Ubuntu 的 EGL/OpenGL 开发依赖、显卡驱动和 CUDA 必须与选用的软件版本匹配。独立准备训练/渲染环境，不要将上述额外组件作为所有推理用户的必需依赖。各阶段的公开参数与目录见[项目结构](project-structure.md)。

## MegaPose 与网络加速

MegaPose 可能在首次调用时克隆公开源码，创建 `.envs/megapose/` Python 3.9 子环境并下载模型。请确保 Conda 可用、源码目录可写；不要把所有 MegaPose 依赖混装到主环境。用法见 [MegaPose 教程](rgbd-megapose.md)。

ONNX / TensorRT 选择会触发后端环境检查，可能安装依赖、导出模型或构建缓存。需要兼容的 CUDA、ONNX Runtime 和 TensorRT 运行库，详见[加速说明](acceleration.md)。纯 PyTorch RGB 推理不要求安装这些可选后端。
