# Installation

Use Python 3.10 and an NVIDIA GPU with CUDA support. Run the following commands from the project root. Prepare model weights separately.

Clone the public project and enter its root before installing dependencies:

```bash
git clone https://github.com/WangYuLin-SEU/HCCEPose.git
cd HCCEPose
```

Application entry points are in `examples/inference/`; public training and rendering stage scripts remain in the root.

## HccePose environment

Install [CUDA-enabled PyTorch](https://pytorch.org/get-started/locally/) suitable for your driver, then install the package dependencies:

```bash
python -m pip install -r requirements-inference.txt
python -m zipfile -e bop_toolkit.zip .
```

Extraction must create `bop_toolkit/bop_toolkit_lib/`. Next, prepare the matching detector and per-object pose weights described in [resource preparation](assets.md).

## Optional WAPR

The RGB-D example uses the [pinned WAPR source](https://github.com/WangYuLin-SEU/WAPR/tree/4e591be318a828c91980c58ef8bbdc34e6599881). To build its wheel, run the following Bash commands with Git and Python 3.10 in a working directory outside this project. The wheel contains the sources for the native components.

```bash
git clone --no-checkout https://github.com/WangYuLin-SEU/WAPR.git WAPR-source
git -C WAPR-source checkout --detach 4e591be318a828c91980c58ef8bbdc34e6599881
python -m pip install "setuptools>=77" "wheel>=0.45"
(cd WAPR-source && WAPR_WHEEL_SOURCE_ONLY=1 python wheel_build/build_wheel.py)
python -m pip install WAPR-source/wheel_build/dist/wapr-0.0.11-py3-none-any.whl
```

If you build on a different machine, copy the wheel named above to the target and run the last installation command with its local path in the **same Python environment as HccePose**. On the target, prepare a CUDA toolkit with `nvcc`, a C++ compiler and CUDA-enabled PyTorch. On Linux the OpenGL renderer also needs EGL/OpenGL development headers. Then prepare the pose estimation components:

```bash
python -m wapr.bootstrap --feature core
```

WAPR stores resources under `wapr/` in the user cache directory by default. The optional `WAPR_CACHE_DIR` must be an absolute path; use the same value for component setup and inference. This directory is separate from HccePose’s `data/` and the Torch Hub cache. The build commands above use Bash. On Windows, use Git Bash and prepare MSVC and CUDA as required upstream; do not paste the Bash subshell command directly into PowerShell.

The first setup needs network access to fetch the published weights and dependencies. Test the complete setup on the target GPU with your models and inputs. See the [setup notes for this WAPR revision](https://github.com/WangYuLin-SEU/WAPR/tree/4e591be318a828c91980c58ef8bbdc34e6599881#start).

## Public training and rendering source

Object preparation, PBR rendering, labeling, detector training, pose training and BOP evaluation scripts are included in the project root. Use the current scripts accompanying this documentation and extract the bundled toolkit first:

```bash
python -m zipfile -e bop_toolkit.zip .
```

Stage scripts such as `s1_p1_obj_rename_center.py` and `s4_p2_train_bf_pbr.py` serve different purposes from application examples in `examples/inference/`. Read the input requirements and edit user data paths before launching each stage.

Training and rendering require more dependencies than inference. PyMeshLab prepares meshes, KASAL describes symmetries, and BlenderProc with `bpy` renders PBR scenes. For rendering, run in the repository root:

```bash
python -m zipfile -e blenderproc.zip .
python -m pip install bpy==3.6.0 --extra-index-url https://download.blender.org/pypi/
```

Ubuntu EGL/OpenGL development libraries, drivers and CUDA must match your software choices. Prepare training/rendering environments separately rather than making all these components mandatory for inference. See [Project structure](project-structure.md) for stages, public parameters and layouts.

## MegaPose and network acceleration

First use of MegaPose may clone public source, create a Python 3.9 environment under `.envs/megapose/`, and download models. Ensure Conda is available and the source directory is writable. Keep MegaPose dependencies separate from the main environment; see the [MegaPose tutorial](rgbd-megapose.md).

ONNX / TensorRT selection triggers environment checks that may install packages, export networks or build caches. Compatible CUDA, ONNX Runtime and TensorRT libraries are required; see [acceleration](acceleration.md). Plain PyTorch RGB inference does not require these optional backends.
