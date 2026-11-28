# Installation

Use Python 3.10 and an NVIDIA GPU with CUDA support. Run the following commands from the project root. Prepare model weights separately.

Clone the public project and enter its root before installing dependencies:

```bash
git clone https://github.com/WangYuLin-SEU/HCCEPose.git
cd HCCEPose
```

RGB, RGB-D, training and rendering examples are in the project root. WAPR uses `s4_p3_test_mi10_bin_picking_RGBD_wapr.py` and automatically postprocesses its predictions.

## HccePose environment

Use a dedicated Python 3.10 environment. If you do not already have one, create it with your Python 3.10 interpreter and activate it before installing packages:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows Git Bash, activate `.venv/Scripts/activate` instead. Keep the same environment active for HccePose, WAPR setup and inference.

Install [CUDA-enabled PyTorch](https://pytorch.org/get-started/locally/) suitable for your driver, then install the package dependencies. For a driver supporting CUDA 12.8, the PyTorch 2.8 example is:

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-inference.txt
python -m zipfile -e bop_toolkit.zip .
```

The requirements pin Ultralytics to 8.3.70, NumPy to 1.26.4 and OpenCV to 4.9.0.80. Keep these versions together; installing a newer OpenCV independently can break the visualization code. Extraction must create `bop_toolkit/bop_toolkit_lib/`. Next, prepare the matching detector and per-object pose weights described in [resource preparation](assets.md).

## Optional WAPR

The RGB-D example uses the [pinned WAPR source](https://github.com/WangYuLin-SEU/WAPR/tree/4e591be318a828c91980c58ef8bbdc34e6599881). To build its wheel, run the following Bash commands with Git and Python 3.10 in a working directory outside this project. The wheel contains the sources for the native components.

```bash
git clone --no-checkout https://github.com/WangYuLin-SEU/WAPR.git WAPR-source
git -C WAPR-source checkout --detach 4e591be318a828c91980c58ef8bbdc34e6599881
python -m pip install "setuptools>=77" "wheel>=0.45"
(cd WAPR-source && WAPR_WHEEL_SOURCE_ONLY=1 python wheel_build/build_wheel.py)
python -m pip install WAPR-source/wheel_build/dist/wapr-0.0.11-py3-none-any.whl
```

If you build on a different machine, copy the wheel named above to the target and run the last installation command with its local path in the **same Python environment as HccePose**. On the target, prepare a CUDA toolkit with `nvcc`, a C++ compiler and CUDA-enabled PyTorch. On Linux the OpenGL renderer also needs EGL/OpenGL development headers.

Return to the HCCEPose project root and choose the resource directory before setup. WAPR defaults to `wapr/` under the user cache. To keep its resources inside this project instead, set an absolute path:

```bash
export WAPR_CACHE_DIR="$(pwd)/.cache/wapr"
```

Use the same value during preparation and inference. It is separate from HccePose's `data/` and the Torch Hub cache. Then inspect the environment, prepare the core and download only the four pose checkpoints:

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

This keeps existing package versions; resolve any reported dependency conflict before retrying. The pinned version's full `python -m wapr.bootstrap --feature core` command also downloads WAPR tutorial samples. Those samples are unnecessary for this HCCEPose example.

The build commands above use Bash. On Windows, use Git Bash and prepare MSVC and CUDA as required upstream; do not paste the Bash subshell command directly into PowerShell.

Linux defaults to TensorRT. An installed wheel attempts to build all four missing FP16 pose engines at first use. With PyTorch 2.8, prepare the engines explicitly using the command below: its fused attention fast path can otherwise cause an unsupported `aten::_native_multi_head_attention` operator during ONNX export.

```bash
python - <<'PY'
import torch
from wapr.export_engines import main as export_engines

torch.backends.mha.set_fastpath_enabled(False)
export_engines()
PY
```

The setting disables fused attention only in this export process; it does not modify WAPR source or the inference recipe. The same command rebuilds a partial engine set. Keep engine caches local to the target GPU and software stack. To check the PyTorch inference path first, explicitly run `export WAPR_BACKEND=torch` before the HCCEPose example. Unset it to return to the upstream default. Engine construction and first-time downloads are preparation work, not steady-state inference time.

For automatic engine construction on another GPU, set `CUDA_VISIBLE_DEVICES` before starting Python and keep the entry script's `device = '0'`; this selects the first visible GPU.

The first setup needs network access to fetch the published weights and dependencies. Test the complete setup on the target GPU with your models and inputs. See the [setup notes for this WAPR revision](https://github.com/WangYuLin-SEU/WAPR/tree/4e591be318a828c91980c58ef8bbdc34e6599881#start).

## Public training and rendering source

Object preparation, PBR rendering, labeling, detector training, pose training and BOP evaluation scripts are included in the project root. Use the current scripts accompanying this documentation and extract the bundled toolkit first:

```bash
python -m zipfile -e bop_toolkit.zip .
```

Stage scripts such as `s1_p1_obj_rename_center.py` and `s4_p2_train_bf_pbr.py` prepare models or train networks; the root `s4_p3_*.py` scripts run application inference. Read the input requirements and edit user data paths before launching each stage.

Training and rendering require more dependencies than inference. PyMeshLab prepares meshes, KASAL describes symmetries, and BlenderProc with `bpy` renders PBR scenes. For rendering, run in the repository root:

```bash
python -m zipfile -e blenderproc.zip .
python -m pip install bpy==3.6.0 --extra-index-url https://download.blender.org/pypi/
```

Ubuntu EGL/OpenGL development libraries, drivers and CUDA must match your software choices. Prepare training/rendering environments separately rather than making all these components mandatory for inference. See [Project structure](project-structure.md) for stages, public parameters and layouts.

## MegaPose and network acceleration

First use of MegaPose may clone public source, create a Python 3.9 environment under `.envs/megapose/`, and download models. Ensure Conda is available and the source directory is writable. Keep MegaPose dependencies separate from the main environment; see the [MegaPose tutorial](rgbd-megapose.md).

ONNX / TensorRT selection triggers environment checks that may install packages, export networks or build caches. Compatible CUDA, ONNX Runtime and TensorRT libraries are required; see [acceleration](acceleration.md). Plain PyTorch RGB inference does not require these optional backends.
