# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
#
# One-shot MegaPose demo prep, plus unified ONNX/TensorRT setup for `HccePose.tester.Tester`.

import importlib
import os
import subprocess
import sys
from pathlib import Path


def _pip_install(packages):
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '--no-input', *packages],
        check=True,
    )


def _try_import(name):
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


def prefer_large_disk_cache():
    '''
    If a common data disk exists (e.g. AutoDL), default TMPDIR / pip cache there
    to avoid filling the root overlay during conda/pip.
    '''
    base = os.environ.get('HCCEPOSE_DATA_DISK')
    if not base:
        if Path('/root/autodl-tmp').is_dir():
            base = '/root/autodl-tmp'
    if not base or not Path(base).is_dir():
        return
    tmp = Path(base) / 'tmp'
    cache = Path(base) / 'pip-cache'
    tmp.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('TMPDIR', str(tmp))
    os.environ.setdefault('PIP_CACHE_DIR', str(cache))


def ensure_bop_toolkit(project_root):
    lib = Path(project_root) / 'bop_toolkit' / 'bop_toolkit_lib'
    if not lib.is_dir():
        raise FileNotFoundError(
            'Missing bop_toolkit at %s. Extract the provided archive at the repo root.\n'
            '未找到 bop_toolkit：请在仓库根目录解压官方提供的 bop_toolkit（含 bop_toolkit_lib）。' % (Path(project_root) / 'bop_toolkit',)
        )


def ensure_torch_or_exit():
    if _try_import('torch'):
        return
    raise RuntimeError(
        'PyTorch is not installed in this interpreter. Install per README, e.g.:\n'
        '  pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 '
        '--index-url https://download.pytorch.org/whl/cu128\n'
        '当前 Python 未安装 torch，请按 README 先安装 PyTorch（勿在此脚本内自动装全套 CUDA 轮子）。'
    )


def ensure_torch_supports_hccepose_tester():
    '''
    `HccePose.tester` uses `torch.amp.autocast` (PyTorch 2.x). Do not run demo
    scripts with `.envs/megapose/bin/python` (torch 1.11).
    '''
    import torch

    if hasattr(torch, 'amp'):
        return
    raise RuntimeError(
        'This Python has PyTorch %s without `torch.amp` (need 2.x for HccePose Tester).\n'
        'Do not use the MegaPose conda interpreter for repo-root demos; use your main env from README.\n'
        '当前环境的 torch 过旧（例如 MegaPose 专用 conda 里的 1.11），请换用 README 中的 PyTorch 2.x 环境运行本脚本。'
        % (getattr(torch, '__version__', 'unknown'),)
    )


# (import_name, pip packages) — versions aligned with README.md where pinned.
_DEMO_PIP = [
    ('cv2', ['opencv-python==4.9.0.80', 'opencv-contrib-python==4.9.0.80']),
    ('numpy', ['numpy==1.26.4']),
    ('scipy', ['scipy']),
    ('tqdm', ['tqdm']),
    ('kornia', ['kornia==0.6.9']),
    ('ultralytics', ['ultralytics==8.3.233']),
    ('imgaug', ['imgaug']),
    ('trimesh', ['trimesh==4.2.2']),
    ('kasal', ['kasal-6d']),
]


def ensure_demo_pip_extras():
    reconcile_numpy_for_hccepose_demo()
    for mod, pkgs in _DEMO_PIP:
        if _try_import(mod):
            continue
        _pip_install(pkgs)
    reconcile_numpy_for_hccepose_demo()
    reconcile_opencv_pins_after_numpy()
    reconcile_polyscope_for_kasal_on_py39()


def reconcile_numpy_for_hccepose_demo():
    '''
    `kasal-6d` can pull NumPy 2.x, which breaks PyTorch 1.x (MegaPose env) and
    `imgaug`. After optional pip installs, pin NumPy back when 2.x is present.
    '''
    try:
        import numpy as np

        if int(np.__version__.split('.')[0]) < 2:
            return
    except ImportError:
        return
    _pip_install(['numpy==1.26.4'])


def reconcile_opencv_pins_after_numpy():
    '''
    Kasal may upgrade opencv-contrib to a build that wants NumPy 2.x; re-apply README pins.
    '''
    _pip_install(['opencv-python==4.9.0.80', 'opencv-contrib-python==4.9.0.80'])


def reconcile_polyscope_for_kasal_on_py39():
    '''
    `kasal` imports `polyscope`; polyscope>=2.6 uses PEP604 unions and requires Python 3.10+.
    Pin a 3.9-compatible wheel after kasal pulls a newer polyscope (import may raise TypeError, not ImportError).
    '''
    if sys.version_info >= (3, 10):
        return
    try:
        from importlib.metadata import version as _pkg_version
    except ImportError:
        from importlib_metadata import version as _pkg_version  # type: ignore
    try:
        _pkg_version('kasal-6d')
    except Exception:
        return
    _pip_install(['polyscope==2.2.0'])


def ensure_nvdiffrast_for_refinement():
    try:
        import nvdiffrast.torch  # noqa: F401
        return
    except ImportError:
        pass
    _pip_install(['ninja', 'diffrp-nvdiffrast'])


def ensure_megapose_stack(cuda_device='0'):
    from Refinement.refinement_group import Refinement_MP

    dev = str(cuda_device)
    if not dev.startswith('cuda:'):
        dev = 'cuda:%s' % dev
    print('HccePose: preparing MegaPose conda env, pip deps, and rclone weights (first run can take several minutes)...', flush=True)
    Refinement_MP(device=dev).ensure_setup()
    print('HccePose: MegaPose environment ready.', flush=True)


def ensure_onnxruntime_for_hccepose():
    '''
    `HccePose.hccepose_acceleration` needs `onnx` and `onnxruntime`.
    Prefer GPU wheels when CUDA is available in the current torch build.
    '''
    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
        return
    except ImportError:
        pass
    cuda_ok = False
    if _try_import('torch'):
        try:
            import torch

            cuda_ok = bool(torch.cuda.is_available())
        except Exception:
            cuda_ok = False
    base = ['onnx']
    if cuda_ok:
        try:
            _pip_install(base + ['onnxruntime-gpu'])
            return
        except subprocess.CalledProcessError:
            pass
    _pip_install(base + ['onnxruntime'])


# Incremental state: first Tester with onnx/tensorrt runs full ORT stack; tensorrt adds verify once.
_ACCELERATION_ENV_STATE = {'ort_base': False, 'tensorrt_verified': False}


def ensure_acceleration_backend_environment(project_root, hccepose_acceleration, foundationpose_acceleration):
    '''
    Called from `Tester.__init__` when either backend may use ONNX Runtime.

    - `onnx` / `tensorrt` for HccePose or FoundationPose: bop_toolkit, torch 2.x, pip extras, onnxruntime.
    - `tensorrt` for either: TensorRT pip best-effort, lib preload, and EP + libnvinfer checks (once per process).
    '''
    global _ACCELERATION_ENV_STATE
    h = str(hccepose_acceleration).lower()
    f = str(foundationpose_acceleration).lower()
    needs_ort = h in ('onnx', 'tensorrt') or f in ('onnx', 'tensorrt')
    needs_tensorrt = h == 'tensorrt' or f == 'tensorrt'
    if not needs_ort:
        return
    pr = project_root or os.getcwd()
    if not _ACCELERATION_ENV_STATE['ort_base']:
        prefer_large_disk_cache()
        ensure_bop_toolkit(pr)
        ensure_torch_or_exit()
        ensure_torch_supports_hccepose_tester()
        ensure_demo_pip_extras()
        ensure_onnxruntime_for_hccepose()
        _ACCELERATION_ENV_STATE['ort_base'] = True
    if needs_tensorrt and not _ACCELERATION_ENV_STATE['tensorrt_verified']:
        ensure_tensorrt_pip_if_no_libnvinfer()
        verify_tensorrt_execution_provider_or_exit()
        _ACCELERATION_ENV_STATE['tensorrt_verified'] = True


def _prepare_bin_picking_acceleration_common(project_root=None):
    '''
    Standalone ONNX prep (e.g. tooling). Prefer `ensure_acceleration_backend_environment` via `Tester`.
    '''
    pr = project_root or os.getcwd()
    ensure_acceleration_backend_environment(pr, 'onnx', 'pytorch')
    return pr


def ensure_tensorrt_pip_if_no_libnvinfer():
    '''
    Best-effort: NVIDIA `tensorrt` wheels ship shared libs under site-packages.
    If nothing like libnvinfer.so* is visible yet, try pip once.
    '''
    from HccePose.hccepose_acceleration import tensorrt_libnvinfer_candidates, _prepare_tensorrt_libraries

    if tensorrt_libnvinfer_candidates():
        _prepare_tensorrt_libraries()
        return
    try:
        _pip_install(['tensorrt'])
    except subprocess.CalledProcessError:
        pass
    _prepare_tensorrt_libraries()


def verify_tensorrt_execution_provider_or_exit():
    '''
    TensorRT EP can be listed by onnxruntime while libnvinfer.so.10 is still missing
    at runtime (then ORT falls back to CPU). Require discoverable TensorRT libs.
    '''
    import onnxruntime as ort
    from HccePose.hccepose_acceleration import (
        _prepare_onnxruntime_cuda,
        _prepare_tensorrt_libraries,
        tensorrt_libnvinfer_candidates,
    )

    _prepare_onnxruntime_cuda(ort)
    avail = ort.get_available_providers()
    if 'TensorrtExecutionProvider' not in avail:
        raise RuntimeError(
            'TensorrtExecutionProvider is not available in this onnxruntime build.\n'
            'PyPI `onnxruntime-gpu` wheels usually ship without TensorRT. Options:\n'
            '  - Install an ONNX Runtime build that includes TensorRT (see onnxruntime.ai TensorRT EP docs).\n'
            '  - Install matching NVIDIA TensorRT and ensure `libnvinfer` is on LD_LIBRARY_PATH,\n'
            '    or use a wheel that provides a `tensorrt_libs` directory next to site-packages.\n'
            '中文：当前 onnxruntime 未包含 TensorRT 执行提供程序。请安装带 TensorRT 的 ONNX Runtime 与对应 CUDA 版 TensorRT，并保证动态库可被加载。'
        )
    _prepare_tensorrt_libraries()
    if not tensorrt_libnvinfer_candidates():
        raise RuntimeError(
            'TensorRT EP is registered but no libnvinfer.so* was found (e.g. libnvinfer.so.10).\n'
            'ORT then fails to load libonnxruntime_providers_tensorrt.so and may fall back to CPU.\n'
            'Fix:\n'
            '  - pip install tensorrt  (NVIDIA wheel for your CUDA, if available), or\n'
            '  - Install TensorRT from developer.nvidia.com and start Python with:\n'
            '      export LD_LIBRARY_PATH=$TENSORRT_ROOT/lib:$LD_LIBRARY_PATH\n'
            '    (Setting LD_LIBRARY_PATH only inside a running process is often too late for ORT.)\n'
            '中文：已注册 TensorRT EP 但找不到 libnvinfer。请在启动 Python 前安装 TensorRT 并把其 lib 加入 LD_LIBRARY_PATH，或 pip install tensorrt。'
        )
    print('HccePose: TensorRT execution provider is available and libnvinfer is on disk.', flush=True)


def prepare_onnx_bin_picking_demo_environment(project_root=None):
    '''
    Optional manual preflight (ORT stack only). `Tester` already calls
    `ensure_acceleration_backend_environment` when `hccepose_acceleration` / `foundationpose_acceleration`
    is `onnx` or `tensorrt`.
    '''
    pr = project_root or os.getcwd()
    ensure_acceleration_backend_environment(pr, 'onnx', 'pytorch')
    print('HccePose: ONNX acceleration environment ready.', flush=True)


def prepare_tensorrt_bin_picking_demo_environment(project_root=None):
    '''
    Optional manual preflight (ORT + TensorRT verify). `Tester` runs the same when tensorrt is selected.
    '''
    pr = project_root or os.getcwd()
    ensure_acceleration_backend_environment(pr, 'tensorrt', 'pytorch')
    print('HccePose: TensorRT acceleration environment ready.', flush=True)


def prepare_rgbd_fp_vs_mp_demo_environment(cuda_device='0', project_root=None):
    '''
    `s4_p3_test_mi10_bin_picking_RGBD_FP_vs_MP.py`: MegaPose + nvdiffrast only.
    ONNX/TensorRT for FoundationPose is handled when constructing `Tester`.
    '''
    project_root = project_root or os.getcwd()
    prepare_rgbd_megapose_demo_environment(cuda_device=cuda_device, project_root=project_root)


def prepare_rgbd_megapose_demo_environment(cuda_device='0', project_root=None):
    '''
    Call once at the start of RGB-D MegaPose demo scripts (before importing bop_loader).

    - Large-disk TMPDIR / PIP_CACHE_DIR when available
    - Requires torch (manual per README); pip-installs common missing wheels
    - nvdiffrast (Refinement imports)
    - bop_toolkit layout
    - Full MegaPose: conda prefix under .envs/megapose, weights via rclone
    '''
    project_root = project_root or os.getcwd()
    prefer_large_disk_cache()
    ensure_bop_toolkit(project_root)
    ensure_torch_or_exit()
    ensure_torch_supports_hccepose_tester()
    ensure_demo_pip_extras()
    ensure_nvdiffrast_for_refinement()
    ensure_megapose_stack(cuda_device=cuda_device)
