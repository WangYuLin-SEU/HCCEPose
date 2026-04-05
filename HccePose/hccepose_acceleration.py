# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

import os
import ctypes
import glob
import site
from pathlib import Path

import numpy as np
import torch


def _import_onnx_modules():
    '''
    ---
    ---
    Import ONNX and ONNXRuntime lazily.  
    ---
    ---
    This helper keeps the acceleration module import-safe when the default
    runtime is still PyTorch. The extra dependencies are imported only when the
    ONNX/TensorRT route is actually requested.

    Returns:
        - onnx: The imported ONNX python module.
        - ort: The imported ONNXRuntime python module.
    ---
    ---
    延迟导入 ONNX 与 ONNXRuntime。  
    ---
    ---
    该辅助函数保证默认仍使用 PyTorch 时，本模块可以安全导入。
    只有在真正启用 ONNX/TensorRT 加速路线时，才会加载额外依赖。

    返回:
        - onnx: 导入后的 ONNX Python 模块。
        - ort: 导入后的 ONNXRuntime Python 模块。
    '''
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            'HccePose ONNX acceleration requires both onnx and onnxruntime in the current environment.'
        ) from exc
    return onnx, ort


def _prepare_onnxruntime_cuda(ort):
    '''
    ---
    ---
    Preload CUDA related shared libraries for ONNXRuntime.  
    ---
    ---
    ONNXRuntime may require CUDA libraries to be loaded explicitly in some
    environments. This helper tries the official preload entry first and falls
    back silently when the runtime version exposes a different signature.

    Args:
        - ort: Imported ONNXRuntime module.

    Returns:
        - None
    ---
    ---
    为 ONNXRuntime 预加载 CUDA 相关动态库。  
    ---
    ---
    在某些环境中，ONNXRuntime 需要显式预加载 CUDA 动态库。
    该函数优先尝试官方预加载接口；若不同版本的签名存在差异，则静默兼容。

    参数:
        - ort: 已导入的 ONNXRuntime 模块。

    返回:
        - None
    '''
    preload_fn = getattr(ort, 'preload_dlls', None)
    if callable(preload_fn):
        try:
            preload_fn()
        except TypeError:
            preload_fn(directory='')
        except Exception:
            pass


def _tensorrt_lib_search_dirs():
    '''
    Directories that may contain libnvinfer (pip layouts, env, common system paths).
    '''
    roots = []
    for sp in list(site.getsitepackages()):
        if sp and os.path.isdir(sp):
            roots.append(sp)
    user_site = getattr(site, 'getusersitepackages', lambda: None)()
    if user_site and os.path.isdir(user_site):
        roots.append(user_site)
    rel_subdirs = (
        'tensorrt_libs',
        os.path.join('nvidia', 'tensorrt', 'lib'),
        os.path.join('nvidia', 'tensorrt', 'lib64'),
    )
    out = []
    for package_root in roots:
        for sub in rel_subdirs:
            d = os.path.join(package_root, sub)
            if os.path.isdir(d):
                out.append(d)
    try:
        import tensorrt as trt

        pkg_dir = Path(trt.__file__).resolve().parent
        out.append(str(pkg_dir))
        out.append(str(pkg_dir.parent))
    except ImportError:
        pass
    trt_root = os.environ.get('TENSORRT_ROOT') or os.environ.get('TensorRT_ROOT')
    if trt_root:
        for sub in ('lib', 'lib64', os.path.join('targets', 'x86_64-linux-gnu', 'lib')):
            d = os.path.join(trt_root, sub)
            if os.path.isdir(d):
                out.append(d)
    for extra in (
        '/usr/lib/x86_64-linux-gnu',
        '/usr/local/tensorrt/lib',
        '/usr/local/TensorRT/lib',
    ):
        if os.path.isdir(extra):
            out.append(extra)
    seen = set()
    uniq = []
    for d in out:
        if d not in seen:
            seen.add(d)
            uniq.append(d)
    return uniq


def tensorrt_libnvinfer_candidates():
    '''
    Paths to libnvinfer*.so* visible under known search dirs (for diagnostics / preflight).
    '''
    hits = []
    for lib_dir in _tensorrt_lib_search_dirs():
        hits.extend(glob.glob(os.path.join(lib_dir, 'libnvinfer.so*')))
    return sorted(set(hits))


def _prepare_tensorrt_libraries():
    '''
    ---
    ---
    Preload TensorRT shared libraries from the current python environment.  
    ---
    ---
    TensorRT execution provider may fail to initialize if the related shared
    libraries are not visible to the process. This helper scans installed
    `tensorrt_libs` packages and loads the common runtime libraries with
    `RTLD_GLOBAL`.

    Returns:
        - loaded: Paths of TensorRT libraries loaded successfully.
    ---
    ---
    从当前 Python 环境中预加载 TensorRT 动态库。  
    ---
    ---
    若进程无法直接看到 TensorRT 相关动态库，TensorRT execution provider
    可能初始化失败。该函数会扫描已安装的 `tensorrt_libs` 目录，并通过
    `RTLD_GLOBAL` 方式预加载常见运行库。

    返回:
        - loaded: 成功加载的 TensorRT 动态库路径列表。
    '''
    for lib_dir in _tensorrt_lib_search_dirs():
        bundle = []
        bundle.extend(sorted(glob.glob(os.path.join(lib_dir, 'libnvinfer.so*'))))
        bundle.extend(sorted(glob.glob(os.path.join(lib_dir, 'libnvonnxparser.so*'))))
        bundle.extend(sorted(glob.glob(os.path.join(lib_dir, 'libnvinfer_plugin.so*'))))
        if not bundle:
            continue
        ok = []
        for lib_path in bundle:
            try:
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                ok.append(lib_path)
            except OSError:
                pass
        if any(os.path.basename(p).startswith('libnvinfer.so') for p in ok):
            return ok
    return []


def _checkpoint_signature(checkpoint_path):
    '''
    ---
    ---
    Build a lightweight cache signature for one checkpoint file.  
    ---
    ---
    The signature combines checkpoint stem, modification time and file size so
    that exported ONNX files are automatically refreshed when the source weights
    change.

    Args:
        - checkpoint_path: Path of the PyTorch checkpoint file.

    Returns:
        - signature: A compact cache key string.
    ---
    ---
    为单个 checkpoint 文件构建轻量级缓存签名。  
    ---
    ---
    该签名由 checkpoint 文件名、修改时间和文件大小组成，用于在源权重变化时
    自动刷新导出的 ONNX 文件。

    参数:
        - checkpoint_path: PyTorch checkpoint 文件路径。

    返回:
        - signature: 紧凑的缓存标识字符串。
    '''
    stat = os.stat(checkpoint_path)
    stem = Path(checkpoint_path).stem
    return f'{stem}_{int(stat.st_mtime)}_{stat.st_size}'


class HccePoseOnnxExportWrapper(torch.nn.Module):
    '''
    ---
    ---
    Thin wrapper used only for exporting the HccePose network body.  
    ---
    ---
    The wrapper exposes the pure network forward graph so ONNX export stays
    focused on the model itself. Post-processing, decoding and pose recovery
    are intentionally kept outside and still reuse the original PyTorch logic.
    ---
    ---
    仅用于导出 HccePose 主网络的轻量封装。  
    ---
    ---
    该封装只暴露纯网络前向图，使 ONNX 导出仅覆盖模型本体。
    后处理、解码和位姿恢复被刻意保留在外部，继续复用原有 PyTorch 逻辑。
    '''
    def __init__(self, net):
        '''
        ---
        ---
        Initialize the export wrapper with the original network module.
        ---
        ---
        使用原始网络模块初始化导出封装。
        '''
        super().__init__()
        self.net = net

    def forward(self, inputs):
        '''
        ---
        ---
        Forward the cropped network input through the pure HccePose network.
        ---
        ---
        将裁剪后的网络输入送入纯 HccePose 网络前向。
        '''
        return self.net(inputs)


class HccePoseOnnxRunner:
    '''
    ---
    ---
    Runtime manager for HccePose ONNX / TensorRT acceleration.  
    ---
    ---
    The runner exports the pure network body to ONNX, creates an
    ONNXRuntime/TensorRT session, executes the network logits, and then reuses
    the original PyTorch decode and geometry logic to keep final behavior
    aligned with the default pipeline.
    ---
    ---
    HccePose 的 ONNX / TensorRT 加速运行管理器。  
    ---
    ---
    该运行器负责将纯网络主体导出为 ONNX，创建 ONNXRuntime/TensorRT
    session，执行网络 logits，并继续复用原始 PyTorch 的解码与几何后处理，
    从而让最终行为尽量与默认推理链保持一致。
    '''
    def __init__(self, pytorch_model, checkpoint_path, cache_dir, device='cuda:0', obj_id=None, input_size=256, provider='onnx'):
        '''
        ---
        ---
        Initialize the accelerated HccePose runner.  
        ---
        ---
        Args:
            - pytorch_model: Reference PyTorch model used for export and shared
              post-processing.
            - checkpoint_path: Source checkpoint path used to derive cache names.
            - cache_dir: Directory that stores exported ONNX files and TRT cache.
            - device: Preferred runtime device.
            - obj_id: Object id used only for readable cache naming.
            - input_size: Crop resolution used by the network.
            - provider: Acceleration backend, usually `onnx` or `tensorrt`.
        ---
        ---
        初始化加速版 HccePose 运行器。  
        ---
        ---
        参数:
            - pytorch_model: 用于导出和共享后处理逻辑的参考 PyTorch 模型。
            - checkpoint_path: 用于构建缓存命名的源 checkpoint 路径。
            - cache_dir: 导出 ONNX 文件与 TRT 缓存的保存目录。
            - device: 期望使用的运行设备。
            - obj_id: 仅用于生成更易读缓存名称的物体编号。
            - input_size: 网络使用的裁剪分辨率。
            - provider: 加速后端，通常为 `onnx` 或 `tensorrt`。
        '''
        self.reference_model = pytorch_model.eval()
        self.checkpoint_path = str(checkpoint_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
        self.obj_id = obj_id
        self.input_size = int(input_size)
        self.ort_module = None
        self.provider = str(provider).lower()
        self.onnx_path = self._ensure_export()
        self.session = self._create_session()
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def _build_onnx_path(self):
        '''
        ---
        ---
        Build the cache path for one exported ONNX model file.  
        ---
        ---
        The cache name is tied to object id, checkpoint signature and input
        size so different objects or weights do not accidentally share one file.
        ---
        ---
        构建单个导出 ONNX 模型文件的缓存路径。  
        ---
        ---
        缓存名称同时绑定物体编号、checkpoint 签名和输入尺寸，
        以避免不同物体或不同权重误用同一导出文件。
        '''
        signature = _checkpoint_signature(self.checkpoint_path)
        obj_name = 'obj_unknown' if self.obj_id is None else f'obj_{int(self.obj_id):02d}'
        return self.cache_dir / f'{obj_name}_{signature}_{self.input_size}.onnx'

    def _ensure_export(self):
        '''
        ---
        ---
        Export the pure HccePose network to ONNX when cache is missing.  
        ---
        ---
        Only the neural network forward graph is exported here. The full
        `inference_batch(...)` logic is intentionally not exported because it
        contains post-processing that is still shared with the PyTorch route.

        Returns:
            - onnx_path: Path to the cached ONNX model.
        ---
        ---
        当缓存缺失时，将纯 HccePose 网络导出为 ONNX。  
        ---
        ---
        这里仅导出神经网络前向图，不导出完整的 `inference_batch(...)`
        逻辑，因为其中的后处理仍与 PyTorch 路线共享。

        返回:
            - onnx_path: 缓存后的 ONNX 模型路径。
        '''
        onnx, _ = _import_onnx_modules()
        onnx_path = self._build_onnx_path()
        if onnx_path.exists():
            return onnx_path

        export_wrapper = HccePoseOnnxExportWrapper(self.reference_model.net).to(self.device).eval()
        dummy_inputs = torch.randn(1, 3, self.input_size, self.input_size, device=self.device, dtype=torch.float32)
        with torch.inference_mode():
            torch.onnx.export(
                export_wrapper,
                dummy_inputs,
                str(onnx_path),
                input_names=['inputs'],
                output_names=['pred_mask_logits', 'pred_front_back_code_logits'],
                dynamic_axes={
                    'inputs': {0: 'batch'},
                    'pred_mask_logits': {0: 'batch'},
                    'pred_front_back_code_logits': {0: 'batch'},
                },
                opset_version=17,
            )
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        return onnx_path

    def _create_session(self):
        '''
        ---
        ---
        Create an ONNXRuntime session for ONNX or TensorRT execution.  
        ---
        ---
        `provider='onnx'` prefers the CUDA execution provider, while
        `provider='tensorrt'` first attempts the TensorRT execution provider and
        enables TensorRT engine caching under the cache directory.

        Returns:
            - session: ONNXRuntime inference session.
        ---
        ---
        为 ONNX 或 TensorRT 执行创建 ONNXRuntime session。  
        ---
        ---
        当 `provider='onnx'` 时，优先使用 CUDA execution provider；
        当 `provider='tensorrt'` 时，会优先尝试 TensorRT execution provider，
        并在缓存目录下启用 TensorRT engine cache。

        返回:
            - session: ONNXRuntime 推理会话。
        '''
        _, ort = _import_onnx_modules()
        self.ort_module = ort
        _prepare_onnxruntime_cuda(ort)
        available = ort.get_available_providers()
        providers = []
        if self.provider == 'tensorrt':
            _prepare_tensorrt_libraries()
            trt_cache_dir = self.cache_dir / 'trt_engine_cache'
            trt_cache_dir.mkdir(parents=True, exist_ok=True)
            if 'TensorrtExecutionProvider' in available:
                providers.append(('TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': 'True',
                    'trt_engine_cache_path': str(trt_cache_dir),
                }))
        if str(self.device).startswith('cuda') and 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
        if 'CPUExecutionProvider' in available:
            providers.append('CPUExecutionProvider')
        if len(providers) == 0:
            providers = available
        return ort.InferenceSession(str(self.onnx_path), providers=providers)

    def forward_logits(self, inputs):
        '''
        ---
        ---
        Run the exported network and return raw logits as torch tensors.  
        ---
        ---
        The runtime receives numpy inputs required by ONNXRuntime, then converts
        outputs back to torch tensors so the downstream PyTorch post-processing
        can stay unchanged.

        Args:
            - inputs: Cropped network inputs as torch tensor or numpy array.

        Returns:
            - pred_mask_logits: Raw mask logits.
            - pred_front_back_code_logits: Raw front/back code logits.
        ---
        ---
        运行导出的网络，并以 torch tensor 形式返回原始 logits。  
        ---
        ---
        运行时先将输入转换为 ONNXRuntime 所需的 numpy，再把输出转回
        torch tensor，以便后续 PyTorch 后处理逻辑保持不变。

        参数:
            - inputs: torch tensor 或 numpy array 形式的网络裁剪输入。

        返回:
            - pred_mask_logits: 原始掩膜 logits。
            - pred_front_back_code_logits: 原始前后表面编码 logits。
        '''
        if torch.is_tensor(inputs):
            input_device = inputs.device
            input_np = inputs.detach().contiguous().cpu().numpy().astype(np.float32)
        else:
            input_device = self.device
            input_np = np.asarray(inputs, dtype=np.float32)
        pred_mask_logits, pred_front_back_code_logits = self.session.run(self.output_names, {self.input_name: input_np})
        pred_mask_logits = torch.from_numpy(pred_mask_logits).to(input_device, dtype=torch.float32)
        pred_front_back_code_logits = torch.from_numpy(pred_front_back_code_logits).to(input_device, dtype=torch.float32)
        return pred_mask_logits, pred_front_back_code_logits

    @torch.inference_mode()
    def inference_batch(self, inputs, Bbox, thershold=0.5):
        '''
        ---
        ---
        Execute accelerated network inference and keep PyTorch post-processing.  
        ---
        ---
        This method mirrors the default HccePose batch inference behavior after
        the network forward pass. The exported backend only replaces the logits
        computation; decoding, coordinate restoration and object-space recovery
        still follow the original PyTorch implementation.

        Args:
            - inputs: Cropped network inputs.
            - Bbox: Bounding boxes used to restore 2D coordinates.
            - thershold: Probability threshold for mask/code binarization.

        Returns:
            - results_dict: Dictionary that matches the original HccePose
              inference outputs used by downstream pose estimation code.
        ---
        ---
        执行加速后的网络推理，并保留 PyTorch 后处理。  
        ---
        ---
        该函数在网络前向之后，尽量复刻默认 HccePose 的 batch 推理行为。
        导出的后端只替换 logits 计算部分；解码、二维坐标恢复和物体空间结果恢复
        仍沿用原始 PyTorch 实现。

        参数:
            - inputs: 裁剪后的网络输入。
            - Bbox: 用于恢复二维坐标的包围盒。
            - thershold: 掩膜与编码二值化的概率阈值。

        返回:
            - results_dict: 与原始 HccePose 推理输出兼容的结果字典，
              可直接供下游位姿估计代码继续使用。
        '''
        pred_mask_logits, pred_front_back_code_logits = self.forward_logits(inputs)
        pred_mask = torch.sigmoid(pred_mask_logits)
        pred_mask[pred_mask > thershold] = 1.0
        pred_mask[pred_mask <= thershold] = 0.0
        pred_mask = pred_mask[:, 0, ...]

        pred_front_code_raw = ((pred_front_back_code_logits.permute(0, 2, 3, 1) + 1) / 2).clone().clamp(0, 1)[..., :24]
        pred_back_code_raw = ((pred_front_back_code_logits.permute(0, 2, 3, 1) + 1) / 2).clone().clamp(0, 1)[..., 24:]

        pred_front_back_code = torch.sigmoid(pred_front_back_code_logits)
        pred_front_back_code[pred_front_back_code > thershold] = 1.0
        pred_front_back_code[pred_front_back_code <= thershold] = 0.0

        pred_front_back_code = pred_front_back_code.permute(0, 2, 3, 1)
        pred_front_code = pred_front_back_code[..., :24]
        pred_back_code = pred_front_back_code[..., 24:]
        pred_front_code = self.reference_model.hcce_decode(pred_front_code) / 255
        pred_back_code = self.reference_model.hcce_decode(pred_back_code) / 255
        if self.reference_model.coord_image is None:
            x = torch.arange(pred_front_code.shape[2], device=pred_front_code.device).to(torch.float32) / pred_front_code.shape[2]
            y = torch.arange(pred_front_code.shape[1], device=pred_front_code.device).to(torch.float32) / pred_front_code.shape[1]
            X, Y = torch.meshgrid(x, y, indexing='xy')
            self.reference_model.coord_image = torch.cat([X[..., None], Y[..., None]], dim=-1)
        coord_image = self.reference_model.coord_image[None, ...].repeat(pred_front_code.shape[0], 1, 1, 1)
        coord_image[..., 0] = coord_image[..., 0] * Bbox[:, None, None, 2] + Bbox[:, None, None, 0]
        coord_image[..., 1] = coord_image[..., 1] * Bbox[:, None, None, 3] + Bbox[:, None, None, 1]
        pred_front_code_0 = pred_front_code * self.reference_model.size_xyz[None, None, None] + self.reference_model.min_xyz[None, None, None]
        pred_back_code_0 = pred_back_code * self.reference_model.size_xyz[None, None, None] + self.reference_model.min_xyz[None, None, None]

        return {
            'pred_mask': pred_mask,
            'coord_2d_image': coord_image,
            'pred_front_code_obj': pred_front_code_0,
            'pred_back_code_obj': pred_back_code_0,
            'pred_front_code': pred_front_code,
            'pred_back_code': pred_back_code,
            'pred_front_code_raw': pred_front_code_raw,
            'pred_back_code_raw': pred_back_code_raw,
            'pred_mask_logits': pred_mask_logits,
            'pred_front_back_code_logits': pred_front_back_code_logits,
        }
