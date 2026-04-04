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
    Import ONNX and ONNXRuntime lazily for FoundationPose acceleration.  
    ---
    ---
    The default FoundationPose path still uses PyTorch. This helper delays the
    optional acceleration dependency import until ONNX or TensorRT is explicitly
    requested.

    Returns:
        - onnx: The imported ONNX python module.
        - ort: The imported ONNXRuntime python module.
    ---
    ---
    为 FoundationPose 加速路线延迟导入 ONNX 与 ONNXRuntime。  
    ---
    ---
    FoundationPose 默认仍走 PyTorch。只有显式启用 ONNX 或 TensorRT
    时，才会通过该函数导入可选加速依赖。

    返回:
        - onnx: 导入后的 ONNX Python 模块。
        - ort: 导入后的 ONNXRuntime Python 模块。
    '''
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            'FoundationPose acceleration requires both onnx and onnxruntime in the current environment.'
        ) from exc
    return onnx, ort


def _prepare_onnxruntime_cuda(ort):
    '''
    ---
    ---
    Preload CUDA related libraries for ONNXRuntime.  
    ---
    ---
    Some ONNXRuntime GPU installations need an explicit preload step before the
    CUDA execution provider can be created reliably.

    Args:
        - ort: Imported ONNXRuntime module.

    Returns:
        - None
    ---
    ---
    为 ONNXRuntime 预加载 CUDA 相关动态库。  
    ---
    ---
    某些 ONNXRuntime GPU 安装环境下，需要先显式预加载动态库，
    才能稳定创建 CUDA execution provider。

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


def _prepare_tensorrt_libraries():
    '''
    ---
    ---
    Preload TensorRT shared libraries for the ONNXRuntime TensorRT provider.  
    ---
    ---
    This helper scans the current python environment for installed TensorRT
    runtime libraries and loads them globally before session creation.

    Returns:
        - loaded: Paths of TensorRT libraries loaded successfully.
    ---
    ---
    为 ONNXRuntime 的 TensorRT provider 预加载 TensorRT 动态库。  
    ---
    ---
    该辅助函数会扫描当前 Python 环境中已安装的 TensorRT 运行库，
    并在创建 session 前以全局方式完成加载。

    返回:
        - loaded: 成功加载的 TensorRT 动态库路径列表。
    '''
    loaded = []
    for package_root in site.getsitepackages():
        lib_dir = os.path.join(package_root, 'tensorrt_libs')
        if not os.path.isdir(lib_dir):
            continue
        lib_paths = sorted(glob.glob(os.path.join(lib_dir, 'libnvinfer*.so*')))
        lib_paths += sorted(glob.glob(os.path.join(lib_dir, 'libnvonnxparser*.so*')))
        lib_paths += sorted(glob.glob(os.path.join(lib_dir, 'libnvinfer_plugin*.so*')))
        for lib_path in lib_paths:
            try:
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                loaded.append(lib_path)
            except OSError:
                pass
        if loaded:
            break
    return loaded


def _checkpoint_signature(checkpoint_path):
    '''
    ---
    ---
    Build a compact cache signature from one checkpoint file.  
    ---
    ---
    The export cache is tied to checkpoint name, modification time and size so
    refiner/scorer caches refresh automatically after weight updates.

    Args:
        - checkpoint_path: Path of the source checkpoint file.

    Returns:
        - signature: Compact signature string used in cache names.
    ---
    ---
    根据单个 checkpoint 文件构建紧凑缓存签名。  
    ---
    ---
    导出缓存会同时绑定 checkpoint 名称、修改时间和文件大小，
    从而在 refiner/scorer 权重更新后自动刷新缓存。

    参数:
        - checkpoint_path: 源 checkpoint 文件路径。

    返回:
        - signature: 用于缓存命名的紧凑签名字符串。
    '''
    stat = os.stat(checkpoint_path)
    stem = Path(checkpoint_path).stem
    return f'{stem}_{int(stat.st_mtime)}_{stat.st_size}'


def _build_providers(provider, device, cache_dir, ort):
    '''
    ---
    ---
    Build ONNXRuntime providers for FoundationPose refiner or scorer.  
    ---
    ---
    The provider list is constructed in priority order. TensorRT is tried first
    when requested, then CUDA, and finally CPU as a safe fallback. TensorRT
    engine cache files are stored under the supplied cache directory.

    Args:
        - provider: Backend selection string, such as `onnx` or `tensorrt`.
        - device: Preferred runtime device.
        - cache_dir: Base cache directory used for TensorRT engine caching.
        - ort: Imported ONNXRuntime module.

    Returns:
        - providers: Provider list accepted by ONNXRuntime.
    ---
    ---
    为 FoundationPose 的 refiner 或 scorer 构建 ONNXRuntime provider 列表。  
    ---
    ---
    provider 会按优先级顺序构建。若请求 TensorRT，则先尝试 TensorRT，
    然后是 CUDA，最后是 CPU 兜底。TensorRT engine cache 会保存在传入的
    cache 目录下。

    参数:
        - provider: 后端选择字符串，例如 `onnx` 或 `tensorrt`。
        - device: 期望使用的运行设备。
        - cache_dir: 用于 TensorRT engine cache 的基础目录。
        - ort: 已导入的 ONNXRuntime 模块。

    返回:
        - providers: ONNXRuntime 可接受的 provider 列表。
    '''
    _prepare_onnxruntime_cuda(ort)
    available = ort.get_available_providers()
    providers = []
    provider = str(provider).lower()
    if provider == 'tensorrt':
        _prepare_tensorrt_libraries()
        trt_cache_dir = Path(cache_dir) / 'trt_engine_cache'
        trt_cache_dir.mkdir(parents=True, exist_ok=True)
        if 'TensorrtExecutionProvider' in available:
            providers.append(('TensorrtExecutionProvider', {
                'trt_engine_cache_enable': 'True',
                'trt_engine_cache_path': str(trt_cache_dir),
            }))
    if str(device).startswith('cuda') and 'CUDAExecutionProvider' in available:
        providers.append('CUDAExecutionProvider')
    if 'CPUExecutionProvider' in available:
        providers.append('CPUExecutionProvider')
    if len(providers) == 0:
        providers = available
    return providers


class FoundationPoseRefineExportWrapper(torch.nn.Module):
    '''
    ---
    ---
    Export wrapper for the FoundationPose refiner network.  
    ---
    ---
    The wrapper keeps only the tensor inputs and the tensor outputs required by
    the refiner. This makes the exported graph small and keeps geometric logic
    outside the ONNX model.
    ---
    ---
    FoundationPose refiner 网络的导出封装。  
    ---
    ---
    该封装只保留 refiner 所需的张量输入和张量输出，
    以便导出的图尽量精简，同时把几何相关逻辑保留在 ONNX 模型之外。
    '''
    def __init__(self, model):
        '''
        ---
        ---
        Initialize the wrapper with the PyTorch refiner model.
        ---
        ---
        使用 PyTorch refiner 模型初始化封装。
        '''
        super().__init__()
        self.model = model

    def forward(self, A, B):
        '''
        ---
        ---
        Forward the paired refiner tensors through the wrapped PyTorch model.
        ---
        ---
        This wrapper only exposes the tensor outputs required by ONNX export:
        translation residuals and rotation residuals.
        ---
        ---
        将成对的 refiner 输入张量送入封装后的 PyTorch 模型前向。
        ---
        ---
        该封装只暴露 ONNX 导出所需的张量输出：
        平移残差与旋转残差。
        '''
        output = self.model(A, B)
        return output['trans'], output['rot']


class FoundationPoseScoreExportWrapper(torch.nn.Module):
    '''
    ---
    ---
    Export wrapper for the FoundationPose scorer network.  
    ---
    ---
    The scorer depends on the candidate count `L`. The wrapper therefore binds
    a fixed `L` value during export so one ONNX graph corresponds to one scorer
    candidate grouping pattern.
    ---
    ---
    FoundationPose scorer 网络的导出封装。  
    ---
    ---
    scorer 的前向与候选数量 `L` 相关，因此导出封装会在导出时绑定一个固定的
    `L` 值，使一份 ONNX 图对应一种 scorer 候选分组模式。
    '''
    def __init__(self, model, fixed_L):
        '''
        ---
        ---
        Initialize the wrapper with a fixed scorer candidate count.
        ---
        ---
        使用固定候选数量初始化 scorer 导出封装。
        '''
        super().__init__()
        self.model = model
        self.fixed_L = int(fixed_L)

    def forward(self, A, B):
        '''
        ---
        ---
        Forward the paired scorer tensors with the fixed candidate count.
        ---
        ---
        The exported scorer graph is specialized for one `L`, so the wrapper
        always forwards with the bound `fixed_L` value.
        ---
        ---
        使用固定候选数量执行成对 scorer 张量的前向。
        ---
        ---
        导出的 scorer 图只对应一个固定的 `L`，因此该封装前向时始终使用
        绑定好的 `fixed_L`。
        '''
        output = self.model(A, B, L=self.fixed_L)
        return output['score_logit']


class FoundationPoseRefineRunner:
    '''
    ---
    ---
    ONNX/TensorRT runtime manager for the FoundationPose refiner.  
    ---
    ---
    Refiner acceleration is managed independently from scorer acceleration. The
    exported model, ONNX cache and TensorRT engine cache are all organized under
    the refiner cache directory so they can be reused across runs.
    ---
    ---
    FoundationPose refiner 的 ONNX/TensorRT 运行管理器。  
    ---
    ---
    refiner 的加速与 scorer 分开管理。导出的模型、ONNX 缓存以及
    TensorRT engine cache 都独立保存在 refiner 对应目录下，以便重复复用。
    '''
    def __init__(self, pytorch_model, checkpoint_path, cache_dir, device='cuda:0', provider='onnx', input_size=160):
        '''
        ---
        ---
        Initialize the FoundationPose refiner runtime.
        ---
        ---
        Args:
            - pytorch_model: Reference PyTorch refiner used for export.
            - checkpoint_path: Source checkpoint path for cache binding.
            - cache_dir: Directory that stores exported ONNX and TRT caches.
            - device: Preferred runtime device.
            - provider: Acceleration backend, usually `onnx` or `tensorrt`.
            - input_size: Spatial input size of the refiner network.
        ---
        ---
        初始化 FoundationPose refiner 运行时。
        ---
        ---
        参数:
            - pytorch_model: 用于导出的参考 PyTorch refiner。
            - checkpoint_path: 用于绑定缓存的源 checkpoint 路径。
            - cache_dir: 导出 ONNX 与 TRT 缓存的保存目录。
            - device: 期望使用的运行设备。
            - provider: 加速后端，通常为 `onnx` 或 `tensorrt`。
            - input_size: refiner 网络的输入空间尺寸。
        '''
        self.reference_model = pytorch_model.eval()
        self.checkpoint_path = str(checkpoint_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
        self.provider = str(provider).lower()
        self.input_size = int(input_size)
        self.onnx_path = self._ensure_export()
        self.session = self._create_session()
        self.input_names = [item.name for item in self.session.get_inputs()]
        self.output_names = [item.name for item in self.session.get_outputs()]

    def _build_onnx_path(self):
        '''
        ---
        ---
        Build the ONNX cache path for the refiner export.
        ---
        ---
        为 refiner 导出构建 ONNX 缓存路径。
        '''
        signature = _checkpoint_signature(self.checkpoint_path)
        return self.cache_dir / f'refinenet_{signature}_{self.input_size}.onnx'

    def _ensure_export(self):
        '''
        ---
        ---
        Export the refiner graph to ONNX when the cache is missing.
        ---
        ---
        The exported graph only contains tensor-to-tensor refiner inference, so
        image crop generation, pose updates and other geometric steps remain in
        the Python side.

        Returns:
            - onnx_path: Cached ONNX file path.
        ---
        ---
        当缓存缺失时，将 refiner 图导出为 ONNX。
        ---
        ---
        导出的图只包含 refiner 的张量到张量推理，因此图像裁剪生成、位姿更新
        等几何步骤仍保留在 Python 侧。

        返回:
            - onnx_path: 缓存后的 ONNX 文件路径。
        '''
        onnx, _ = _import_onnx_modules()
        onnx_path = self._build_onnx_path()
        if onnx_path.exists():
            return onnx_path
        wrapper = FoundationPoseRefineExportWrapper(self.reference_model).to(self.device).eval()
        dummy_A = torch.randn(1, 6, self.input_size, self.input_size, device=self.device, dtype=torch.float32)
        dummy_B = torch.randn(1, 6, self.input_size, self.input_size, device=self.device, dtype=torch.float32)
        mha_fastpath = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)
        try:
            with torch.inference_mode():
                torch.onnx.export(
                    wrapper,
                    (dummy_A, dummy_B),
                    str(onnx_path),
                    input_names=['A', 'B'],
                    output_names=['trans', 'rot'],
                    dynamic_axes={
                        'A': {0: 'batch'},
                        'B': {0: 'batch'},
                        'trans': {0: 'batch'},
                        'rot': {0: 'batch'},
                    },
                    opset_version=17,
                )
        finally:
            torch.backends.mha.set_fastpath_enabled(mha_fastpath)
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        return onnx_path

    def _create_session(self):
        '''
        ---
        ---
        Create the ONNXRuntime session for the refiner graph.
        ---
        ---
        为 refiner 图创建 ONNXRuntime session。
        '''
        _, ort = _import_onnx_modules()
        providers = _build_providers(self.provider, self.device, self.cache_dir, ort)
        return ort.InferenceSession(str(self.onnx_path), providers=providers)

    def forward(self, A, B):
        '''
        ---
        ---
        Run one refiner forward pass and return translation/rotation tensors.
        ---
        ---
        Inputs `A` and `B` represent the paired observation/render feature maps
        expected by the refiner. The output dictionary keeps the same tensor
        field names as the PyTorch model.

        Args:
            - A: First input tensor block for the refiner.
            - B: Second input tensor block for the refiner.

        Returns:
            - outputs: Dictionary with `trans` and `rot` tensors.
        ---
        ---
        执行一次 refiner 前向，并返回平移与旋转张量。
        ---
        ---
        输入 `A` 与 `B` 对应 refiner 所需的成对观测/渲染特征图。
        输出字典继续沿用 PyTorch 模型中的 `trans` 与 `rot` 字段命名。

        参数:
            - A: refiner 的第一组输入张量。
            - B: refiner 的第二组输入张量。

        返回:
            - outputs: 包含 `trans` 与 `rot` 张量的结果字典。
        '''
        if torch.is_tensor(A):
            output_device = A.device
            A_np = A.detach().contiguous().cpu().numpy().astype(np.float32)
        else:
            output_device = self.device
            A_np = np.asarray(A, dtype=np.float32)
        if torch.is_tensor(B):
            B_np = B.detach().contiguous().cpu().numpy().astype(np.float32)
        else:
            B_np = np.asarray(B, dtype=np.float32)
        trans, rot = self.session.run(self.output_names, {self.input_names[0]: A_np, self.input_names[1]: B_np})
        return {
            'trans': torch.from_numpy(trans).to(output_device, dtype=torch.float32),
            'rot': torch.from_numpy(rot).to(output_device, dtype=torch.float32),
        }


class FoundationPoseScoreRunner:
    '''
    ---
    ---
    ONNX/TensorRT runtime manager for the FoundationPose scorer.  
    ---
    ---
    The scorer is exported separately from the refiner because its interface and
    batching semantics differ. In particular, different candidate counts `L`
    need different exported sessions, so the runner maintains one session cache
    per `L` value.
    ---
    ---
    FoundationPose scorer 的 ONNX/TensorRT 运行管理器。  
    ---
    ---
    scorer 与 refiner 分开导出，因为两者接口和批处理语义不同。
    特别是候选数量 `L` 不同时，需要独立的导出 session，
    因此该运行器会为每个 `L` 维护一份单独的 session 缓存。
    '''
    def __init__(self, pytorch_model, checkpoint_path, cache_dir, device='cuda:0', provider='onnx', input_size=160):
        '''
        ---
        ---
        Initialize the FoundationPose scorer runtime.
        ---
        ---
        The arguments are similar to the refiner runner, but the scorer keeps
        multiple sessions internally for different `L` values.
        ---
        ---
        初始化 FoundationPose scorer 运行时。
        ---
        ---
        其参数与 refiner runner 类似，但 scorer 会在内部按 `L` 管理多份 session。
        '''
        self.reference_model = pytorch_model.eval()
        self.checkpoint_path = str(checkpoint_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
        self.provider = str(provider).lower()
        self.input_size = int(input_size)
        self._sessions = {}

    def _build_onnx_path(self, L):
        '''
        ---
        ---
        Build the ONNX cache path for one scorer candidate count `L`.
        ---
        ---
        为指定候选数量 `L` 构建 scorer 的 ONNX 缓存路径。
        '''
        signature = _checkpoint_signature(self.checkpoint_path)
        return self.cache_dir / f'scorenet_{signature}_{self.input_size}_L{int(L)}.onnx'

    def _ensure_export(self, L):
        '''
        ---
        ---
        Export one scorer graph for a fixed candidate count `L`.
        ---
        ---
        The scorer groups candidate poses by `L`, so one static ONNX graph is
        exported for each requested `L` value. This avoids shape ambiguity and
        keeps TensorRT engine caching stable.

        Returns:
            - onnx_path: Cached ONNX file path for the requested `L`.
        ---
        ---
        为固定候选数量 `L` 导出一份 scorer 图。
        ---
        ---
        scorer 会按 `L` 对候选位姿分组，因此每个 `L` 都需要单独导出一份
        静态 ONNX 图，这样既能避免 shape 歧义，也有利于稳定复用 TensorRT engine。

        返回:
            - onnx_path: 对应 `L` 的缓存 ONNX 文件路径。
        '''
        onnx, _ = _import_onnx_modules()
        onnx_path = self._build_onnx_path(L)
        if onnx_path.exists():
            return onnx_path
        wrapper = FoundationPoseScoreExportWrapper(self.reference_model, L).to(self.device).eval()
        dummy_batch = max(int(L), 1)
        dummy_A = torch.randn(dummy_batch, 6, self.input_size, self.input_size, device=self.device, dtype=torch.float32)
        dummy_B = torch.randn(dummy_batch, 6, self.input_size, self.input_size, device=self.device, dtype=torch.float32)
        mha_fastpath = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)
        try:
            with torch.inference_mode():
                torch.onnx.export(
                    wrapper,
                    (dummy_A, dummy_B),
                    str(onnx_path),
                    input_names=['A', 'B'],
                    output_names=['score_logit'],
                    dynamic_axes={
                        'A': {0: 'pair_batch'},
                        'B': {0: 'pair_batch'},
                        'score_logit': {0: 'group_batch'},
                    },
                    opset_version=17,
                )
        finally:
            torch.backends.mha.set_fastpath_enabled(mha_fastpath)
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        return onnx_path

    def _get_session(self, L):
        '''
        ---
        ---
        Get or lazily create the scorer session for one `L`.
        ---
        ---
        The TensorRT engine cache is also split by `L`, so each candidate count
        can build and reuse its own optimized engine independently.

        Returns:
            - runtime: Session bundle containing session, input names and output names.
        ---
        ---
        获取或按需创建指定 `L` 的 scorer session。
        ---
        ---
        TensorRT engine cache 同样会按 `L` 拆分，因此不同候选数量可以独立
        构建并复用各自优化后的 engine。

        返回:
            - runtime: 包含 session、输入名和输出名的运行时字典。
        '''
        L = int(L)
        if L in self._sessions:
            return self._sessions[L]
        _, ort = _import_onnx_modules()
        onnx_path = self._ensure_export(L)
        providers = _build_providers(self.provider, self.device, self.cache_dir / f'L{L}', ort)
        session = ort.InferenceSession(str(onnx_path), providers=providers)
        self._sessions[L] = {
            'session': session,
            'input_names': [item.name for item in session.get_inputs()],
            'output_names': [item.name for item in session.get_outputs()],
        }
        return self._sessions[L]

    def forward(self, A, B, L):
        '''
        ---
        ---
        Run one scorer forward pass for the given candidate grouping `L`.
        ---
        ---
        Inputs `A` and `B` are paired scorer tensors. The output tensor
        `score_logit` keeps the same semantic meaning as the PyTorch scorer.

        Args:
            - A: First scorer input tensor block.
            - B: Second scorer input tensor block.
            - L: Candidate count per group used by the scorer.

        Returns:
            - outputs: Dictionary with `score_logit` tensor.
        ---
        ---
        在给定候选分组 `L` 下执行一次 scorer 前向。
        ---
        ---
        输入 `A` 与 `B` 对应 scorer 的成对张量输入。
        输出 `score_logit` 与 PyTorch scorer 中的语义保持一致。

        参数:
            - A: scorer 的第一组输入张量。
            - B: scorer 的第二组输入张量。
            - L: scorer 使用的每组候选数量。

        返回:
            - outputs: 包含 `score_logit` 张量的结果字典。
        '''
        runtime = self._get_session(L)
        if torch.is_tensor(A):
            output_device = A.device
            A_np = A.detach().contiguous().cpu().numpy().astype(np.float32)
        else:
            output_device = self.device
            A_np = np.asarray(A, dtype=np.float32)
        if torch.is_tensor(B):
            B_np = B.detach().contiguous().cpu().numpy().astype(np.float32)
        else:
            B_np = np.asarray(B, dtype=np.float32)
        score_logit, = runtime['session'].run(runtime['output_names'], {
            runtime['input_names'][0]: A_np,
            runtime['input_names'][1]: B_np,
        })
        return {
            'score_logit': torch.from_numpy(score_logit).to(output_device, dtype=torch.float32),
        }
