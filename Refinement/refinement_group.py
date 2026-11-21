# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from Refinement import MegaPose as megapose_module
from Refinement.MegaPose import MEGAPOSE_PIP_PACKAGES


class Refinement_MP:
    '''
    ---
    ---
    MegaPose-based refinement/inference group used by the current project.
    ---
    ---
    This class manages the external MegaPose environment, model files, object
    registration and job-based inference so the main tester can call MegaPose
    with the same high-level style as other refinement modules.
    ---
    ---
    当前项目使用的 MegaPose 微调/推理分组类。
    ---
    ---
    该类负责管理外部 MegaPose 环境、模型文件、物体注册以及基于 job 的推理，
    使主 tester 能以与其他 refinement 模块相似的高层方式调用 MegaPose。
    '''
    def __init__(
        self,
        megapose_repo_dir=None,
        megapose_python=None,
        megapose_data_dir=None,
        device='cuda:0',
        model_name_rgb='megapose-1.0-RGB-multi-hypothesis',
        model_name_rgbd='megapose-1.0-RGBD',
        cache_dir=None,
    ):
        '''
        ---
        ---
        Initialize the MegaPose refinement group.
        ---
        ---
        The constructor records repository, environment, data and cache paths so
        later setup, model download and job execution can be handled lazily.
        ---
        ---
        初始化 MegaPose 微调分组。
        ---
        ---
        构造函数会记录仓库、环境、数据与缓存路径，
        以便后续按需完成安装、模型下载和 job 推理。
        '''
        self.project_root = Path(__file__).resolve().parents[1]
        self.megapose_script_path = Path(megapose_module.__file__).resolve()
        self.megapose_repo_dir = Path(megapose_repo_dir) if megapose_repo_dir is not None else self.project_root / 'third_party_megapose6d'
        if megapose_python is not None:
            self.megapose_python = Path(megapose_python)
            self.env_prefix = self.megapose_python.resolve().parents[1]
        else:
            self.env_prefix = self.project_root / '.envs' / 'megapose'
            self.megapose_python = self.env_prefix / 'bin' / 'python'
        self.megapose_data_dir = Path(megapose_data_dir) if megapose_data_dir is not None else self.megapose_repo_dir / 'local_data'
        self.device = str(device)
        self.model_name_rgb = model_name_rgb
        self.model_name_rgbd = model_name_rgbd
        self.cache_dir = Path(cache_dir) if cache_dir is not None else self.project_root / '.megapose_jobs'
        self.tmp_dir = self.project_root / '.tmp'
        self.conda_pkgs_dir = self.project_root / '.conda_pkgs'
        self.repo_url = 'https://github.com/megapose6d/megapose6d.git'
        self.bop_toolkit_dir = self.project_root / 'bop_toolkit'
        self.object_records = []
        self.obj_id_to_label = {}
        self.last_visualization = None

    def _get_conda_exe(self):
        '''
        ---
        ---
        Resolve the conda executable used to manage the MegaPose environment.
        Never hardcode an install path: use the same discovery order as typical
        conda workflows (env vars, then the running interpreter layout, then PATH).
        ---
        ---
        获取用于管理 MegaPose 环境的 conda 可执行程序。
        不写死安装路径：依次使用环境变量、当前解释器所在前缀、再在 PATH 中查找。
        '''
        def _ok(path):
            return path is not None and Path(path).is_file() and os.access(path, os.X_OK)

        for key in ('CONDA_EXE', 'HCCEPOSE_CONDA_EXE'):
            p = os.environ.get(key)
            if _ok(p):
                return str(Path(p).resolve())

        prefix = os.environ.get('CONDA_PREFIX')
        if prefix:
            pfx = Path(prefix).resolve()
            prefix_candidates = [pfx / 'bin' / 'conda']
            if pfx.parent.name == 'envs':
                prefix_candidates.append(pfx.parent.parent / 'bin' / 'conda')
            for rel in prefix_candidates:
                if _ok(rel):
                    return str(rel.resolve())

        # Interpreter is often .../conda_root/bin/python or .../conda_root/envs/name/bin/python
        try:
            bin_dir = Path(sys.executable).resolve().parent
            if bin_dir.name == 'bin':
                direct = bin_dir / 'conda'
                if _ok(direct):
                    return str(direct)
                cur = bin_dir.parent
                for _ in range(8):
                    if cur.parent == cur:
                        break
                    cand = cur / 'bin' / 'conda'
                    if _ok(cand):
                        return str(cand.resolve())
                    cur = cur.parent
        except (OSError, ValueError):
            pass

        p = shutil.which('conda')
        if _ok(p):
            return str(Path(p).resolve())

        raise RuntimeError(
            'Conda executable not found. Set CONDA_EXE (or HCCEPOSE_CONDA_EXE) to the conda binary, '
            'activate a conda env so CONDA_PREFIX is set, or ensure `conda` is on PATH.\n'
            '未找到 conda：请设置 CONDA_EXE（或 HCCEPOSE_CONDA_EXE）、先 conda activate 使 CONDA_PREFIX 生效，'
            '或把 conda 加入 PATH。'
        )

    def _require_bop_toolkit(self):
        '''
        ---
        ---
        Verify that the project-root `bop_toolkit` checkout exists (user-provided).
        ---
        ---
        校验项目根下的 `bop_toolkit` 目录已由用户自备且包含 `bop_toolkit_lib`。
        '''
        lib_dir = self.bop_toolkit_dir / 'bop_toolkit_lib'
        if not self.bop_toolkit_dir.is_dir() or not lib_dir.is_dir():
            raise FileNotFoundError(
                'Missing bop_toolkit at %s (expected directory bop_toolkit_lib inside). '
                'Extract your archive at the HCCEPose project root, or place a clone of '
                'https://github.com/thodan/bop_toolkit there as folder "bop_toolkit". '
                'Automatic git clone is disabled.\n'
                '未找到 bop_toolkit：请在项目根解压提供的压缩包，或自行准备含 bop_toolkit_lib 的 '
                'bop_toolkit 目录（不再自动克隆）。' % (self.bop_toolkit_dir,)
            )

    def _run_host_subprocess(self, args, cwd=None, env=None, capture_output=True):
        '''
        ---
        ---
        Run one host-side subprocess with shared temp and conda cache settings.
        ---
        ---
        使用统一的临时目录和 conda 缓存设置执行一条宿主机子进程命令。
        '''
        host_env = os.environ.copy()
        host_env['TMPDIR'] = str(self.tmp_dir)
        host_env['CONDA_PKGS_DIRS'] = str(self.conda_pkgs_dir)
        if env is not None:
            host_env.update(env)
        completed = subprocess.run(
            [str(arg) for arg in args],
            cwd=str(cwd) if cwd is not None else None,
            env=host_env,
            check=False,
            capture_output=capture_output,
            text=True if capture_output else False,
        )
        if completed.returncode != 0:
            if capture_output:
                stdout = completed.stdout or ''
                stderr = completed.stderr or ''
            else:
                stdout = (
                    '(not captured because capture_output=False; see process output above, '
                    'or call with capture_output=True for logs in this exception)'
                )
                stderr = stdout
            raise RuntimeError(
                'MegaPose host command failed\nCMD: %s\nSTDOUT:\n%s\nSTDERR:\n%s'
                % (' '.join(map(str, args)), stdout, stderr)
            )
        return completed

    def _make_env(self):
        '''
        ---
        ---
        Build the runtime environment dictionary for MegaPose subprocess jobs.
        ---
        ---
        构建 MegaPose 子进程 job 使用的运行环境变量字典。
        '''
        env = os.environ.copy()
        conda_lib = self.env_prefix / 'lib'
        conda_bin = self.env_prefix / 'bin'
        # Resolved paths: MegaPose + yaml load must not depend on process cwd (IDE / scripts).
        env['MEGAPOSE_DIR'] = os.fspath(self.megapose_repo_dir.resolve())
        env['MEGAPOSE_DATA_DIR'] = os.fspath(self.megapose_data_dir.resolve())
        env['CONDA_PREFIX'] = str(self.env_prefix)
        env['PYTHONPATH'] = str(self.bop_toolkit_dir) + os.pathsep + str(self.megapose_repo_dir / 'src') + os.pathsep + env.get('PYTHONPATH', '')
        env['PATH'] = str(conda_bin) + os.pathsep + env.get('PATH', '')
        env['LD_LIBRARY_PATH'] = str(conda_lib) + os.pathsep + env.get('LD_LIBRARY_PATH', '')
        env['TMPDIR'] = str(self.tmp_dir)
        env['CONDA_PKGS_DIRS'] = str(self.conda_pkgs_dir)
        if self.device.startswith('cuda:'):
            env['CUDA_VISIBLE_DEVICES'] = self.device.split(':', 1)[1]
        return env

    def _run_env_python(self, args, cwd=None, capture_output=True):
        '''
        ---
        ---
        Run one python command inside the prepared MegaPose environment.
        ---
        ---
        在准备好的 MegaPose 环境中执行一条 Python 命令。
        '''
        completed = subprocess.run(
            [str(self.megapose_python)] + [str(arg) for arg in args],
            cwd=str(cwd) if cwd is not None else str(self.megapose_repo_dir),
            env=self._make_env(),
            check=False,
            capture_output=capture_output,
            text=True if capture_output else False,
        )
        if completed.returncode != 0:
            stdout = completed.stdout if capture_output else ''
            stderr = completed.stderr if capture_output else ''
            raise RuntimeError(
                'MegaPose subprocess failed\nCMD: %s\nSTDOUT:\n%s\nSTDERR:\n%s'
                % (' '.join(map(str, [self.megapose_python] + list(args))), stdout, stderr)
            )
        return completed

    def _ensure_python_runtime(self, force_reinstall=False):
        '''
        ---
        ---
        Ensure that the dedicated MegaPose python environment exists.
        ---
        ---
        确保 MegaPose 专用 Python 环境已经可用。
        '''
        needs_install = force_reinstall
        if not needs_install:
            try:
                self._run_env_python([
                    '-c',
                    'import joblib, megapose, numpy, cv2, pandas, trimesh, transforms3d, png, bokeh, imageio, psutil, pyarrow, h5py'
                ])
            except RuntimeError:
                needs_install = True
        if needs_install:
            self._run_env_python(['-m', 'pip', 'install', '-e', '.', '--no-deps'], cwd=self.megapose_repo_dir, capture_output=False)
            self._run_env_python(['-m', 'pip', 'install', *MEGAPOSE_PIP_PACKAGES], capture_output=False)

    def _ensure_pinocchio(self):
        '''
        ---
        ---
        MegaPose imports `pinocchio` at runtime. Fresh conda envs get it from
        `conda install pinocchio` in `ensure_setup`; a manually prepared prefix
        (e.g. pip-only PyTorch) may skip that path—install on demand.
        ---
        ---
        MegaPose 运行时需要 `pinocchio`。全新 conda 环境会在 `ensure_setup` 里用
        `conda install pinocchio` 安装；若前缀是手工准备的（例如仅用 pip 装 torch），
        可能漏装本依赖，此处按需补装。
        '''
        if not self.megapose_python.exists():
            return
        try:
            self._run_env_python(['-c', 'import pinocchio'])
        except RuntimeError:
            conda_exe = self._get_conda_exe()
            # capture_output=True so solver/download errors appear in the raised RuntimeError
            self._run_host_subprocess(
                [
                    conda_exe, 'install', '-y', '-p', str(self.env_prefix),
                    '--override-channels', '-c', 'conda-forge', 'pinocchio',
                ],
                capture_output=True,
            )
            self._run_env_python(['-c', 'import pinocchio'])

    def _ensure_mkl_numpy_for_megapose(self):
        '''
        ---
        ---
        `pinocchio` pulls conda-forge MKL 2024+ while PyTorch 1.11 `libtorch_cpu` expects
        an MKL stack that still resolves `iJIT_NotifyEvent` (ImportError at import time).
        MegaPose upstream uses `np.float_`, removed in NumPy 2.0.
        Pin MKL 2021.4 and NumPy 1.26.x inside the MegaPose prefix when needed.
        ---
        ---
        pinocchio 会拉取较新的 MKL，与 PyTorch 1.11 的 libtorch_cpu（iJIT 符号）不兼容；
        MegaPose 源码依赖 NumPy 1.x。按需将前缀内的 MKL / NumPy 固定到兼容版本。
        '''
        if not self.megapose_python.exists():
            return
        try:
            self._run_env_python([
                '-c',
                'import numpy, torch; '
                'assert int(numpy.__version__.split(".")[0]) < 2; '
                'torch.zeros(1)',
            ])
            return
        except RuntimeError:
            pass
        conda_exe = self._get_conda_exe()
        self._run_host_subprocess(
            [
                conda_exe, 'install', '-y', '-p', str(self.env_prefix),
                '--override-channels', '-c', 'conda-forge',
                'mkl=2021.4.0', 'numpy=1.26.4',
            ],
            capture_output=False,
        )
        self._run_env_python([
            '-c',
            'import numpy, torch; '
            'assert int(numpy.__version__.split(".")[0]) < 2; '
            'torch.zeros(1)',
        ])

    def _models_root(self):
        '''
        ---
        ---
        Return the local root directory that stores MegaPose model weights.
        ---
        ---
        返回本地保存 MegaPose 模型权重的根目录。
        '''
        return self.megapose_data_dir / 'megapose-models'

    def _required_model_files(self):
        '''
        ---
        ---
        List the MegaPose model files required by the currently configured variants.
        Inference loads `config.yaml` next to each checkpoint; partial mirrors that
        only ship `.pth.tar` must trigger a full rclone sync.
        ---
        ---
        列出当前配置下 MegaPose 各变体所需的模型文件。
        推理需要每个 checkpoint 旁的 `config.yaml`；仅含权重的镜像不完整时应触发 rclone 补全。
        '''
        out = []
        for run_id in (
            'coarse-rgb-906902141',
            'refiner-rgb-653307694',
            'refiner-rgbd-288182519',
        ):
            base = self._models_root() / run_id
            out.append(base / 'checkpoint.pth.tar')
            out.append(base / 'config.yaml')
        return out

    def ensure_setup(self):
        '''
        ---
        ---
        Ensure that the MegaPose repository, environment and model files exist.
        ---
        ---
        This is the main lazy setup entry used before MegaPose inference.
        ---
        ---
        确保 MegaPose 仓库、运行环境和模型文件已经就绪。
        ---
        ---
        这是 MegaPose 推理前使用的主要按需初始化入口。
        '''
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.conda_pkgs_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._require_bop_toolkit()

        repo_created = False
        env_created = False

        if not (self.megapose_repo_dir / 'src' / 'megapose').exists():
            self._run_host_subprocess(
                ['git', 'clone', self.repo_url, str(self.megapose_repo_dir)],
                capture_output=False,
            )
            repo_created = True

        if not self.megapose_python.exists():
            conda_exe = self._get_conda_exe()
            # Upstream MegaPose requires Python 3.9; see megapose6d conda/environment_full.yaml
            # --override-channels: ignore ~/.condarc mirrors (e.g. broken pkgs/free) so setup stays reproducible
            self._run_host_subprocess(
                [
                    conda_exe, 'create', '-y', '-p', str(self.env_prefix),
                    '--override-channels', '-c', 'conda-forge', 'python=3.9',
                ],
                capture_output=False,
            )
            self._run_host_subprocess(
                [
                    conda_exe, 'install', '-y', '-p', str(self.env_prefix),
                    '--override-channels', '-c', 'pytorch', '-c', 'conda-forge',
                    'pytorch=1.11', 'torchvision=0.12', 'cudatoolkit=11.3', 'pillow', 'rclone',
                ],
                capture_output=False,
            )
            self._run_host_subprocess(
                [
                    conda_exe, 'install', '-y', '-p', str(self.env_prefix),
                    '--override-channels', '-c', 'conda-forge', 'pinocchio',
                ],
                capture_output=False,
            )
            env_created = True

        self._ensure_python_runtime(force_reinstall=repo_created or env_created)
        self._ensure_pinocchio()
        self._ensure_mkl_numpy_for_megapose()

        self.megapose_data_dir.mkdir(parents=True, exist_ok=True)
        self._sync_megapose_pretrained_weights()

    def _ensure_rclone_in_megapose_env(self):
        '''
        Older prefixes may lack `rclone`; MegaPose weights are fetched via rclone.
        '''
        rclone_bin = self.env_prefix / 'bin' / 'rclone'
        if rclone_bin.is_file():
            return
        conda_exe = self._get_conda_exe()
        self._run_host_subprocess(
            [
                conda_exe, 'install', '-y', '-p', str(self.env_prefix),
                '--override-channels', '-c', 'conda-forge', 'rclone',
            ],
            capture_output=False,
        )

    def _rclone_conf_path(self):
        cfg = self.megapose_repo_dir / 'rclone.conf'
        if not cfg.is_file():
            raise FileNotFoundError(
                'MegaPose rclone.conf missing at %s (expected next to third_party_megapose6d checkout).'
                % cfg
            )
        return os.fspath(cfg)

    def _rclone_fetch_megapose_inference_artifacts(self):
        '''
        Upstream `megapose.scripts.download` uses `rclone copyto` with `--exclude *epoch*`;
        with some rclone/cache states the tree sync reports "up to date" while `config.yaml`
        never appears locally. Pull only `config.yaml` + `checkpoint.pth.tar` per run via
        `rclone copy`, which is reliable for HCCEPose inference.
        '''
        rclone = os.fspath(self.env_prefix / 'bin' / 'rclone')
        cfg = self._rclone_conf_path()
        meg_root = self.megapose_data_dir / 'megapose-models'
        meg_root.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env['PATH'] = os.fspath(self.env_prefix / 'bin') + os.pathsep + env.get('PATH', '')
        run_ids = (
            'coarse-rgb-906902141',
            'refiner-rgb-653307694',
            'refiner-rgbd-288182519',
        )
        for rid in run_ids:
            dstdir = meg_root / rid
            dstdir.mkdir(parents=True, exist_ok=True)
            dst = os.fspath(dstdir) + '/'
            for name in ('config.yaml', 'checkpoint.pth.tar'):
                src = 'inria_data:megapose-models/%s/%s' % (rid, name)
                completed = subprocess.run(
                    [
                        rclone, 'copy', src, dst,
                        '--config', cfg,
                        '--retries', '8',
                        '--low-level-retries', '32',
                        '--stats-one-line',
                    ],
                    env=env,
                    capture_output=True,
                    text=True,
                )
                if completed.returncode != 0:
                    raise RuntimeError(
                        'rclone copy failed for %s (exit %s)\n%s\n%s'
                        % (src, completed.returncode, completed.stdout, completed.stderr)
                    )

    def _sync_megapose_pretrained_weights(self):
        '''
        Ensure every MegaPose run dir has checkpoint + config.yaml (Inria rclone).
        Prefer per-file `rclone copy`; fall back to upstream script once if needed.
        '''
        self._ensure_rclone_in_megapose_env()

        def _missing():
            return [p for p in self._required_model_files() if not p.is_file()]

        if not _missing():
            return
        self._rclone_fetch_megapose_inference_artifacts()
        if not _missing():
            return
        try:
            self._run_env_python(['-m', 'megapose.scripts.download', '--megapose_models'], capture_output=False)
        except RuntimeError:
            pass
        self._rclone_fetch_megapose_inference_artifacts()
        missing = _missing()
        if missing:
            raise FileNotFoundError(
                'MegaPose pretrained files are still incomplete after rclone. Missing:\n  %s\n'
                'Check your network connection. The project pulls `config.yaml` and '
                '`checkpoint.pth.tar` per run via rclone (see Refinement_MP._rclone_fetch_megapose_inference_artifacts).\n'
                % '\n  '.join(os.fspath(p) for p in missing)
            )

    def download_models(self):
        '''
        ---
        ---
        Download MegaPose model weights if they are not available locally.
        ---
        ---
        如本地不存在所需权重，则下载 MegaPose 模型文件。
        '''
        self.ensure_setup()

    def download_example_data(self):
        '''
        ---
        ---
        Download the official MegaPose example data bundle.
        ---
        ---
        下载官方 MegaPose 示例数据包。
        '''
        self.ensure_setup()
        self._run_env_python(['-m', 'megapose.scripts.download', '--example_data'], capture_output=False)

    def register_objects(self, obj_id_list, obj_model_list, mesh_units='mm'):
        '''
        ---
        ---
        Register project objects so MegaPose can map object ids to mesh labels.
        ---
        ---
        将项目中的物体注册到 MegaPose 中，以建立物体编号到 mesh label 的映射。
        '''
        self.ensure_setup()
        self.object_records = []
        self.obj_id_to_label = {}
        for obj_id, model_path in zip(obj_id_list, obj_model_list):
            label = 'obj_%06d' % int(obj_id)
            mesh_abs = str(Path(model_path).expanduser().resolve())
            record = {
                'obj_id': int(obj_id),
                'label': label,
                'mesh_path': mesh_abs,
                'mesh_units': str(mesh_units),
            }
            self.object_records.append(record)
            self.obj_id_to_label[int(obj_id)] = label

    def _select_model_name(self, depth=None, model_name=None):
        '''
        ---
        ---
        Select the MegaPose model variant according to the available inputs.
        ---
        ---
        根据当前可用输入自动选择 MegaPose 模型变体。
        '''
        if model_name is not None:
            return model_name
        if depth is not None:
            return self.model_name_rgbd
        return self.model_name_rgb

    def inference_batch(self, K, rgb, depth, bboxes_xyxy, obj_ids, model_name=None, initial_poses=None, megapose_vis=False, megapose_vis_stages=None, megapose_vis_max_items=4):
        '''
        ---
        ---
        Run one MegaPose batch inference or refinement job.
        ---
        ---
        The method packages image, depth, bbox, object id and optional initial
        poses, sends them to the MegaPose subprocess job, and returns project-
        compatible pose outputs and optional visualization results.

        Args:
            - K: Camera intrinsic matrix.
            - rgb: Image array; typically BGR uint8 from OpenCV, converted to RGB before the MegaPose job.
            - depth: Optional depth map in meters.
            - bboxes_xyxy: 2D detection boxes in xyxy format.
            - obj_ids: Object ids aligned with the boxes.
            - model_name: Optional explicit MegaPose model name.
            - initial_poses: Optional initial poses used for refiner-only mode.
            - megapose_vis: Whether to build MegaPose refinement visualization.
            - megapose_vis_stages: Requested refinement stages to visualize.
            - megapose_vis_max_items: Maximum number of poses to visualize.

        Returns:
            - results: MegaPose outputs in the project-compatible format.
            - ``last_visualization`` (read after the call): uint8 **BGR** H×W×3 when ``megapose_vis``; tensor panels are converted to BGR before OpenCV drawing so ``cv2.imwrite`` matches viewers.
        ---
        ---
        执行一次 MegaPose 的 batch 推理或微调 job。
        ---
        ---
        该函数会打包图像、深度、bbox、物体编号以及可选初始位姿，
        发送给 MegaPose 子进程 job，并返回与当前项目兼容的位姿结果以及可选可视化。

        参数:
            - K: 相机内参矩阵。
            - rgb: 输入图像；通常为 OpenCV 读取的 BGR uint8，会在写入 MegaPose job 前转为 RGB。
            - depth: 可选的米单位深度图。
            - bboxes_xyxy: xyxy 格式的二维检测框。
            - obj_ids: 与检测框一一对应的物体编号。
            - model_name: 可选的显式 MegaPose 模型名。
            - initial_poses: 用于仅微调模式的可选初始位姿。
            - megapose_vis: 是否构建 MegaPose 微调可视化。
            - megapose_vis_stages: 需要显示的微调阶段。
            - megapose_vis_max_items: 最多可视化多少个位姿。

        返回:
            - results: 与当前项目兼容格式的 MegaPose 输出结果。
            - 调用后可读 ``last_visualization``：开启 ``megapose_vis`` 时为 uint8 **BGR**；张量图在绘制前已转 BGR，便于 ``cv2.imwrite`` 与看图软件一致。
        '''
        self.last_visualization = None
        if len(self.object_records) == 0:
            raise RuntimeError('MegaPose objects are not registered.')

        obj_ids = np.asarray(obj_ids, dtype=np.int64)
        if bboxes_xyxy is None:
            bboxes_xyxy = np.zeros((obj_ids.shape[0], 4), dtype=np.float32)
        else:
            bboxes_xyxy = np.asarray(bboxes_xyxy, dtype=np.float32)

        if initial_poses is not None:
            initial_poses = np.asarray(initial_poses, dtype=np.float32)
            if initial_poses.shape[0] == 0:
                empty_poses = np.zeros((0, 4, 4), dtype=np.float32)
                empty_scores = np.zeros((0,), dtype=np.float32)
                empty_ids = np.zeros((0,), dtype=np.int64)
                return empty_poses, empty_scores, empty_ids, None
        elif bboxes_xyxy.shape[0] == 0:
            empty_poses = np.zeros((0, 4, 4), dtype=np.float32)
            empty_scores = np.zeros((0,), dtype=np.float32)
            empty_ids = np.zeros((0,), dtype=np.int64)
            return empty_poses, empty_scores, empty_ids, None

        labels = []
        for obj_id in obj_ids.tolist():
            if int(obj_id) not in self.obj_id_to_label:
                raise KeyError('MegaPose object is not registered: %s' % str(obj_id))
            labels.append(self.obj_id_to_label[int(obj_id)])

        selected_model = self._select_model_name(depth=depth, model_name=model_name)
        with tempfile.TemporaryDirectory(dir=str(self.cache_dir), prefix='megapose_job_') as tmp_dir:
            tmp_dir = Path(tmp_dir)
            input_npz = tmp_dir / 'inputs.npz'
            input_json = tmp_dir / 'inputs.json'
            output_npz = tmp_dir / 'outputs.npz'
            rgb_np = np.asarray(rgb, dtype=np.uint8)
            if rgb_np.ndim == 3 and rgb_np.shape[2] == 3:
                rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_BGR2RGB)
            depth_np = None if depth is None else np.asarray(depth, dtype=np.float32)
            np.savez_compressed(
                input_npz,
                rgb=rgb_np,
                K=np.asarray(K, dtype=np.float32),
                bboxes_xyxy=bboxes_xyxy,
                obj_ids=obj_ids,
                depth=np.asarray([] if depth_np is None else depth_np, dtype=np.float32),
                initial_poses=np.asarray([] if initial_poses is None else initial_poses, dtype=np.float32),
            )
            input_json.write_text(json.dumps({
                'model_name': selected_model,
                'objects': self.object_records,
                'labels': labels,
                'has_depth': depth_np is not None,
                'has_initial_poses': initial_poses is not None,
                'vis_enabled': bool(megapose_vis),
                'vis_stages': megapose_vis_stages,
                'vis_max_items': int(megapose_vis_max_items),
            }))
            self._run_env_python([
                str(self.megapose_script_path),
                '--megapose-job',
                '--input-json', str(input_json),
                '--input-npz', str(input_npz),
                '--output-npz', str(output_npz),
            ])
            outputs = np.load(output_npz, allow_pickle=False)
            poses = outputs['poses'].astype(np.float32)
            scores = outputs['scores'].astype(np.float32)
            valid_ids = outputs['valid_ids'].astype(np.int64)
            if 'vis_image' in outputs and outputs['vis_image'].size > 0:
                self.last_visualization = outputs['vis_image'].astype(np.uint8)
            return poses, scores, valid_ids, self.last_visualization

