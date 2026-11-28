#!/usr/bin/env python3
# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

# HCCEPose public asset downloader / HCCEPose 公开资源下载工具
"""
Download subsets of the Hugging Face dataset SEU-WYL/HccePose into the project tree.

Default local_dir is the HCCEPose repo root (parent of scripts/). Downloads keep
the Hub paths (e.g. test_imgs/, demo-bin-picking/). Bundled samples instead live
under sample_data/images, sample_data/rgbd and sample_data/videos; point the
entry script at the downloaded directory when using additional Hub inputs.
下载保留 Hub 路径；仓库自带样例放在 sample_data 下。使用额外下载的输入时，
将入口脚本的数据路径指向实际下载目录。

Endpoint modes:
  hf     — official hub (clears HF_ENDPOINT for this process if unset by user).
  mirror — sets HF_ENDPOINT to https://hf-mirror.com (mainland China mirror; may change).
  auto   — TCP probe huggingface.co:443; if unreachable, use mirror.

Mirrors are third-party; if download fails, try --endpoint hf from a network that
can reach huggingface.co, or update DEFAULT_MIRROR in this file.

Requires: huggingface_hub. Use --preset demo --dest data for the seven-file inference example.
使用 --preset demo --dest data 下载七文件推理示例；旧预设默认仍写入项目根目录。
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import os
import socket
import sys

# The wget fallback reuses validation without requiring the optional Hub SDK.
# wget 备用下载器复用校验逻辑时，不强制依赖可选 Hub SDK。
try:
    from huggingface_hub import snapshot_download
except ModuleNotFoundError as exc:
    if exc.name != 'huggingface_hub':
        raise
    snapshot_download = None


# Public resource revision; bytes and weight digests match the documentation.
# 公开资源版本；字节数和权重摘要与文档一致。
HF_REVISION = '4e482e540314622981ccd4fce5beddac7f9d3e1d'
CHECKPOINT_PATH = 'demo-bin-picking/HccePose/obj_01/best_score/0_9837step824500'
REQUIRED_DEMO_FILES = {
    'demo-bin-picking/models/models_info.json': (941, None),
    'demo-bin-picking/models/obj_000001.ply': (128911, None),
    CHECKPOINT_PATH: (349827430, '3308469a5082feb98181286026003e0b00ea8a349b8f0921006873a177dd4c70'),
    'demo-bin-picking/yolo11/train_obj_s/detection/obj_s/yolo11-detection-obj_s.pt':
        (114453888, '5a958cf1d90d3529f02ba1021aab164a369316f187f98fb01b151c484d3cfd13'),
    'test_imgs_RGBD/000003_rgb.png': (208215, None),
    'test_imgs_RGBD/000003_depth.png': (59239, None),
    'test_imgs_RGBD/000003_camK.json': (184, None),
}


def validate_selected_demo(root, patterns):
    """Validate selected demo files and report extra checkpoints without deletion.
    校验选中的示例文件，发现额外权重时报错而不删除文件。
    """
    selected = [name for name in REQUIRED_DEMO_FILES
                if any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)]
    if CHECKPOINT_PATH in selected:
        checkpoint = os.path.join(root, *CHECKPOINT_PATH.split('/'))
        directory = os.path.dirname(checkpoint)
        if not os.path.isdir(directory):
            raise FileNotFoundError(checkpoint)
        extras = [os.path.join(directory, name) for name in os.listdir(directory)
                  if os.path.isfile(os.path.join(directory, name)) and name != os.path.basename(checkpoint)]
        if extras:
            raise RuntimeError('Unexpected checkpoint files: ' + ', '.join(sorted(extras)))
    for relative_path in selected:
        expected_size, expected_digest = REQUIRED_DEMO_FILES[relative_path]
        path = os.path.join(root, *relative_path.split('/'))
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        actual_size = os.path.getsize(path)
        if actual_size != expected_size:
            raise RuntimeError(f'File size mismatch: {path}; expected {expected_size}, got {actual_size}')
        if expected_digest is not None:
            digest = hashlib.sha256()
            with open(path, 'rb') as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != expected_digest:
                raise RuntimeError(f'SHA-256 mismatch: {path}')
    camera_relative = 'test_imgs_RGBD/000003_camK.json'
    if camera_relative in selected:
        # Write a separate matrix only after validation; keep calibration intact.
        # 全部校验通过后才另存矩阵，保留原始标定文件。
        camera_dir = os.path.join(root, 'test_imgs_RGBD')
        with open(os.path.join(camera_dir, '000003_camK.json'), encoding='utf-8') as handle:
            camera = json.load(handle)
        values = [float(camera[key]) for key in ('fx', 'fy', 'cx', 'cy')]
        if not all(math.isfinite(x) for x in values) or values[0] <= 0 or values[1] <= 0:
            raise ValueError('Invalid frame intrinsics in 000003_camK.json')
        fx, fy, cx, cy = values
        matrix = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        with open(os.path.join(camera_dir, '000003_camera_matrix.json'), 'w', encoding='utf-8') as handle:
            json.dump(matrix, handle, indent=2)
    return selected


DATASET_ID = "SEU-WYL/HccePose"
DEFAULT_MIRROR = "https://hf-mirror.com"
OFFICIAL_HOST = "huggingface.co"
OFFICIAL_PORT = 443
CONNECT_TIMEOUT_SEC = 5

# Paths inside the dataset repo (allow_patterns for snapshot_download)
PRESET_PATTERNS: dict[str, list[str]] = {
    "demo": list(REQUIRED_DEMO_FILES),
    "test": [
        "test_imgs/*",
        "test_videos/*",
        "test_imgs_RGBD/*",
    ],
    "dataset": [
        "demo-bin-picking/**",
    ],
    "tex": [
        "demo-tex-objs/**",
    ],
    "weights": [
        "demo-bin-picking/models/**",
        "demo-bin-picking/yolo11/**",
        "demo-bin-picking/HccePose/**",
    ],
}



def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _tcp_open(host: str, port: int, timeout: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _configure_endpoint(mode: str) -> str:
    """Return label describing active endpoint ('official' or mirror URL)."""
    if mode == "mirror":
        os.environ["HF_ENDPOINT"] = DEFAULT_MIRROR.rstrip("/")
        return os.environ["HF_ENDPOINT"]
    if mode == "hf":
        os.environ.pop("HF_ENDPOINT", None)
        return "https://huggingface.co"
    # auto
    if _tcp_open(OFFICIAL_HOST, OFFICIAL_PORT, CONNECT_TIMEOUT_SEC):
        os.environ.pop("HF_ENDPOINT", None)
        return "https://huggingface.co"
    os.environ["HF_ENDPOINT"] = DEFAULT_MIRROR.rstrip("/")
    return os.environ["HF_ENDPOINT"]


def _patterns_for_presets(presets: list[str]) -> list[str]:
    out: list[str] = []
    for p in presets:
        if p == "all":
            for key in ("test", "dataset", "tex"):
                out.extend(PRESET_PATTERNS[key])
            return sorted(set(out))
        if p not in PRESET_PATTERNS:
            raise SystemExit(f"Unknown preset {p!r}. Choose from: {sorted(PRESET_PATTERNS)} + all")
        out.extend(PRESET_PATTERNS[p])
    return sorted(set(out))


def _run_download(local_dir: str, allow_patterns: list[str]) -> None:
    if snapshot_download is None:
        raise ImportError('Install huggingface_hub or use scripts/wget_hf_demo_assets.py')
    snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        revision=HF_REVISION,
        endpoint=os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
        token=False,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
        max_workers=8,
    )
    selected = validate_selected_demo(local_dir, allow_patterns)
    print(f"[i] Validated {len(selected)} fixed demo files")




def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"Download SEU-WYL/HccePose dataset slices into the HCCEPose project root."
    )
    parser.add_argument(
        "--preset",
        action="append",
        dest="presets",
        metavar="NAME",
        help="One or more of: demo (seven files), test, dataset, tex, weights, all (can repeat).",
    )
    parser.add_argument(
        "--endpoint",
        choices=("auto", "hf", "mirror"),
        default="auto",
        help="Hub endpoint: official (hf), China mirror, or auto-detect (default).",
    )
    parser.add_argument(
        "--dest",
        default=None,
        help="Destination directory (default: HCCEPose project root).",
    )
    args = parser.parse_args()
    if not args.presets:
        parser.error("pass at least one --preset (e.g. --preset demo --dest data)")

    dest = os.path.abspath(args.dest or _project_root())
    os.makedirs(dest, exist_ok=True)

    active = _configure_endpoint(args.endpoint)
    print(f"[i] Using hub endpoint: {active}")

    try:
        patterns = _patterns_for_presets(args.presets)
        print(f"[i] Presets: {args.presets} -> {len(patterns)} allow_patterns")
        print(f"[i] Downloading to: {dest}")
        _run_download(dest, patterns)
    except Exception as e:
        if args.endpoint == "auto" and os.environ.get("HF_ENDPOINT"):
            print("[!] Download failed via mirror; retry with: --endpoint hf (from abroad)", file=sys.stderr)
        elif args.endpoint == "auto":
            print("[!] Download failed; retry with: --endpoint mirror", file=sys.stderr)
        print(f"[!] Error: {e}", file=sys.stderr)
        return 1

    print("\n[i] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
