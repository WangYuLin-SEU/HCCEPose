#!/usr/bin/env python3
# Author: helper script for HCCEPose — see README / hf-dataset-card/README.md
"""
Download subsets of the Hugging Face dataset SEU-WYL/HccePose into the project tree.

Default local_dir is the HCCEPose repo root (parent of scripts/). Files are written
with the same relative paths as on the Hub (e.g. test_imgs/, demo-bin-picking/),
matching s4_p3_* scripts that use os.path.join(current_dir, 'test_imgs'), etc.

Endpoint modes:
  hf     — official hub (clears HF_ENDPOINT for this process if unset by user).
  mirror — sets HF_ENDPOINT to https://hf-mirror.com (mainland China mirror; may change).
  auto   — TCP probe huggingface.co:443; if unreachable, use mirror.

Mirrors are third-party; if download fails, try --endpoint hf from a network that
can reach huggingface.co, or update DEFAULT_MIRROR in this file.

Requires: huggingface_hub (pip install huggingface_hub). Dataset card: hf-dataset-card/README.md.
Optional: HF_TOKEN if you use private forks.
"""
from __future__ import annotations

import argparse
import os
import socket
import sys


DATASET_ID = "SEU-WYL/HccePose"
DEFAULT_MIRROR = "https://hf-mirror.com"
OFFICIAL_HOST = "huggingface.co"
OFFICIAL_PORT = 443
CONNECT_TIMEOUT_SEC = 5

# Paths inside the dataset repo (allow_patterns for snapshot_download)
PRESET_PATTERNS: dict[str, list[str]] = {
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

FOUNDATIONPOSE_WEIGHTS_REPO = "gpue/foundationpose-weights"


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
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
        max_workers=8,
    )


def _download_foundationpose_weights(local_dir: str) -> None:
    from huggingface_hub import snapshot_download

    dest = os.path.join(local_dir, "_downloads_foundationpose_weights")
    os.makedirs(dest, exist_ok=True)
    print(
        "\n[!] FoundationPose weights: community mirror repo (not NVIDIA).\n"
        "    License: https://github.com/NVlabs/FoundationPose — verify before use.\n"
        f"    Downloading {FOUNDATIONPOSE_WEIGHTS_REPO} -> {dest}\n"
    )
    snapshot_download(
        repo_id=FOUNDATIONPOSE_WEIGHTS_REPO,
        repo_type="model",
        local_dir=dest,
        local_dir_use_symlinks=False,
        max_workers=8,
    )
    print(
        "\n[i] Arrange refiner/scorer folders under project root as expected by scripts:\n"
        "    2023-10-28-18-33-37/  and  2024-01-11-20-02-45/\n"
        "    (each with config.yml, model_best.pth). See README and NVlabs/FoundationPose.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"Download SEU-WYL/HccePose dataset slices into the HCCEPose project root."
    )
    parser.add_argument(
        "--preset",
        action="append",
        dest="presets",
        metavar="NAME",
        help="One or more of: test, dataset, tex, weights, all (can repeat).",
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
    parser.add_argument(
        "--foundationpose",
        action="store_true",
        help=f"Also download {FOUNDATIONPOSE_WEIGHTS_REPO} into <dest>/_downloads_foundationpose_weights (read license).",
    )
    args = parser.parse_args()
    if not args.presets:
        parser.error("pass at least one --preset (e.g. --preset test)")

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

    if args.foundationpose:
        try:
            _download_foundationpose_weights(dest)
        except Exception as e:
            print(f"[!] FoundationPose optional download failed: {e}", file=sys.stderr)
            return 1

    print("\n[i] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
