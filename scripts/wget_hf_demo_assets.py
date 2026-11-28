#!/usr/bin/env python3
"""
Download HCCEPose demo assets via wget; verify size against Hugging Face tree API.
Skips files that already match Hub-reported byte size. Omits test_imgs *_show_* and
test_videos *_show_* (generated artifacts, not inputs).

Use when huggingface_hub snapshot_download stalls or disconnects: same file layout
as download_hf_assets.py (repo root paths).

Hub paths are retained; bundled samples are separately stored under sample_data/.
保留 Hub 的目录名称；仓库自带样例单独存放在 sample_data/。

Run from repo root:
  python scripts/wget_hf_demo_assets.py
  python scripts/wget_hf_demo_assets.py --endpoint hf --jobs 8
  python scripts/wget_hf_demo_assets.py --verify-only
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

# Resolve the adjacent shared manifest even when launched outside the project.
# 从当前脚本定位相邻的共用清单，支持从仓库外启动。
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from download_hf_assets import HF_REVISION, validate_selected_demo

DATASET = "SEU-WYL/HccePose"
OFFICIAL_HUB = "https://huggingface.co"
DEFAULT_MIRROR = "https://hf-mirror.com"
OFFICIAL_HOST = "huggingface.co"
OFFICIAL_PORT = 443
CONNECT_TIMEOUT_SEC = 5
DEFAULT_JOBS = min(8, max(2, (os.cpu_count() or 4) * 2))
DATASET_SUBTREES = [
    "test_imgs",
    "test_videos",
    "test_imgs_RGBD",
    "demo-bin-picking/models",
    "demo-bin-picking/yolo11",
    "demo-bin-picking/HccePose",
    "demo-tex-objs/models",
    "demo-tex-objs/yolo11",
    "demo-tex-objs/HccePose",
]
_print_lock = threading.Lock()


def _tcp_open(host: str, port: int, timeout: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def hub_base(endpoint: str) -> tuple[str, str]:
    """Return (origin_without_trailing_slash, label_for_logs)."""
    if endpoint == "mirror":
        return DEFAULT_MIRROR.rstrip("/"), DEFAULT_MIRROR
    if endpoint == "hf":
        return OFFICIAL_HUB.rstrip("/"), "official"
    # auto
    if _tcp_open(OFFICIAL_HOST, OFFICIAL_PORT, CONNECT_TIMEOUT_SEC):
        return OFFICIAL_HUB.rstrip("/"), "official (auto)"
    return DEFAULT_MIRROR.rstrip("/"), f"mirror (auto, {DEFAULT_MIRROR})"


def dataset_resolve_base(base: str) -> str:
    return f"{base}/datasets/{DATASET}/resolve/{HF_REVISION}"




def tree_api_dataset(subtree: str, base: str) -> list[dict]:
    enc = urllib.parse.quote(subtree, safe="/")
    url = f"{base}/api/datasets/{DATASET}/tree/{HF_REVISION}/{enc}?recursive=true"
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.loads(r.read().decode())




def should_skip_test_img(path: str) -> bool:
    if not path.startswith("test_imgs/"):
        return False
    base = path.rsplit("/", 1)[-1]
    if "_show_" in base:
        return True
    if base.startswith("IMG_") and base.endswith(".jpg"):
        return False
    return True


def should_skip_test_video(path: str) -> bool:
    if not path.startswith("test_videos/"):
        return False
    return "_show_" in path


def collect_dataset_manifest(base: str) -> dict[str, int]:
    manifest: dict[str, int] = {}
    for sub in DATASET_SUBTREES:
        try:
            items = tree_api_dataset(sub, base)
        except Exception as e:
            raise RuntimeError(f"Cannot read the pinned resource tree {sub!r}: {e}") from e
        for it in items:
            if it.get("type") != "file":
                continue
            p = it["path"]
            if p.startswith("test_imgs/") and should_skip_test_img(p):
                continue
            if p.startswith("test_videos/") and should_skip_test_video(p):
                continue
            sz = int(it.get("size") or 0)
            if sz <= 0:
                continue
            manifest[p] = sz
    return manifest




def local_file_ok(dest: str, expected: int) -> bool:
    return os.path.isfile(dest) and os.path.getsize(dest) == expected


def wget_file(url: str, dest: str, expected_size: int | None = None) -> bool:
    os.makedirs(os.path.dirname(dest), mode=0o755, exist_ok=True)
    tmp = dest + ".part"
    if os.path.isfile(tmp):
        try:
            os.remove(tmp)
        except OSError:
            pass
    cmd = [
        "wget", "-q", "-c", "-O", tmp,
        "--tries=20", "--waitretry=5", "--timeout=60", "--read-timeout=120",
        url,
    ]
    try:
        subprocess.run(cmd, check=True)
        if not os.path.isfile(tmp):
            return False
        if os.path.getsize(tmp) <= 0 or (expected_size is not None and os.path.getsize(tmp) != expected_size):
            os.remove(tmp)
            return False
        os.replace(tmp, dest)
        return True
    except subprocess.CalledProcessError:
        if os.path.isfile(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        return False


def _download_task(rel: str, expected: int | None, url: str, root: str) -> bool:
    """expected None => only require nonempty file after download."""
    with _print_lock:
        print(f"[*] {rel}", flush=True)
    dest = os.path.join(root, rel)
    if expected is not None:
        if local_file_ok(dest, expected):
            return True
    else:
        if os.path.isfile(dest) and os.path.getsize(dest) > 0:
            return True
    # Replace an existing file only after a complete download has passed its size check.
    # 下载完成且大小校验通过后才替换已有文件。
    if not wget_file(url, dest, expected):
        return False
    if expected is not None:
        return local_file_ok(dest, expected)
    return os.path.isfile(dest) and os.path.getsize(dest) > 0


def run_downloads(
    jobs: list[tuple[str, int | None, str]],
    root: str,
    workers: int,
) -> tuple[int, int]:
    """Return (ok_count, fail_count). ok includes skipped-already-valid."""
    if not jobs:
        return 0, 0
    if workers < 1:
        workers = 1
    ok = fail = 0
    if workers == 1:
        for rel, exp, url in jobs:
            if _download_task(rel, exp, url, root):
                ok += 1
            else:
                fail += 1
        return ok, fail

    with ThreadPoolExecutor(max_workers=workers) as ex:
        future_map = {
            ex.submit(_download_task, rel, exp, url, root): rel for rel, exp, url in jobs
        }
        for fut in as_completed(future_map):
            rel = future_map[fut]
            try:
                if fut.result():
                    ok += 1
                else:
                    fail += 1
            except Exception as e:
                fail += 1
                with _print_lock:
                    print(f"[!] {rel}: {e}", file=sys.stderr, flush=True)
    return ok, fail


def verify_only(root: str, manifest: dict[str, int], label: str) -> tuple[int, int]:
    bad = 0
    ok = 0
    for rel, exp in sorted(manifest.items()):
        dest = os.path.join(root, rel)
        if local_file_ok(dest, exp):
            ok += 1
        else:
            bad += 1
            got = os.path.getsize(dest) if os.path.isfile(dest) else -1
            print(f"  {rel}  expected={exp}  local={got}")
    print(f"[{label}] {ok} ok, {bad} bad")
    return ok, bad


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Wget-based demo asset download (fallback when snapshot_download is unstable)."
    )
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument(
        "--endpoint",
        choices=("auto", "hf", "mirror"),
        default="auto",
        help="Hub API + file URLs: official (hf), China mirror, or TCP auto-detect (default).",
    )
    ap.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=DEFAULT_JOBS,
        metavar="N",
        help=f"Parallel wget workers (default {DEFAULT_JOBS}). Use 1 for strictly sequential.",
    )
    args = ap.parse_args()
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(root)
    base, label = hub_base(args.endpoint)
    print(f"[i] Hub base: {base} ({label})")
    print(f"[i] Parallel jobs: {args.jobs}")
    ds_base = dataset_resolve_base(base)
    ds = collect_dataset_manifest(base)
    if args.verify_only:
        b1 = verify_only(root, ds, "dataset")[1]
        if b1:
            return 1
        validate_selected_demo(root, list(ds))
        return 0

    jobs: list[tuple[str, int | None, str]] = []
    skipped = 0
    for rel, expected in sorted(ds.items()):
        dest = os.path.join(root, rel)
        if local_file_ok(dest, expected):
            skipped += 1
            continue
        url = f"{ds_base}/{urllib.parse.quote(rel, safe='/')}"
        jobs.append((rel, expected, url))

    print(f"[i] already complete (skipped): {skipped}, to download: {len(jobs)}")
    ok, fail = run_downloads(jobs, root, args.jobs)
    print(f"[i] downloaded ok={ok}, failed={fail} (skipped before queue: {skipped})")
    if fail:
        return 1
    validate_selected_demo(root, list(ds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
