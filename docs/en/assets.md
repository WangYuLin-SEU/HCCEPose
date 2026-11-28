# Models and inputs

## Matching resources

Meshes, model metadata, detector class order and per-object pose weights must match. Published resources are available from the [HCCEPose resource page](https://huggingface.co/datasets/SEU-WYL/HccePose). A small set of captures is bundled under `sample_data/`; download model weights and training data separately.

`dataset_root` follows the published loader layout:

```text
objects/
  models/models_info.json
  models/obj_000001.ply
  yolo11/train_obj_s/detection/obj_s/yolo11-detection-obj_s.pt
  HccePose/obj_01/best_score/
```

`Tester` loads every object listed in `models_info.json` during initialization, so prepare pose weights for each object and keep the object order in that file consistent with the detector’s class order. The object ID list passed to `predict` only selects objects for the current prediction; it does not limit which models are loaded at initialization.

Set the dataset path, object IDs and image paths in the entry script. Use detector and pose checkpoints from a trusted source that permits your intended use. Check their hashes against the provider’s records before loading them.

Bundled inputs are grouped under `sample_data/images/`, `sample_data/rgbd/` and `sample_data/videos/`; `tests/` contains code tests. The root `s4_p3_*.py` examples use those bundled directories. Hugging Face downloads retain their upstream names (`test_imgs/`, `test_imgs_RGBD/`, `test_videos/`), so point the example at the downloaded input directory when using additional files. The recipe below explicitly downloads to `data/`.

## Download one RGB-D example

After [installation](installation.md), run the minimal downloader from the project root:

```bash
python scripts/download_hf_assets.py --preset demo --dest data --endpoint hf
```

It fetches and validates the seven files below and writes a separate camera matrix. If the official endpoint is unreachable, retry with `--endpoint mirror`; both routes request the same fixed revision and keep validation enabled. If a service returns an error, keep the existing files and retry after connectivity is restored.

If all seven files have already been prepared on a connected machine, copy the `data/` layout to the target and validate it offline:

```bash
HF_HUB_OFFLINE=1 python scripts/download_hf_assets.py --preset demo --dest data --endpoint hf
```

Offline mode does not waive the file-size, checkpoint-directory or weight-digest checks. Missing or damaged files still stop the command.

The equivalent explicit recipe below can be saved as a Python script at the project root. Assets go under `data/`, preserving the Hugging Face layout. Use `data/demo-bin-picking/` as `dataset_root`; there is no need to rename it to `objects/`.

```python
import hashlib
import json
import os
from huggingface_hub import snapshot_download

project_dir = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(project_dir, "data")

# Sizes from the fixed public revision; verify both weights against their LFS SHA-256.
checkpoint_relative_path = "demo-bin-picking/HccePose/obj_01/best_score/0_9837step824500"
required_files = {
    "demo-bin-picking/models/models_info.json": (941, None),
    "demo-bin-picking/models/obj_000001.ply": (128911, None),
    checkpoint_relative_path: (
        349827430, "3308469a5082feb98181286026003e0b00ea8a349b8f0921006873a177dd4c70"),
    "demo-bin-picking/yolo11/train_obj_s/detection/obj_s/yolo11-detection-obj_s.pt": (
        114453888, "5a958cf1d90d3529f02ba1021aab164a369316f187f98fb01b151c484d3cfd13"),
    "test_imgs_RGBD/000003_rgb.png": (208215, None),
    "test_imgs_RGBD/000003_depth.png": (59239, None),
    "test_imgs_RGBD/000003_camK.json": (184, None),
}

# Share one seven-file list between downloading and validation; omit training ZIPs and symmetry helpers.
snapshot_download(
    repo_id="SEU-WYL/HccePose",
    repo_type="dataset",
    revision="4e482e540314622981ccd4fce5beddac7f9d3e1d",
    local_dir=data_root,
    allow_patterns=list(required_files),
    endpoint="https://huggingface.co",
    token=False,
)

# The loader scans every regular file here; report extras without deleting them.
checkpoint_path = os.path.join(data_root, *checkpoint_relative_path.split("/"))
checkpoint_dir = os.path.dirname(checkpoint_path)
extra_files = [os.path.join(checkpoint_dir, name) for name in os.listdir(checkpoint_dir)
               if os.path.isfile(os.path.join(checkpoint_dir, name))
               and name != os.path.basename(checkpoint_path)]
if extra_files:
    raise RuntimeError("Unexpected checkpoint files: " + ", ".join(sorted(extra_files)))

# The SDK may return a local folder when offline; validate before converting the camera.
for relative_path, (expected_size, expected_sha256) in required_files.items():
    path = os.path.join(data_root, *relative_path.split("/"))
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    actual_size = os.path.getsize(path)
    if actual_size != expected_size:
        raise RuntimeError(f"File size mismatch: {path}; expected {expected_size}, got {actual_size}")
    if expected_sha256 is not None:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise RuntimeError(f"SHA-256 mismatch: {path}")

# Use this frame's intrinsics; convert the field-based JSON to the entry's 3×3 array.
capture_dir = os.path.join(data_root, "test_imgs_RGBD")
with open(os.path.join(capture_dir, "000003_camK.json"), encoding="utf-8") as handle:
    camera = json.load(handle)
cam_K = [[camera["fx"], 0, camera["cx"]],
         [0, camera["fy"], camera["cy"]], [0, 0, 1]]
camera_path = os.path.join(capture_dir, "000003_camera_matrix.json")
with open(camera_path, "w", encoding="utf-8") as handle:
    json.dump(cam_K, handle, indent=2)
```

This call accesses public resources without login. `local_dir` determines where files are stored; `data/.cache/huggingface/` holds download metadata. Reruns reuse existing files and validate them again. The sizes and weight digests refer to the fixed HF revision above. Exactly seven files are selected, excluding training ZIPs and symmetry helpers. If validation fails, inspect the reported file before downloading it again; the example does not delete existing files. Keep only the specified extensionless checkpoint in `best_score/`: do not rename or extract it, or add documentation files there.

After confirming a file is damaged, add `force_download=True` to the download call to fetch it again. Inspect any extra checkpoints or documentation files before moving them out of `best_score/` yourself.

The RGB and depth files are `data/test_imgs_RGBD/000003_rgb.png` and `data/test_imgs_RGBD/000003_depth.png`. The converted camera array is `data/test_imgs_RGBD/000003_camera_matrix.json`. Do not substitute `demo-bin-picking/camera.json` for this frame: it has different intrinsics and resolution. Continue with [your first prediction](quickstart.md) or [WAPR integration](rgbd-wapr.md).

The existing download tool also provides a minimal preset:

```bash
python scripts/download_hf_assets.py --preset demo --dest data --endpoint hf
```

It uses the same fixed seven-file manifest, validates it and writes the separate camera matrix. Other presets retain the legacy project-root layout unless `--dest` is supplied; the wget fallback also writes that legacy root layout. See `python scripts/download_hf_assets.py --help` before selecting larger training presets.

## Preparing RGB inference offline

The default network initializes a torchvision ResNet-34 before it loads each object's HccePose checkpoint. On a connected machine, preload the backbone into a fresh Torch Hub cache and print its checkpoint directory:

```bash
python -c "from torchvision.models import ResNet34_Weights; ResNet34_Weights.IMAGENET1K_V1.get_state_dict(check_hash=True)"
python -c "from pathlib import Path; import torch; print(Path(torch.hub.get_dir()) / 'checkpoints')"
```

Copy `resnet34-b627a593.pth` into the checkpoint directory printed by the second command **on the target machine**. Torchvision checks the published SHA-256 prefix `b627a593` on download. Compare the full SHA-256 digest on both machines after copying. [Torchvision ResNet-34 weights](https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.resnet34.html)

## Images and camera

Color images use OpenCV BGR order. `camera.json` is a 3×3 array of pixel intrinsics matching the original image resolution.

Depth must already be registered to the color image. WAPR expects meters; explicitly multiply integer millimeter depth by `1e-3`. Identical dimensions alone do not establish registration.

## Other public data and training resources

The seven-file manifest above is sufficient for the image and WAPR examples. Select additional resources by task rather than downloading the complete dataset repository.

| Task | Public directory or file |
| --- | --- |
| Bin-picking models, weights and training data | [demo-bin-picking](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/4e482e540314622981ccd4fce5beddac7f9d3e1d/demo-bin-picking) |
| Textured-object models and weights | [demo-tex-objs](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/4e482e540314622981ccd4fce5beddac7f9d3e1d/demo-tex-objs) |
| Image inputs | [test_imgs](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/4e482e540314622981ccd4fce5beddac7f9d3e1d/test_imgs) |
| Video inputs | [test_videos](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/4e482e540314622981ccd4fce5beddac7f9d3e1d/test_videos) |
| Registered RGB-D frames | [test_imgs_RGBD](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/4e482e540314622981ccd4fce5beddac7f9d3e1d/test_imgs_RGBD) |
| Textures | [cc0textures-512.zip](https://huggingface.co/datasets/SEU-WYL/HccePose/blob/4e482e540314622981ccd4fce5beddac7f9d3e1d/cc0textures-512.zip) |
| Seven BOP dataset weights | [BOP evaluation and weight index](bop-evaluation.md) |

Hugging Face `local_dir` retains upstream paths. After downloading `demo-tex-objs/`, that directory is the dataset root; do not nest another copy of its name. Training ZIP files and extensionless checkpoints are different: extract ZIPs following the training tutorial, while keeping checkpoint filenames intact for direct loading.

For your own data, expose one external root and derive mesh, scene and label paths from it. Models use millimeters. Convert depth according to capture metadata rather than guessing from the extension. See [object preparation](object-preparation.md) for the full new-model workflow.
