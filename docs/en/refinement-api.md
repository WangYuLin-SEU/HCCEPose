# Refinement and RGB-D interfaces

RGB inference needs only color and intrinsics. Before adding depth, register pixels and check units, then choose [WAPR](rgbd-wapr.md) or [MegaPose](rgbd-megapose.md). Their entry points, initialization requirements and scores differ.

## Inputs and units

| Item | HccePose / MegaPose integration | WAPR mask integration |
| --- | --- | --- |
| Color | OpenCV BGR `uint8 H×W×3` | Adapter accepts BGR and converts to the public API’s RGB order. |
| Depth | Optional for MegaPose RGB; meters for RGB-D | Required finite, nonnegative `H×W` values in meters. |
| Camera | Original-image `3×3` pixel intrinsics | Same pixel intrinsics as color, depth and full-frame masks. |
| Model | BOP meshes registered in millimeters, converted internally | Both vertices and `diameter_m` in meters. |
| Initial pose | MegaPose refines HccePose hypotheses | This mask-estimation entry point does not accept HccePose pose seeds. |
| Translation output | `Tester` results remain millimeters | `pose_4x4` / example `pose_4x4_m` uses meters. |

Equal image dimensions do not prove registration. The adapter checks dimensions, but cannot verify sensor extrinsics or distortion correction; complete those before providing input.

## Restore crop masks to the original image

```python
from Refinement.wapr_public import restore_full_frame_masks

entry = predictions[obj_id]
masks = restore_full_frame_masks(entry, image_bgr.shape)
```

The result is a boolean `N×H×W` array in the object’s original crop-mask order. The helper inversely warps into the resized/padded canvas, removes padding, and restores original dimensions. Do not apply `Detect_Bbox_tfs_128` directly to the original image or pass `128×128` crop masks against full-frame depth.

Optional `mask_threshold=0.5` controls mask foreground only, not WAPR score acceptance. Mismatched mask and transform counts raise an error.

## Call the public WAPR API

```python
import numpy as np
import trimesh
from wapr.estimator import WAPREstimator
from Refinement.wapr_public import estimate_from_masks

# Prepare same-frame image_bgr, depth_m, cam_K, masks, and millimeter mesh/diameter.
mesh = trimesh.load(mesh_path, force='mesh', process=False)
mesh.apply_scale(1e-3)
diameter_m = diameter_mm * 1e-3
instances = [
    {'mesh': mesh, 'diameter_m': diameter_m, 'mask': mask}
    for mask in masks
]
estimator = WAPREstimator(device='cuda:0')
results = estimate_from_masks(estimator, image_bgr, depth_m, cam_K, instances)
for result in results:
    pose_m = np.asarray(result['pose_4x4'])
    score = float(result['score_6d'])
    print(pose_m, score)
```

Different objects can supply different meshes and diameters. Each list entry represents an instance; outputs align one-to-one with the list. If masks are already available, prepare your own full-frame masks and call this function without creating a HccePose `Tester`.

The adapter checks depth, intrinsics, mask dimensions and observed positive depth, and validates output count, pose shape and finite scores. An empty list returns `[]` without calling the backend. Empty masks or regions lacking positive depth raise instead of generating artificial zero scores.

`score_6d` is computed for the current input, not a probability or object label. Successful estimation, score acceptance and target confirmation are separate decisions. Applications define acceptance, suppression and ranking. See the [WAPR tutorial](rgbd-wapr.md) for the file-based entry point.

## Register and call MegaPose

```python
tester.register_megapose()
results_rgb = tester.predict(cam_K, image_bgr, [1], depth=None, use_megapose=True)
results_rgbd = tester.predict(cam_K, image_bgr, [1], depth=depth_m, use_megapose=True)
```

Omitting depth selects RGB; registered meter depth selects RGB-D. Registration uses all loader objects and millimeter meshes. Inference runs in a separate environment. See [results](results-api.md) for scores and valid instance indices.

## Repeated processing and resources

Reuse initialized HccePose and WAPR objects across frames. Importing `Refinement.wapr_public` does not install dependencies, download resources or initialize a GPU. Constructing the public backend may prepare its resources. Populate the caches described in [Installation](installation.md) before working offline.


For application-level score filtering and rendered-mask NMS after estimation, use the [optional postprocessing tutorial](wapr-postprocessing.md). This leaves the adapter return contract unchanged.
