# Refinement and RGB-D interfaces

RGB inference needs only color and intrinsics. Before adding depth, register pixels and check units, then choose [WAPR](rgbd-wapr.md) or [MegaPose](rgbd-megapose.md). Their entry points, initialization requirements and scores differ.

## Inputs and units

| Item | HccePose / MegaPose integration | WAPR integration |
| --- | --- | --- |
| Color | OpenCV BGR `uint8 H×W×3` | Adapter accepts BGR and converts to the public API’s RGB order. |
| Depth | Optional for MegaPose RGB; meters for RGB-D | Required finite, nonnegative `H×W` values in meters. |
| Camera | Original-image `3×3` pixel intrinsics | Same pixel intrinsics as color, depth and full-frame masks. |
| Model | BOP meshes registered in millimeters, converted internally | Both vertices and `diameter_m` in meters. |
| Initial pose | MegaPose refines HccePose hypotheses | None for `estimate`; optional per instance for `hybrid`; required for `refine`. |
| Translation output | `Tester` results remain millimeters | `Tester.Rts` uses millimeters; raw `pose_4x4` / `pose_4x4_m` uses meters. |

Equal image dimensions do not prove registration. The adapter checks dimensions, but cannot verify sensor extrinsics or distortion correction; complete those before providing input.

## Register and call WAPR through Tester

```python
# Optional eager registration; predict also registers automatically on first use.
tester.register_wapr()
results = tester.predict(cam_K, image_bgr, [1], depth=depth_m,
                         use_wapr=True, wapr_mode='estimate', wbps_threshold=0.75,
                         wapr_nms_iou_threshold=0.5)
entry = results.get(1)
if entry is not None:
    print(entry['Rts'], entry['wapr_scores'], entry['wapr_instance_indices'])
raw_records = results['wapr_postprocess']['raw_records']
```

Reuse the same `Tester` across frames. All three WAPR modes apply WBPS and per-object rendered-mask NMS, keeping masks, crops, confidences and poses aligned. `conf` remains YOLO confidence. An all-rejected object has zero rows; no detections gives an empty report without loading WAPR weights. Choose WAPR or MegaPose for each call. If NMS fails, prediction raises and `tester.last_wapr_raw_records` still contains this frame’s estimates; it is `None` until estimation completes.

| `wapr_mode` | Initialization and scoring |
| --- | --- |
| `estimate` (default) | Skip PnP. Use the unchanged public multi-instance mask estimator and its candidate group. |
| `hybrid` | Run PnP, append each valid seed to that instance's WAPR rotation/depth candidates, refine them and score the complete group once. A failed or invalid seed uses the public mask-estimation route instead. |
| `refine` | Run PnP, refine valid seeds through unmasked WAPR and SAPR, then score the original/refined pair. Generate no global rotation candidates. Failed or invalid PnP instances are listed in `skipped_instances` without a pose or fabricated score. |

With the fixed public recipe, estimation uses 12 candidates, a seeded hybrid group uses 13, and refinement scores 2. The original and refined pose are genuine stages of the same seed, not duplicate padding for WBPS. `within_group` selects a pose; `score_6d` uses the maximum `between_group` of that complete instance group, following the public score definition. Scores from separate runs are not merged to select a winner. Changing group composition can change scores, so check the explicit `0.75` gate for each mode.

Seeded modes compose the [fixed public refinement and WBPS components](https://github.com/WangYuLin-SEU/WAPR/blob/4e591be318a828c91980c58ef8bbdc34e6599881/wapr/estimator.py). They process scoring groups separately; the default estimator retains its public cross-instance batched path. Recipe iterations, crop ratios, rotation caps and backend selection are unchanged. Unsupported group sizes or engine profiles raise; no silent backend switch or scoring-group padding is added. PnP uses the selected `pnp_op`; combined solvers still require a matching checkpoint prior.

## Restore crop masks to the original image

```python
from Refinement.refinement_group import restore_full_frame_masks

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
from Refinement.refinement_group import estimate_from_masks

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

Reuse initialized HccePose and WAPR objects across frames. `Refinement.refinement_group` provides both `Refinement_MP` and `Refinement_WAPR`, along with shared RGB-D reading and mask restoration. Creating `Refinement_WAPR` does not load pose weights; its first nonempty inference call constructs the public backend, which may prepare resources. Populate the caches described in [Installation](installation.md) before working offline.


`Tester` owns score filtering and rendered-mask NMS and applies them automatically in `predict`. Its static `postprocess_wapr` and `postprocess_saved_wapr` methods can recheck existing records without loading inference weights. The lower-level `estimate_from_masks` still returns raw mask estimates. `Refinement_WAPR.inference_batch(..., mode='estimate')` also supports `hybrid` and `refine`; supply original-object meter `pose_4x4_m` seeds in the instance dictionaries for those modes. The refinement group only wraps backend estimation and refinement, returning raw results. See [postprocessing](wapr-postprocessing.md).
