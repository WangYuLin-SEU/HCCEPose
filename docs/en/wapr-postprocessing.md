# WAPR score filtering and rendered-mask NMS

The [WAPR inference example](rgbd-wapr.md) saves every estimated pose and score, then automatically applies a score gate and the [fixed public WAPR NMS implementation](https://github.com/WangYuLin-SEU/WAPR/blob/4e591be318a828c91980c58ef8bbdc34e6599881/wapr/suppression.py). Prediction and postprocessing run from one root script; a second command is not required.

## Prepare the same frame and models

Run `s4_p3_test_mi10_bin_picking_RGBD_wapr.py`. It keeps raw predictions in `outputs/wapr.json` and writes the automatic postprocessing report to `outputs/wapr_filtered.json`, using the same image, original-image intrinsics and object meshes. Raw records contain `object_id`, `instance_index`, `pose_4x4_m` and `score_6d`.

The script reads BOP meshes from `<dataset_root>/models/obj_XXXXXX.ply`. BOP vertices use millimeters, so `mesh_scale_to_m=1e-3` converts them to meters. Set it to `1` for meshes already in meters. Mesh coordinates and the saved pose must refer to the same original object frame; do not recenter the mesh afterward.

The score gate runs on CPU. When any record passes it, rendered-mask NMS requires the CUDA renderer from the [WAPR installation](installation.md#optional-wapr), in the same Python environment. The integrated postprocessing stage reuses the prepared meshes without running the pose networks again. Missing meshes or renderer failures raise errors rather than silently returning an unfiltered list; raw JSON remains available if NMS fails.

Source: [s4_p3_test_mi10_bin_picking_RGBD_wapr.py](https://github.com/WangYuLin-SEU/HCCEPose/blob/main/s4_p3_test_mi10_bin_picking_RGBD_wapr.py).

## Run prediction with automatic postprocessing

The root script uses bundled `sample_data/rgbd/000003_*` inputs and the models prepared on the [resource page](assets.md#download-one-rgb-d-example). Check its paths and editable thresholds, then run:

```bash
python s4_p3_test_mi10_bin_picking_RGBD_wapr.py
```

`Tester.predict` completes filtering and NMS automatically; the root script only reads inputs and saves the returned results. To compare another threshold on an existing raw file without rerunning prediction, use Tester's static method, without constructing a Tester or loading model weights:

```python
from HccePose.tester import Tester

# These paths and poses must belong to the same frame.
report = Tester.postprocess_saved_wapr(
    "data/demo-bin-picking",
    "data/test_imgs_RGBD/000003_rgb.png",
    "data/test_imgs_RGBD/000003_camera_matrix.json",
    "outputs/wapr.json",
    "outputs/wapr_filtered.json",
    wbps_threshold=0.75,
    nms_iou_threshold=0.5,
    device="cuda:0",
    mesh_scale_to_m=1e-3,
)
retained = report["retained_records"]
```

`Tester.postprocess_wapr(records, meshes_by_id, cam_K, image.shape, wbps_threshold, nms_iou_threshold, device)` handles existing in-memory records. `meshes_by_id` maps each qualifying BOP ID to its meter mesh. All three WAPR prediction modes use this same method inside `Tester`; no filtering or NMS implementation is kept in the entry script or refinement group.

## Two different thresholds

| Parameter | Example value | Behavior |
| --- | --- | --- |
| `score_threshold` in the script / `wbps_threshold` in Tester | `0.75` | Apply the WBPS score gate to the returned `score_6d`: keep `score_6d >= 0.75` before NMS. This is an application choice, not a probability cutoff calibrated for every object. Scores above 1 are preserved. |
| `nms_iou_threshold` | `0.5` | Within the same `object_id`, consider higher scores first and suppress a lower-ranked pose when rendered-mask IoU is strictly greater than 0.5. Equality is kept. |

The IoU value follows the [public WAPR example](https://github.com/WangYuLin-SEU/WAPR/blob/4e591be318a828c91980c58ef8bbdc34e6599881/examples/10_robi_zigzag.py). Both values are visible script variables for the application; they do not change WAPR's scientific defaults. Tune them for your objects and camera.

`wapr_mode` selects `estimate`, `hybrid` or `refine`. The same numeric thresholds apply by default, but different candidate groups can produce different WBPS values; the gate is not calibrated equally across modes. See [the mode definitions](refinement-api.md).

## What the NMS compares

For each score-qualified pose, WAPR renders that object's CAD silhouette using its estimated object-to-camera transform and the original image intrinsics. NMS compares these pose-dependent masks, not just the original detector boxes or input masks. Different object IDs use their own meshes and are processed separately.

The public greedy implementation gives earlier input rows priority when scores tie. A suppressed row does not suppress later rows. Output records preserve their original input order rather than being reordered by score. Empty projected masks are also removed by the backend.

Highly overlapping projections can belong to distinct objects, particularly under occlusion or at different depths. Silhouette IoU alone does not establish physical identity or 6D accuracy; inspect such cases and adjust the policy for your application. This example does not add a separate translation/rotation-distance test or cross-category suppression.

## Read the output

| Key | Meaning |
| --- | --- |
| `raw_records` | All input records, including their original scores and instance identities. |
| `decisions` | Per-input `input_index`, object/instance identity, score and selection status. |
| `kept_input_indices` | Indices into the raw list, in input order. |
| `retained_records` | Records that passed the score gate and survived rendered-mask NMS. |
| `score_threshold`, `nms_iou_threshold`, `nms_method` | The policy used for this output. |

A decision has status `kept`, `below_score_threshold` or `not_retained_by_mask_nms`. The last status includes empty projections as well as overlap suppression, so it is not automatically a confirmed duplicate. No skipped item is assigned a fake zero score.

The inference report also records `wapr_mode` and `skipped_instances`. In `refine`, the latter lists PnP failures or invalid seeds; these unestimated instances have no raw score. Reprocessing an existing raw file applies only filtering and NMS to the available records; it does not reconstruct skipped instances or change inference mode.

## Relation to the website example

The triangular-part viewer displays saved results. This script computes selection from the supplied JSON and models; it does not rewrite the viewer. The retained count depends on the current inputs and thresholds. Passing a score gate or NMS is also not ground-truth confirmation. Keep the raw file when comparing policies.
