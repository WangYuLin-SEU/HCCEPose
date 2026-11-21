# WAPR score filtering and rendered-mask NMS

The [WAPR inference example](rgbd-wapr.md) saves every estimated pose and score. This optional second step applies a score gate, then removes overlapping pose projections with the [fixed public WAPR NMS implementation](https://github.com/WangYuLin-SEU/WAPR/blob/4e591be318a828c91980c58ef8bbdc34e6599881/wapr/suppression.py). It does not change the adapter or rerun the pose networks.

## Prepare the same frame and models

First run `examples/inference/predict_wapr.py` and keep its `outputs/wapr.json`. Supply the same image, original-image intrinsics and object meshes when postprocessing. The JSON must contain `object_id`, `instance_index`, `pose_4x4_m` and `score_6d` for each input instance.

The script reads BOP meshes from `<dataset_root>/models/obj_XXXXXX.ply`. BOP vertices use millimeters, so `mesh_scale_to_m=1e-3` converts them to meters. Set it to `1` for meshes already in meters. Mesh coordinates and the saved pose must refer to the same original object frame; do not recenter the mesh afterward.

The score gate runs on CPU. When any record passes it, rendered-mask NMS requires the CUDA renderer from the [WAPR installation](installation.md#optional-wapr), in the same Python environment. No pose-network weights are loaded again by this script. Missing meshes or renderer failures raise errors rather than silently returning an unfiltered list.

The script is available in the public source: [postprocess_wapr.py](https://github.com/WangYuLin-SEU/HCCEPose/blob/main/examples/inference/postprocess_wapr.py).

## Run the optional script

The editable values at the bottom of `examples/inference/postprocess_wapr.py` use the [downloaded RGB-D example](assets.md#download-one-rgb-d-example). Check the paths, then run from the project root:

```bash
python examples/inference/postprocess_wapr.py
```

The raw JSON remains unchanged. The result is written separately to `outputs/wapr_filtered.json`. For Python applications:

```python
from examples.inference.postprocess_wapr import main

# These paths and poses must belong to the same frame.
report = main(
    "data/demo-bin-picking",
    "data/test_imgs_RGBD/000003_rgb.png",
    "data/test_imgs_RGBD/000003_camera_matrix.json",
    "outputs/wapr.json",
    "outputs/wapr_filtered.json",
    score_threshold=0.75,
    nms_iou_threshold=0.5,
    device="cuda:0",
    mesh_scale_to_m=1e-3,
)
retained = report["retained_records"]
```

You can also call `filter_records(records, meshes_by_id, cam_K, image.shape, score_threshold, nms_iou_threshold, device)` on in-memory records. `meshes_by_id` maps each qualifying BOP object ID to its meter mesh. Records keep the same schema as `predict_wapr.py`.

## Two different thresholds

| Parameter | Example value | Behavior |
| --- | --- | --- |
| `score_threshold` | `0.75` | Apply the WBPS score gate to the returned `score_6d`: keep `score_6d >= 0.75` before NMS. This is an application choice, not a probability cutoff calibrated for every object. Scores above 1 are preserved. |
| `nms_iou_threshold` | `0.5` | Within the same `object_id`, consider higher scores first and suppress a lower-ranked pose when rendered-mask IoU is strictly greater than 0.5. Equality is kept. |

The IoU value follows the [public WAPR example](https://github.com/WangYuLin-SEU/WAPR/blob/4e591be318a828c91980c58ef8bbdc34e6599881/examples/10_robi_zigzag.py). Both values are visible script variables for the application; they do not change WAPR's scientific defaults. Tune them for your objects and camera.

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

## Relation to the website example

The triangular-part viewer displays saved results. This script computes selection from the supplied JSON and models; it does not rewrite the viewer. The retained count depends on the current inputs and thresholds. Passing a score gate or NMS is also not ground-truth confirmation. Keep the raw file when comparing policies.
