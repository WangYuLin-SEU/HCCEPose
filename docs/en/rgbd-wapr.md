# WAPR integration

## From masks and depth to 3D estimation

Install WAPR [separately](installation.md#optional-wapr). This adapter targets the [published WAPR version](https://github.com/WangYuLin-SEU/WAPR/tree/4e591be318a828c91980c58ef8bbdc34e6599881). It restores instance masks predicted by HccePose for the current image to the full frame and calls `estimate_many_categories_many_instances`. This entry estimates poses from regions; it does not accept HccePose pose seeds.

The adapter converts the BGR image to RGB and passes it to WAPR with registered depth, intrinsics and instance data.


## Triangular-part RGB-D example

The example below pairs the triangular-part input with pose projections, measured depth points and CAD meshes. It contains 16 saved pose/score records, including 7 accepted entries; low-scoring and duplicate entries remain inspectable. Scores do not indicate ground-truth correctness or object-presence probability.

![Triangular-part RGB-D input and accepted pose projections](../assets/wapr-wbps-filtered-bin-picking.jpg)

The right image shows only the seven accepted instances. Different colors distinguish instances, and the labels retain their original score ranks.

## Linked RGB and 3D inspection

Select an image box, an entry below, or a 3D mesh to highlight it in both views. Choose Pan above the canvases and drag a colored axis, or Rotate and drag a graduated ring; scroll to zoom. Orbit retains drag-to-rotate and right/Shift-drag-to-pan. “Align with RGB” restores the calibrated view. On touchscreens, drag axes or rings with one finger and pinch to zoom; Orbit supports one-finger rotation and two-finger pan/zoom. Adding a second finger ends a handle drag and switches to two-finger navigation. These controls adjust only the 3D display; image projections, saved poses and scores remain unchanged. White points represent observed depth; green, red and amber surfaces indicate accepted, low-scoring and duplicate poses. A gold outline marks the selected mesh.

<iframe class="wapr-scene-frame" src="../assets/wapr-viewer/index.html?lang=en&amp;v=english-default-20261118" title="Interactive triangular-part RGB-D poses, scores and measured depth cloud" width="900" height="900" style="display:block;width:100%;max-width:100%;height:900px;box-sizing:border-box;margin:24px 0;border:0;border-radius:18px" loading="lazy" allow="fullscreen" allowfullscreen></iframe>

The viewer loads saved images, points, poses and scores; it does not run model inference. Switch among all scored entries, accepted entries and the selected entry; buttons for low-scoring and duplicate entries are disabled in accepted mode. Depth points and CAD meshes can be toggled independently. With CAD meshes enabled, use “CAD wireframe mode” to display them as wireframes.

## Per-region score comparison

Each tile pairs an original crop on the left with its pose projection on the right. Scores and selection states are drawn directly on the image. The compact four-by-four montage supports horizontal scrolling on narrow screens and click-to-enlarge. Green denotes accepted entries, amber duplicates and red low scores. Retained entries keep their original ranks.

<div class="wapr-montage">
<img src="../assets/wapr-region-crops/000003-contact-sheet-en.png" width="1960" height="1320" alt="Four-by-four triangular-part crop and pose-projection montage with scores and accepted, duplicate or low-score states">
</div>

Scores in this scene are saved display records. Running the interface below computes results for new input; it neither rewrites the website scene automatically nor guarantees the same detection counts or scores.

For an optional score gate and CAD rendered-mask NMS, follow [score filtering and NMS](wapr-postprocessing.md). The separate script preserves raw output and calls the published WAPR suppression function.

## Inputs and execution

Prepare a BGR image, registered depth, original-image intrinsics, meter meshes and meter diameters. The example explicitly converts published BOP millimeter models to meters.

Edit the paths, object IDs and depth conversion in `examples/inference/predict_wapr.py`, then run:

```bash
python examples/inference/predict_wapr.py
```

For the [example downloaded on the resource page](assets.md#download-one-rgb-d-example), you can also call this from the project root:

```python
from examples.inference.predict_wapr import main

# This frame stores raw depth in millimeters; check sensor units for your own data.
records = main("data/demo-bin-picking", "data/test_imgs_RGBD/000003_rgb.png",
               "data/test_imgs_RGBD/000003_depth.png", "data/test_imgs_RGBD/000003_camera_matrix.json",
               [1], "outputs/wapr.json", depth_scale_to_m=1e-3)
```

`outputs/wapr.json` preserves input instance order and contains object IDs, instance indices, `pose_4x4_m` (object to camera, translation in meters), and `score_6d` computed for the current input. A score is not a probability or object confirmation; define acceptance, deduplication and ranking policies for your application.

Request each object ID once and use IDs listed in the dataset’s `models_info.json`. Multiple detections of the same category produce separate records. If none of the requested objects is detected, the output is `[]`. Each crop mask needs its own transform. A count mismatch, an empty mask, or a mask with no positive depth raises an error.


## Multiple objects, instances and mask order

`object_ids` selects categories; list each ID once. Detections produce all instances for that category. The entry point restores masks in object-group and within-group order, prepares the matching meter mesh and diameter for each entry, then calls the public batch API. Output `object_id` and `instance_index` retain the input association.

Multiple regions do not necessarily represent distinct physical objects. Detections may miss targets, duplicate them or include background. A computable WAPR pose and score do not confirm object presence. Preserve raw results and instance identities, then check, suppress and rank according to application rules. Display count does not measure recall.

## Integrate into your Python application

```python
from Refinement.wapr_public import restore_full_frame_masks, estimate_from_masks

# predictions belongs to this frame; supply each object's own mesh and diameter_m.
masks = restore_full_frame_masks(predictions[obj_id], image_bgr.shape)
instances = [{'mesh': mesh, 'diameter_m': diameter_m, 'mask': mask} for mask in masks]
results_wapr = estimate_from_masks(estimator, image_bgr, depth_m, cam_K, instances)
```

Mesh vertices, diameter and depth must already be in meters. `estimator` is a WAPR object initialized following the public installation guide. This entry point estimates from the supplied masks without HccePose PnP seeds. See the [refinement API](refinement-api.md) for input checks, full-frame restoration and independently supplied masks.

## Output file and errors

| Field | Meaning |
| --- | --- |
| `object_id` | BOP object ID. |
| `instance_index` | Row within that object’s prediction group. |
| `pose_4x4_m` | Object-to-camera homogeneous transform with meter translation. |
| `score_6d` | WAPR score computed from this image, depth and mask. |

The JSON contains no fixed experimental scores and uses no ground-truth labels to generate them. Missing resources, invalid intrinsics and duplicate or unknown IDs raise before model loading. Shape mismatches, nonfinite depth, masks without valid depth and invalid outputs are not treated as successes. See [results](results-api.md) for reading and unit conversions.

Multi-frame applications can reuse `Tester` and the WAPR estimator. The file entry point is convenient initially but reinitializes models on repeated calls; organize high-frequency loops using the [API tutorial](refinement-api.md). Separate first-time resource preparation from inference timing.
