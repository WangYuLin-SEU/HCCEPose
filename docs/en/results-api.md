# Results and coordinates

## HccePose

`Tester.predict` groups detections by object ID. Within a group, `pnp_success[i]` records whether HccePose’s PnP solver succeeded for detection `i`. Check this boolean before using an RGB pose. With too few correspondences the solver can return the identity matrix and `pnp_success[i]` is `False`. When optional refinement is enabled, the flag still describes the initial PnP solve.

`Rts[i]` is a 4×4 object-to-camera transform with translation in millimeters. The mask `pred_mask[i]` is in crop coordinates. Use the existing helper to invert the crop transform, remove padding and restore the original resolution:

```python
from Refinement.refinement_group import restore_full_frame_masks

mask_i = restore_full_frame_masks(results[obj_id], image_bgr.shape)[i]
```

`image_bgr` is the original image passed to `predict`, and `i` is the instance index within that object group. The helper uses the matching `Detect_Bbox_tfs_128` and returns boolean masks at the original image size.

## WAPR

`pose_4x4_m` is an object-to-camera pose with meter translations. The adapter’s `score_6d` is computed by WAPR for the current input. It differs from YOLO confidence and is not guaranteed to lie in a probability interval.

Results preserve input instance order. The saved `instance_index` is local to an object group, not a detector-global box ID or ground-truth label.

Coordinate and channel-order conversions occur explicitly at the adapter boundary. The original Tester's millimeter convention is unchanged.

With `Tester.predict(..., use_wapr=True)`, `Rts` contains only retained poses in millimeters. `wapr_scores` holds their WBPS values, and `wapr_instance_indices` maps each retained row to the original object-local detection index. Masks, transforms, crops, logits and YOLO `conf` are filtered by the same indices. Only `wapr_mode='estimate'` skips PnP and omits `pnp_success`. In `hybrid` and `refine`, the filtered `pnp_success` and `Rts_hccepose` still describe the original PnP solve. An object whose detections all fail filtering or are skipped has empty arrays, including `Rts` of shape `0×4×4`.

Each object also exposes unfiltered `wapr_raw_records` and `wapr_decisions`. The top-level `wapr_postprocess` report contains all original meter poses and scores, decisions and retained records across objects. Its `kept_input_indices` addresses the report’s raw list; it is distinct from the object-local `wapr_instance_indices`. The raw list follows requested object order and original detection order within each object.

`wapr_mode` identifies the selected mode. `wapr_pose_sources` aligns with retained rows: `wapr` denotes a global WAPR candidate, `pnp_refined` a refined PnP seed, and `pnp` the original seed retained by the two-pose refinement scoring group. Raw records preserve this as `hypothesis_source`; seeded groups also expose `hypothesis_index` and `hypothesis_count`. In PnP modes they record `pnp_success` and, when valid, `pnp_pose_4x4_m` before refinement.

`refine` does not estimate failed or invalid PnP instances. The report lists them separately in `skipped_instances`, with object ID, original instance index and `pnp_failed` or `invalid_pnp_pose`; they have no fabricated score and no raw pose record. Each object's `wapr_skipped_instances` is its subset. `hybrid` may retain a WAPR result even when its initial `pnp_success` is false; that flag does not describe the final WAPR result.

## Per-object fields

Shapes below use the default `crop_size=256`; `N` is the instance count for that object. Tensors may reside on the GPU; call `detach().cpu()` before NumPy conversion.

| Field | Type / shape | Meaning |
| --- | --- | --- |
| `Rts` | NumPy `N × 4 × 4` | Final poses; translation in mm. Refinement may replace the initial poses. |
| `pred_mask` | torch `N × 128 × 128` | Binary mask in network crop space. |
| `coord_2d_image` | torch `N × 128 × 128 × 2` | Pixel coordinates in the resized/padded input canvas. |
| `pred_front_code_obj`, `pred_back_code_obj` | torch `N × 128 × 128 × 3` | Decoded front/back coordinates in model space, using model units. |
| `pred_front_code`, `pred_back_code` | torch `N × 128 × 128 × 3` | Decoded coordinates normalized to `[0, 1]`. |
| `pred_front_code_raw`, `pred_back_code_raw` | torch `N × 128 × 128 × 24` | Display values computed as `(logits + 1) / 2`, then clamped to `[0, 1]`; these are not unmodified logits. |
| `pred_mask_logits` | torch `N × 1 × 128 × 128` | Mask logits. |
| `pred_front_back_code_logits` | torch `N × 48 × 128 × 128` | Front/back code logits. |
| `Detect_Bbox_tfs_128` | torch `N × 3 × 3` | Homographies from the processed canvas into network output crops. |
| `crop_rgbs` | torch `N × 3 × 256 × 256` | BGR crops normalized with BGR-permuted ImageNet statistics. |
| `pnp_success` | NumPy bool `N` | Initial HccePose PnP status; absent only in WAPR `estimate` mode. |
| `Rts_hccepose` | NumPy `N × 4 × 4` | Original PnP poses in mm for retained `hybrid` / `refine` rows; also used by MegaPose when it returns scores. |
| `wapr_scores` | NumPy `N` | WAPR-only: retained WBPS values, not YOLO confidence. |
| `wapr_instance_indices` | NumPy int `N` | WAPR-only: original object-local detection row for each retained instance. |
| `conf` | torch `N` | YOLO detection score, distinct from WAPR scores. |

## Visualization and timing

| Top-level key | Meaning |
| --- | --- |
| `show_2D_results` | YOLO detection boxes. |
| `show_6D_vis0` | Instance crops, masks and front/back code montage. |
| `show_6D_vis1` | Combined coordinate and pose views of the processed image. |
| `show_6D_vis2` | Compact full-scene pose view. |
| `show_megapose` | Present when requested MegaPose debug images are generated. |
| `time` | Seconds from before detection through pose computation; excludes initialization and subsequent visualization. |
| `time_dict` | Wall-clock timings for `yolo`, `hccepose`, `megapose`, `wapr`, `visualization`, and `other`. |

The `wapr` stage includes mask restoration, estimation and NMS. Its first call also includes deferred WAPR registration/resource loading; warm up before measuring repeated-frame performance.

The four base images require `hccepose_vis=True`. They use BGR and may combine multiple panels, so their dimensions need not equal the original image. `time_dict` includes visualization, whereas `time` ends before it. Their totals need not agree, and neither is a synchronized GPU benchmark.

When MegaPose returns scores, `Rts_hccepose` preserves the initial poses and `Rts` updates successful poses. `megapose_scores` aligns with `megapose_valid_pose_ids`, not necessarily every instance row. `pnp_success` still describes only initial PnP. See the [refinement API](refinement-api.md).

```python
# Convert translations explicitly to meters; rotations are unchanged.
entry = results.get(1)
if entry is not None:
    translations_m = entry['Rts'][:, :3, 3] * 1e-3
```
