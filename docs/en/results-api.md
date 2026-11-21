# Results and coordinates

## HccePose

`Tester.predict` groups detections by object ID. Within a group, `pnp_success[i]` records whether HccePose’s PnP solver succeeded for detection `i`. Check this boolean before using an RGB pose. With too few correspondences the solver can return the identity matrix and `pnp_success[i]` is `False`. When optional refinement is enabled, the flag still describes the initial PnP solve.

`Rts[i]` is a 4×4 object-to-camera transform with translation in millimeters. The mask `pred_mask[i]` is in crop coordinates. Use the existing helper to invert the crop transform, remove padding and restore the original resolution:

```python
from Refinement.wapr_public import restore_full_frame_masks

mask_i = restore_full_frame_masks(results[obj_id], image_bgr.shape)[i]
```

`image_bgr` is the original image passed to `predict`, and `i` is the instance index within that object group. The helper uses the matching `Detect_Bbox_tfs_128` and returns boolean masks at the original image size.

## WAPR

`pose_4x4_m` is an object-to-camera pose with meter translations. The adapter’s `score_6d` is computed by WAPR for the current input. It differs from YOLO confidence and is not guaranteed to lie in a probability interval.

Results preserve input instance order. The saved `instance_index` is local to an object group, not a detector-global box ID or ground-truth label.

Coordinate and channel-order conversions occur explicitly at the adapter boundary. The original Tester's millimeter convention is unchanged.

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
| `pnp_success` | NumPy bool `N` | Initial HccePose PnP success status. |
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
| `time_dict` | Wall-clock timings for `yolo`, `hccepose`, `megapose`, `visualization`, and `other`. |

The four base images require `hccepose_vis=True`. They use BGR and may combine multiple panels, so their dimensions need not equal the original image. `time_dict` includes visualization, whereas `time` ends before it. Their totals need not agree, and neither is a synchronized GPU benchmark.

When MegaPose returns scores, `Rts_hccepose` preserves the initial poses and `Rts` updates successful poses. `megapose_scores` aligns with `megapose_valid_pose_ids`, not necessarily every instance row. `pnp_success` still describes only initial PnP. See the [refinement API](refinement-api.md).

```python
# Convert translations explicitly to meters; rotations are unchanged.
entry = results.get(1)
if entry is not None:
    translations_m = entry['Rts'][:, :3, 3] * 1e-3
```
