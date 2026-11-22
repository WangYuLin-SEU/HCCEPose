# Utilities and advanced interfaces

After running inference with [`Tester`](tester-api.md), use these helpers to save images and print stage timings. The lower-level interfaces below provide access to the network and checkpoints. See the [results API](results-api.md) for output shapes and units.

## Save visualization artifacts

```python
from HccePose.test_script_utils import save_visual_artifacts

save_visual_artifacts(artifacts, enabled=True)
```

`artifacts` is an iterable of `(path, image)` pairs. Images use OpenCV BGR `H × W × 3`; use uint8 images when saving the tester outputs. With `enabled=False`, the helper returns `[]` without writing. It skips `None` paths/images, creates parent directories and calls `cv2.imwrite()`.

```python
# output_dir is the chosen local output folder.
import os

saved_paths = save_visual_artifacts([
    (os.path.join(output_dir, 'pose.jpg'), results.get('show_6D_vis2')),
    (os.path.join(output_dir, 'detection.jpg'), results.get('show_2D_results')),
])
```

Existing files at the same path may be overwritten. The helper records each attempted path without checking the return value of `cv2.imwrite()`. Check the destination files when you need to confirm that writing succeeded.

Source: [`save_visual_artifacts`](https://github.com/WangYuLin-SEU/HCCEPose/blob/main/HccePose/test_script_utils.py).

## Print stage times

```python
from HccePose.test_script_utils import print_stage_time_breakdown

print_stage_time_breakdown(results_dict, enabled=False, prefix='')
```

Set `enabled=True` to print `time`, stage seconds and percentages. It returns `None`. Missing or malformed timing dictionaries print an unavailable message. Percentages use `results_dict['time']`; visualization can fall outside that interval, so percentages need not sum to 100%. See [timing definitions](results-api.md).

Source: [`print_stage_time_breakdown`](https://github.com/WangYuLin-SEU/HCCEPose/blob/main/HccePose/test_script_utils.py).

## Lower-level visualization

```python
vis_rgb_mask_Coord(
    rgb_c, pred_mask, pred_front_code, pred_back_code, img_path=None,
)

vis_rgb_mask_Coord_origin(
    cam_K, obj_ids_l, obj_ids_all, BBox_3d_l, Rts_l, conf_s,
    rgb_c, pred_mask, pred_front_code, pred_back_code, img_path=None,
)
```

These functions live in `HccePose.visualization`. The first returns a crop/code canvas; the second returns `(combined_canvas, compact_canvas)` with model box projections. The inputs are normalized BGR tensors plus code/mask tensors. The second function requires masks and codes already warped into the image canvas and poses in the same model coordinate convention as `BBox_3d_l`.

`vis_rgb_mask_Coord` returns the visualization and also writes it to disk when you supply `img_path`. `vis_rgb_mask_Coord_origin` returns two images and does not use its `img_path` argument. For normal inference, let `Tester` prepare the inputs and pass its `show_*` images to the saving helper above.

Sources: [crop visualization](https://github.com/WangYuLin-SEU/HCCEPose/blob/main/HccePose/visualization.py), [full-canvas visualization](https://github.com/WangYuLin-SEU/HCCEPose/blob/main/HccePose/visualization.py).

## Network and checkpoints

```python
from HccePose.network_model import HccePose_BF_Net

HccePose_BF_Net(
    efficientnet_key=None,
    input_channels=3,
    min_xyz=None,
    size_xyz=None,
)

network.inference_batch(inputs, Bbox, thershold=0.5)
```

`inputs` is a normalized BGR crop tensor, typically `N × 3 × 256 × 256`. `Bbox` is a tensor `N × 4` containing `[x, y, width, height]` in the associated image canvas. `min_xyz` and `size_xyz` must describe the object's axis-aligned model bounds and be compatible with the tensor device.

Use the API spelling **`thershold`** to threshold the sigmoid mask and code values. The call returns dense predictions, decoded model coordinates and logits. Detection and PnP are separate steps for computing the final `Rts`. See [result field shapes](results-api.md). `efficientnet_key=None` selects ResNet. Other values select EfficientNet.

Checkpoint helpers are also in `HccePose.network_model`:

```python
get_checkpoint(path)
load_checkpoint(
    check_point_path, net, optimizer=None, local_rank=0, CUDA_DEVICE='0',
)
```

Both paths refer to a checkpoint **directory**. `get_checkpoint()` parses score prefixes before `step` and returns the highest-scoring file path. `load_checkpoint()` loads CUDA weights and optional optimizer state, returning `best_score`, `iteration_step`, and `keypoints_`. Supplying an optimizer requires restoring its state; only a missing or empty directory starts a new training run. An existing corrupt or incompatible checkpoint stops restoration with its path and cause. Older weights may omit keypoints_, but an empty prior supports only PnP modes that do not require it. Use architecture-compatible weights; keep notes and obsolete checkpoints out of `best_score/`.

Sources: [network constructor](https://github.com/WangYuLin-SEU/HCCEPose/blob/main/HccePose/network_model.py), [batch inference](https://github.com/WangYuLin-SEU/HCCEPose/blob/main/HccePose/network_model.py), [checkpoint selection](https://github.com/WangYuLin-SEU/HCCEPose/blob/main/HccePose/network_model.py), [checkpoint loading](https://github.com/WangYuLin-SEU/HCCEPose/blob/main/HccePose/network_model.py).

## Training and evaluation boundaries

Training and evaluation have separate Python scripts. Set the parameters in the relevant script. There is no shared `train()` or `evaluate()` entry point. The scripts below live in the [repository root](installation.md#public-training-and-rendering-source).

| Script | Purpose |
| --- | --- |
| `s4_p2_train_bf_pbr.py` | Per-object training. `ide_debug=True` selects single-GPU mode; `False` enables DDP. The dataset path is a placeholder requiring editing. |
| `s4_p2_test_bf_pbr.py` | Evaluates the loaded network and PnP on cropped PBR images. |
| `s4_p2_test_bf_pbr_bop_challenge.py` | Default LM-O example with external GDRNPP detections; writes segmentation JSON and pose CSV. It does not invoke the official scorer or upload results. |

The training script sets `total_iteration=50001` and `batch_size=24`. To monitor training, it samples `ratio=0.01` from the same `train_pbr` directory. Use the appropriate evaluation script for a held-out test set.

The BOP script uses `efficientnet_key=1`, the listed LM-O object IDs and the dataset's detection JSON. It writes `seg2d_<dataset>.json` and `det6d_<dataset>.csv` for subsequent scoring. Both files use a `time=-1` placeholder. Measure runtime separately.

Sources: [training settings](https://github.com/WangYuLin-SEU/HCCEPose/blob/main/s4_p2_train_bf_pbr.py), [PBR validation](https://github.com/WangYuLin-SEU/HCCEPose/blob/main/s4_p2_test_bf_pbr.py), [challenge setup](https://github.com/WangYuLin-SEU/HCCEPose/blob/main/s4_p2_test_bf_pbr_bop_challenge.py), [challenge output](https://github.com/WangYuLin-SEU/HCCEPose/blob/main/s4_p2_test_bf_pbr_bop_challenge.py).
