# Tester API

`HccePose.tester.Tester` processes one OpenCV image or video frame. A detector proposes regions for known objects, per-object networks predict masks and front/back coordinates, and PnP estimates poses. Each object category may have multiple instances.

## Constructor

```python
from HccePose.tester import Tester

tester = Tester(
    bop_dataset_item,
    show_op=True,
    hccepose_vis=None,
    CUDA_DEVICE='0',
    top_K=None,
    crop_size=256,
    efficientnet_key=None,
    megapose_model_name_rgb='megapose-1.0-RGB-multi-hypothesis',
    megapose_model_name_rgbd='megapose-1.0-RGBD',
    hccepose_acceleration='pytorch',
    hccepose_acceleration_cache_dir=None,
)
```

| Argument | Meaning |
| --- | --- |
| `bop_dataset_item` | [BOP loader](data-api.md) containing models, the asset root and object ordering. |
| `hccepose_vis` | Overrides `show_op` when not `None`; controls the four base visualization fields. |
| `CUDA_DEVICE` | CUDA index as a string, such as `'0'`. The current inference pipeline requires CUDA. |
| `top_K` | Reserved; does not limit detections or poses. |
| `crop_size` | Defaults to `256`; must match the network and visualization layout, rather than the original image size. |
| `efficientnet_key` | `None` selects ResNet; other values select EfficientNet. Must match the checkpoint. |
| `megapose_model_name_rgb`, `megapose_model_name_rgbd` | Model names used when registering MegaPose. |
| `hccepose_acceleration` | `'pytorch'`, `'onnx'` or `'tensorrt'`; selects the network execution backend. |
| `hccepose_acceleration_cache_dir` | Export/backend cache location, or `None` for per-object defaults. |

The compatibility arguments `acceleration` and `acceleration_cache_dir` remain available. Non-`None` values override the corresponding HccePose-specific settings. See [acceleration](acceleration.md).

Initialization loads every object listed in `models_info.json`, rather than only the IDs later requested for prediction. Each needs a matching checkpoint. YOLO category indices map through the loader’s `obj_id_list`, whose order must match detector training.

A CPU fallback message during initialization does not imply full CPU support: object-bound tensors and checkpoint loading still use CUDA paths. Invalid or incompatible checkpoints report the object, path and cause.

## Prediction

```python
results = tester.predict(
    cam_K, image_bgr, object_ids,
    confidence_threshold=0.75,
    conf=0.75,
    iou=0.50,
    max_det=200,
    pnp_op='ransac+comb',
    depth=None,
    use_megapose=False,
    megapose_vis=False,
    megapose_vis_stages=None,
)
```

| Input | Contract |
| --- | --- |
| `cam_K` | NumPy `3×3` original-image intrinsics in pixels, matching the input dimensions. |
| `image_bgr` | NumPy `H×W×3`, `uint8`, in OpenCV BGR order. |
| `object_ids` | BOP object IDs, not YOLO category indices; each may return multiple instances. |
| `conf` | Detection confidence threshold passed to YOLO. |
| `confidence_threshold` | Controls only drawing in the 2D detection image; does not filter returned poses. |
| `iou`, `max_det` | Detector suppression setting and maximum number of detections passed to YOLO. |
| `pnp_op` | `epnp`, `ransac`, `ransac+vvs`, `ransac+comb` or `ransac+vvs+comb`. |
| `depth` | Optional registered `H×W` depth in meters; enables the MegaPose RGB-D model. |
| `use_megapose` | Requests refinement; registers MegaPose on first use if needed. |
| `megapose_vis`, `megapose_vis_stages` | Request MegaPose debug images and select stages. |

Combined PnP modes require `keypoints_` in the checkpoint. Missing priors raise before detection rather than silently changing the solver. Inspect `pnp_success` for each instance’s initial PnP status.

Inputs whose longest side exceeds 640 pixels are resized and then padded to the detector’s 32-pixel stride. Intrinsics and optional depth are adjusted together. Pixel-valued result fields use this processed canvas. Use the [mask restoration helper](results-api.md) to return to input resolution.

```python
# Initialize the models once; repeat prediction for images or video frames.
results = tester.predict(cam_K, image_bgr, [1])
entry = results.get(1)
if entry is not None:
    valid = entry['pnp_success']
    poses_mm = entry['Rts'][valid]
    print(poses_mm[:, :3, 3])
```

Undetected objects have no corresponding integer key. String keys contain timings and images. See the [results dictionary](results-api.md). Call WAPR through the separate [mask integration function](refinement-api.md).
