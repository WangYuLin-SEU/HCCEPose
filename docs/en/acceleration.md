# ONNX and TensorRT acceleration

First run the [PyTorch RGB example](quickstart.md). HccePose can execute its network forward pass with ONNX Runtime; TensorRT uses ONNX Runtime’s execution provider. Detection, coordinate decoding, PnP and result assembly remain in the existing pipeline.

## Select a backend and cache

```python
from HccePose.tester import Tester

tester = Tester(
    dataset,
    CUDA_DEVICE='0',
    hccepose_vis=False,
    hccepose_acceleration='onnx',
    hccepose_acceleration_cache_dir=None,
)
results = tester.predict(cam_K, image_bgr, [1])
```

Change `'onnx'` to `'tensorrt'` for TensorRT; `'pytorch'` is the default. `acceleration` and `acceleration_cache_dir` are compatibility aliases. These arguments do not select YOLO or MegaPose backends.

Each object defaults to `<dataset>/HccePose/obj_XX/onnx_cache/`, including exports and TensorRT engine caches. Filenames identify the checkpoint, input size and related details. Do not mix caches from incompatible networks or devices. Initial export and engine building can substantially increase startup time.

## Initialization and environment changes

Selecting ONNX/TensorRT invokes `ensure_acceleration_backend_environment`, which checks the toolkit, PyTorch and common dependencies and may install missing packages using the current interpreter. Use a dedicated environment and install CUDA-enabled PyTorch beforehand.

- Preparation may change NumPy to `1.26.4` and install pinned OpenCV packages.
- An importable ONNX Runtime is not necessarily GPU-enabled; inspect its actual providers.
- TensorRT requires compatible ONNX Runtime, CUDA, drivers and `libnvinfer`. Installing the `tensorrt` Python package alone does not ensure this ONNX Runtime supports TensorRT.
- `TMPDIR` and `PIP_CACHE_DIR` can select temporary storage and pip caches; distinguish these from model resources.

See the public [HccePose runner](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/HccePose/hccepose_acceleration.py). The current project retains the HccePose network acceleration entry point.

## Inspect providers and outputs

```python
runner = tester.HccePose_Runtime_Item[1]
print(runner.session.get_providers())
```

The ONNX path prefers CUDA, then CPU; TensorRT prefers TensorRT, CUDA, then CPU, subject to availability. Registered providers do not fully describe actual operator placement; consult ONNX Runtime logs. The complete HccePose pipeline still requires CUDA.

For comparisons, fix images, weights, PnP options and visualization settings, and preserve separate results. Mask and coordinate differences measure numerical agreement. Accuracy needs ground-truth [BOP evaluation](bop-evaluation.md). Do not match independent multi-instance detections merely by list position.

## Comparison recipes and timing

Public upstream includes [ONNX](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/s4_p3_test_mi10_bin_picking_onnx.py) and [TensorRT comparison scripts](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/s4_p3_test_mi10_bin_picking_tensorrt.py). They live in the upstream root; adjust input paths for your assets.

First use may include downloads, installation, export or engine building. Separate those from repeated inference. `time` and `time_dict` are wall-clock measurements defined in [results](results-api.md). Host/GPU transfers, detection and PnP can remain bottlenecks; faster network execution does not imply proportional end-to-end acceleration.
