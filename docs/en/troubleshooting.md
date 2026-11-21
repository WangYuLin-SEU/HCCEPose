# Troubleshooting

To check whether a problem is specific to your inputs, run the [PyTorch RGB demo](quickstart.md) in the main HccePose environment with the original weights, images and camera matrix.

## Installation and imports

| Symptom | Check and next action |
| --- | --- |
| `No module named bop_toolkit` | Extract `bop_toolkit.zip` at the repository root. Check for `bop_toolkit/bop_toolkit_lib/` and run the example from the root. See [installation](installation.md). |
| `All ufuncs must have type numpy.ufunc` | Check which Python executable runs the script. In its environment, install compatible versions of NumPy and SciPy. The Python 3.10 reference uses NumPy `1.26.4` and SciPy `1.15.3`. |
| `numpy.core.multiarray` or `_ARRAY_API` error on `import cv2` | Align OpenCV and NumPy with the reference pins in [installation](installation.md); mixed NumPy major versions and binary wheels can cause import failures. |
| Missing `torch.amp` | Run the root demo in the main HccePose PyTorch 2.x environment. `.envs/megapose/bin/python` belongs to the separate MegaPose subprocess runtime. |
| CUDA unavailable, or an error after “Using CPU” | Use the command below to check whether PyTorch can see the GPU. `Tester` moves model geometry to CUDA even after printing this message. Inference requires a working GPU installation. |
| EGL/OpenGL initialization failure | Check the EGL-capable GPU driver and system graphics dependencies from [installation](installation.md). Rendering and some visualization/refinement workflows need these beyond Python wheels. |

To identify the active interpreter and GPU availability without starting inference:

```bash
python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available())"
```

## Assets and first execution

| Symptom | Check and next action |
| --- | --- |
| A torchvision/ResNet download starts despite existing HccePose weights | The default network constructor calls `models.resnet34(pretrained=True)` before restoring the task checkpoint. First initialization may therefore download backbone weights into the PyTorch cache. |
| A `best_score` checkpoint reports iteration `0` | A checkpoint initialized or fine-tuned from a front-surface model can retain this iteration count. Try it on the supplied sample images to check that it loads. |

## Outputs look wrong or no object appears

Check these before altering model parameters:

1. **Object identity and weights:** the object ID, mesh/model information, detector, and per-object HccePose checkpoint must describe the same trained object.
2. **Image colors:** `Tester.predict` expects OpenCV BGR. Do not apply `COLOR_RGB2BGR` to an image already loaded by `cv2.imread`.
3. **Camera geometry:** use intrinsics for the supplied input resolution. If you resize or crop outside `Tester`, transform the intrinsics accordingly; internal resizing and padding are handled by `Tester`.
4. **Units:** supplied BOP-style model coordinates and pose translations are in millimeters. A refinement depth map must be in meters; use the sample camera metadata or the sensor's actual depth scale.
5. **Detection:** inspect the 2D output and the selected object ID. Per-object result entries may be absent when no instance survives filtering; callers should check that the entry exists before reading poses.

The [results reference](results-api.md) explains processed-image coordinates, optional fields, and pose outputs. [RGB inference](rgb-inference.md) documents the example thresholds.

## Optional backends

| Symptom | Check and next action |
| --- | --- |
| ONNX setup runs pip or changes package versions | On first use, the setup helper can install dependencies. Use a dedicated environment. See [acceleration](acceleration.md) for the packages it may change. |
| ONNX Runtime imports but has no CUDA provider | The setup helper returns when ONNX and ONNX Runtime import successfully, even for a CPU-only runtime. Install an appropriate GPU runtime in the active environment and inspect the session/provider diagnostics. |
| TensorRT execution provider missing | The selected ONNX Runtime build must include `TensorrtExecutionProvider`; installing the TensorRT Python package alone does not add a missing provider to ORT. |
| TensorRT provider is listed but `libnvinfer` cannot load | Install matching TensorRT libraries and make their directory visible before starting Python. Check the CUDA/driver/library combination. Use the working PyTorch route while resolving the optional backend. |
| MegaPose first use spends a long time preparing files | Setup can clone `third_party_megapose6d/`, create a Python 3.9 environment under `.envs/megapose/`, install its own stack, and fetch models. It requires conda, network access, writable storage, and available disk space. See [MegaPose](rgbd-megapose.md). |
| MegaPose model config/checkpoint missing | Keep the full upstream model layout, including config files; a checkpoint-only mirror may be incomplete. Review the downloader/rclone error emitted by the setup rather than copying an unrelated checkpoint. |

When reporting an error, include the command, script name, Python executable, backend, relevant file paths and the first exception in the log. For a MegaPose subprocess error, keep the full log so the traceback from that environment is available.

## WAPR inputs and download caches

See [Installation](installation.md) for WAPR packages, compilation and resources. This project uses separate `estimate_from_masks` calls. Check original-frame masks, registered meter depth, intrinsics and meter meshes first. Empty masks, missing positive depth and output count errors raise explicitly.

Hugging Face offline fallback may return an existing directory. The assets recipe therefore checks seven exact file sizes and both weight digests. Follow [Assets](assets.md) to redownload corrupt files. Inspect and move extra old checkpoints or notes out of `best_score/`; they are never automatically deleted.

`pnp_success=False` means initial PnP failed. Do not treat a fallback identity transform as a reliable pose. Missing `keypoints_` for the default combined PnP raises with the checkpoint path; first verify object and network compatibility.
