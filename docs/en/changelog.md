# Changelog

### June 2, 2026

The BOP Writer now uses pyrender with multiple processes to generate ground-truth masks and scene metadata. [Source commit](https://github.com/WangYuLin-SEU/HCCEPose/commit/f75cc4d5db2fc298fc7b853f8c75273757872382)

### April 6, 2026

The installation guide added a minimal inference setup, first-run timing notes, troubleshooting and an optional [`requirements-inference.txt`](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/requirements-inference.txt). The ONNX scripts may install dependencies automatically. Run them in a dedicated venv or conda environment.

### April 4, 2026

- Added the MegaPose refinement interface and RGB-D examples.
- HccePose supports ONNX or TensorRT through `hccepose_acceleration`; `time_dict` provides stage timings.
- Released RGB-D frames `000000` through `000003`. Each frame includes `{stem}_rgb.png`, `{stem}_depth.png` and `{stem}_camK.json`. See the [sample files](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/4e482e540314622981ccd4fce5beddac7f9d3e1d/test_imgs_RGBD) and [download instructions](assets.md).
- Pass OpenCV BGR images directly to `Tester.predict`; MegaPose debug images are saved in BGR.

### October 28, 2025

Updated label generation. The PBR rendering script can create a default `camera.json` automatically. [Rendering script](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/s2_p1_gen_pbr_data.py)

### October 27, 2025

Released [cc0textures-512](https://huggingface.co/datasets/SEU-WYL/HccePose/blob/4e482e540314622981ccd4fce5beddac7f9d3e1d/cc0textures-512.zip), a texture library of about 600 MB that can replace the original 44 GB CC0Textures library in the rendering example.

## Historical roadmap

The 2025 roadmap targeted the end of that year. It marked three items complete: weights for the seven core BOP datasets, the BOP Challenge evaluation workflow, and tutorials covering object preparation, rendering, YOLOv11 labels and training, and HccePose(BF) labels and training.

Three items were still planned at the time: inter-frame pose tracking, real-scene pose dataset preparation with HccePose(BF), and training on a mix of PBR and real data. The [video example](video.md) in this repository runs inference independently on each frame.
