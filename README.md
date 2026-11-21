# HCCEPose

![HCCEPose](docs/assets/hccepose-readme-banner.webp)

English · [中文](README_CN.md)

From image pixels to an object’s pose in 3D. HccePose(BF) predicts coordinates on both front and back surfaces of known objects and estimates object-to-camera rotation and translation from ultra-dense 2D–3D correspondences.


## Find your starting point

| Goal | Documentation |
| --- | --- |
| Run pretrained inference | [Installation](docs/en/installation.md) → [Models and inputs](docs/en/assets.md) → [First prediction](docs/en/quickstart.md) |
| Process images or video | [RGB images](docs/en/rgb-inference.md) · [Video frames](docs/en/video.md) |
| Estimate or refine with depth | [WAPR](docs/en/rgbd-wapr.md) · [MegaPose](docs/en/rgbd-megapose.md) · [Depth comparison](docs/en/depth-comparison.md) |
| Train new objects | [Meshes and symmetry](docs/en/object-preparation.md) → [PBR rendering](docs/en/rendering.md) → [Detector](docs/en/detector-training.md) → [Surface labels](docs/en/coordinate-labels.md) → [Pose training](docs/en/pose-training.md) |
| Integrate into an application | [Tester](docs/en/tester-api.md) · [Results and coordinates](docs/en/results-api.md) · [Refinement](docs/en/refinement-api.md) · [Data loading](docs/en/data-api.md) · [Utilities](docs/en/utilities-api.md) |
| Accelerate, evaluate or explore | [ONNX/TensorRT](docs/en/acceleration.md) · [BOP evaluation](docs/en/bop-evaluation.md) · [Public changelog](docs/en/changelog.md) · [Project structure](docs/en/project-structure.md) |

## Get started

The project provides RGB inference, WAPR integration, and an optional interface to an external MegaPose installation. Prepare matching detector and object pose weights, object meshes, camera intrinsics, and input images, then run an inference example. See [installation](docs/en/installation.md), [resource conventions](docs/en/assets.md) and [your first prediction](docs/en/quickstart.md).

```bash
python examples/inference/predict_image.py
```

Run this command from the project root after editing the resource paths and object IDs at the end of the entry script. Outputs are saved to `outputs/rgb/`.

## WAPR integration

Install [WAPR](https://github.com/WangYuLin-SEU/WAPR/tree/4e591be318a828c91980c58ef8bbdc34e6599881) separately. The example restores instance masks predicted by HccePose for the current image to the full frame, then estimates poses using registered depth, camera intrinsics, and each object’s metric mesh and diameter. Each instance produces `pose_4x4_m` (object to camera, translation in meters) and a `score_6d` computed for that input. Input instance order is preserved. This API estimates poses from masks without HccePose pose seeds. See [WAPR usage](docs/en/rgbd-wapr.md).

Optional score filtering and per-object CAD rendered-mask NMS are provided in [the postprocessing tutorial](docs/en/wapr-postprocessing.md) and `examples/inference/postprocess_wapr.py`. The raw inference JSON is kept separately.

## Method

![HccePose(BF) paper figure](show_vis/fig2.jpg)

HccePose(BF) network architecture and pose estimation pipeline. See the [method](docs/en/method.md) and [results contract](docs/en/results-api.md).


## New objects and the complete training workflow

Public upstream includes mesh centering, rotational symmetry, BOP metadata, PBR rendering, YOLO training, front/back labels, per-object pose training and BOP evaluation scripts. Bilingual docs retain stage-by-stage procedures, public parameters, output layouts and example figures. Stage scripts remain in the repository root; for the original published revision, follow [Installation](docs/en/installation.md#public-training-and-rendering-source); use this project’s `examples/inference/` for application inference.

See [BOP evaluation](docs/en/bop-evaluation.md) for public weights, historical results and leaderboard images. Those historical results describe the original published settings, rather than other data, models or configurations.

## Documentation website

[Project website](https://wangyulin-seu.github.io/hccepose/?lang=en) · [Documentation](https://wangyulin-seu.github.io/hccepose/docs/index.html?lang=en) · [All projects](https://wangyulin-seu.github.io/?lang=en)


```bash
python -m pip install -r docs/requirements.txt
python docs/build.py
python -m http.server 8768 --directory pages
```

Homepage: `http://127.0.0.1:8768/`; docs: `http://127.0.0.1:8768/docs/index.html?lang=en`.

See [project structure](docs/en/project-structure.md), [LICENSE](LICENSE) and [public sources](PUBLIC_SOURCES.md). Third-party components retain their own licenses.

[HccePose(BF), ICCV 2025](https://arxiv.org/abs/2510.10177)

## Citation

```bibtex
@InProceedings{HccePose_BF,
    author    = {Wang, Yulin and Hu, Mengting and Li, Hongli and Luo, Chen},
    title     = {HccePose(BF): Predicting Front \& Back Surfaces to Construct Ultra-Dense 2D-3D Correspondences for Pose Estimation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {7166-7175}
}
```
