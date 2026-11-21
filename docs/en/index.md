# Documentation guide

HccePose(BF) estimates known-object 6D poses from RGB images. Start with pretrained inference, then integrate your own images, depth or new objects.

## Overview

- [Method and pipeline](method.md)
- [Project structure](project-structure.md)

## Start here

- [Installation](installation.md)
- [Models and inputs](assets.md)
- [Your first prediction](quickstart.md)
- [Troubleshooting](troubleshooting.md)

## Examples

- [RGB image inference](rgb-inference.md)
- [Video inference](video.md)
- [WAPR integration](rgbd-wapr.md)
- [WAPR score filtering and NMS](wapr-postprocessing.md)
- [MegaPose refinement](rgbd-megapose.md)
- [Refinement and depth comparison](depth-comparison.md)
- [ONNX and TensorRT](acceleration.md)

## Train new objects

- [Meshes and symmetry](object-preparation.md)
- [PBR rendering](rendering.md)
- [Detector training](detector-training.md)
- [Front/back coordinate labels](coordinate-labels.md)
- [Pose network training](pose-training.md)

## API reference

- [Tester API](tester-api.md)
- [Results and coordinates](results-api.md)
- [Refinement and RGB-D API](refinement-api.md)
- [Data loading API](data-api.md)
- [Utilities and advanced API](utilities-api.md)

## Project

- [BOP evaluation](bop-evaluation.md)
- [Public changelog](changelog.md)
- [Citation and licenses](citation.md)
