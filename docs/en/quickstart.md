# Your first prediction

Work from the project root, which contains `examples/inference/`. Complete [installation](installation.md) and [resource preparation](assets.md) first.

## RGB example

Edit the resource paths and object IDs at the end of `examples/inference/predict_image.py`, then run:

```bash
python examples/inference/predict_image.py
```

Outputs go to `outputs/rgb/`, including detection and standard pose visualizations. Applications may also call:

```python
from examples.inference.predict_image import main

# Use your own image, intrinsics and matching model resources.
results = main("data/demo-bin-picking", "data/test_imgs_RGBD/000003_rgb.png",
               "data/test_imgs_RGBD/000003_camera_matrix.json", [1], "outputs/rgb")
```

This step requires matching weights and a CUDA-capable GPU environment.
