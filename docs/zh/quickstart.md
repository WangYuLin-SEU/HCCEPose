# 第一次预测

在包含 `examples/inference/` 的项目根目录操作。先完成[环境安装](installation.md)和[资源准备](assets.md)。

## RGB 示例

编辑 `examples/inference/predict_image.py` 末尾的资源路径和物体编号，再执行：

```bash
python examples/inference/predict_image.py
```

结果保存于 `outputs/rgb/`，包含检测框和标准位姿可视化。也可在 Python 应用中调用：

```python
from examples.inference.predict_image import main

# 路径指向自己准备的图像、内参及配套模型。
results = main("data/demo-bin-picking", "data/test_imgs_RGBD/000003_rgb.png",
               "data/test_imgs_RGBD/000003_camera_matrix.json", [1], "outputs/rgb")
```

这一步需要匹配的权重与支持 CUDA 的 GPU 环境。
