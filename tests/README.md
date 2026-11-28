# Running the tests / 运行测试

## English

Use the Python environment prepared in [installation](../docs/en/installation.md), with the inference dependencies and PyTorch installed. From the project root, install the test and documentation dependencies, then run:

```bash
python -m pip install -r tests/requirements.txt -r docs/requirements.txt
python -m zipfile -e bop_toolkit.zip .
python -m pytest -q tests
python docs/build.py
```

`test_rgb_runtime.py` uses synthetic detections and coordinates to exercise image padding, OpenCV PnP, per-instance success reporting and checkpoint errors. `test_wapr_public.py` checks image channels, metric units, restored masks, instance identity, empty detections and invalid inputs with a recording estimator. Neither suite needs object weights or an installed WAPR backend. `test_wapr_postprocess.py` checks score boundaries, per-object NMS forwarding, units, instance order, errors and raw-file preservation using a recording backend. `test_docs_build.py` checks search text and document resource links; the website build also validates local links, anchors and resources.

These checks cover interface and geometry behavior. To evaluate model accuracy, throughput or a new CUDA installation, run inference with matching weights and data on the intended GPU.

## 中文

使用[安装说明](../docs/zh/installation.md)中准备的 Python 环境，先安装推理依赖和 PyTorch。在项目根目录安装测试与文档依赖，再运行：

```bash
python -m pip install -r tests/requirements.txt -r docs/requirements.txt
python -m zipfile -e bop_toolkit.zip .
python -m pytest -q tests
python docs/build.py
```

`test_rgb_runtime.py` 使用合成检测与坐标检查图像补边、OpenCV PnP、逐实例求解状态和权重文件异常。`test_wapr_public.py` 使用记录输入的模拟估计器，检查图像通道、米制单位、掩码还原、实例标识、空检测及无效输入；两组测试均无需物体权重或 WAPR 后端。`test_wapr_postprocess.py` 使用记录调用的模拟后端检查分数边界、按物体分组的 NMS 调用、单位、实例顺序、异常及原始文件保护。`test_docs_build.py` 检查搜索正文和文档资源链接，网站构建还会检查本地链接、锚点与资源。

这些检查覆盖接口与几何行为。评估模型精度、吞吐量或新的 CUDA 环境时，需在目标 GPU 上使用匹配权重和数据运行推理。
