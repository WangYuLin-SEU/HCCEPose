# ONNX 与 TensorRT 加速

先跑通 [PyTorch RGB 示例](quickstart.md)。HccePose 可用 ONNX Runtime 执行网络前向计算，TensorRT 则通过 ONNX Runtime 的执行提供程序运行。检测、坐标解码、PnP 和结果整理仍在原流程中。

## 后端与缓存选择

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

将 `'onnx'` 改为 `'tensorrt'` 选择 TensorRT，`'pytorch'` 为默认路径。`acceleration` 与 `acceleration_cache_dir` 是兼容别名。上述参数不切换 YOLO 或 MegaPose 的执行后端。

默认每个物体使用 `<dataset>/HccePose/obj_XX/onnx_cache/`，其中包含导出模型及 TensorRT engine 缓存。文件名记录 checkpoint、输入尺寸等信息，不要把不同网络或设备的缓存混用。首次导出与 engine 构建可能显著增加启动时间。

## 初始化与环境变化

选择 ONNX/TensorRT 时，`ensure_acceleration_backend_environment` 会检查工具包、PyTorch 和常用依赖，并可能通过当前解释器安装缺失包。使用独立环境；主环境中的 CUDA PyTorch 需先安装好。

- 环境准备可能调整 NumPy 到 `1.26.4`，安装指定 OpenCV 包。
- ONNX Runtime 已能导入时，不保证它就是 GPU 版本；检查实际 provider。
- TensorRT 需要兼容的 ONNX Runtime、CUDA、驱动与 `libnvinfer`。仅安装 `tensorrt` Python 包不代表当前 ONNX Runtime 支持 TensorRT。
- `TMPDIR` 与 `PIP_CACHE_DIR` 可用于选择临时目录及 pip 缓存；不要把模型资源目录与临时目录混淆。

源码与上游接口见 [HccePose runner](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/HccePose/hccepose_acceleration.py)。当前项目保留 HccePose 网络加速入口。

## 检查 provider 与输出

```python
runner = tester.HccePose_Runtime_Item[1]
print(runner.session.get_providers())
```

ONNX 路径优先 CUDA，再到 CPU；TensorRT 路径优先 TensorRT，再到 CUDA、CPU，实际可用性取决于安装。注册 provider 与每个算子的实际执行并非完全等价，需要结合 ONNX Runtime 日志。完整 HccePose 流程仍需要 CUDA。

对比后端时，固定图像、权重、PnP 设置及可视化开关，并分别保存结果。掩码和坐标差异反映数值一致性，最终精度需按 [BOP](bop-evaluation.md) 对照真值评估。多实例场景不能仅按列表位置匹配两个独立检测结果。

## 对比示例与计时

公开上游包含 [ONNX 对比脚本](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/s4_p3_test_mi10_bin_picking_onnx.py) 与 [TensorRT 对比脚本](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/s4_p3_test_mi10_bin_picking_tensorrt.py)。它们位于上游根目录，需按实际资源调整输入路径。

首次运行可能包含下载、依赖安装、导出或构建，不应计入稳定重复推理的速度。`time` 和 `time_dict` 为墙钟计时，定义见[结果字段](results-api.md)。主机与 GPU 间的数据搬运、检测和 PnP 仍可能成为瓶颈，因此网络前向加速不等于整条流程同比加速。
