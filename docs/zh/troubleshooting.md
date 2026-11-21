# 故障排查

如果不确定问题是否来自自己的图像或模型，先在 HccePose 主环境中运行[PyTorch RGB 示例](quickstart.md)，使用配套权重、图像和相机内参。

## 安装与导入

| 现象 | 检查与处理 |
| --- | --- |
| `No module named bop_toolkit` | 在仓库根目录解压 `bop_toolkit.zip`，确认能看到 `bop_toolkit/bop_toolkit_lib/`，再从根目录运行示例。见[环境安装](installation.md)。 |
| `All ufuncs must have type numpy.ufunc` | 确认运行脚本的 Python 解释器，再在对应环境里安装兼容的 NumPy 和 SciPy。Python 3.10 参考环境使用 NumPy `1.26.4` 和 SciPy `1.15.3`。 |
| `import cv2` 时出现 `numpy.core.multiarray` 或 `_ARRAY_API` | 按[安装文档](installation.md)统一 OpenCV 和 NumPy 版本；混用不同 NumPy 主版本与二进制 wheel 可能导致导入失败。 |
| 缺少 `torch.amp` | 使用 HccePose 主环境中的 PyTorch 2.x 运行根目录示例。`.envs/megapose/bin/python` 属于独立的 MegaPose 子进程运行环境。 |
| CUDA 不可用，或显示“Using CPU”后报错 | 用下面的命令检查 PyTorch 能否识别显卡。即使打印了这条提示，`Tester` 仍会将模型几何数据送入 CUDA，推理需要可用的 GPU 环境。 |
| EGL/OpenGL 初始化失败 | 按[安装文档](installation.md)检查显卡驱动是否支持 EGL，以及系统图形库是否齐全。渲染、部分可视化和位姿精修都依赖这些库。 |

可以先查看解释器和 GPU 状态，不启动模型推理：

```bash
python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available())"
```

## 资源与首次运行

| 现象 | 检查与处理 |
| --- | --- |
| 已有 HccePose 权重仍触发 torchvision/ResNet 下载 | 默认网络构造函数在恢复任务 checkpoint 前调用 `models.resnet34(pretrained=True)`，首次初始化仍可能向 PyTorch 缓存下载主干权重。 |
| `best_score` 中的 checkpoint 显示迭代次数为 `0` | 从正表面模型初始化或继续微调所得的权重可能保留这个计数。可用配套示例图验证权重是否能加载。 |

## 结果异常或没有物体

调整模型参数前，依次检查以下内容：

1. **物体与权重。** 核对物体 ID、网格和模型元数据，并确认检测器和 HccePose 权重均适用于这个物体。
2. **图像通道。** `cv2.imread` 得到的图像已经是 `Tester.predict` 需要的 BGR 格式，可直接传入。
3. **相机内参。** 内参应对应输入分辨率。在 `Tester` 外部裁剪或缩放图像时，记得调整内参。`Tester` 会自行处理内部的缩放和补边。
4. **单位。** 示例中的 BOP 模型坐标和输出平移用毫米，位姿精修所需的深度图用米。转换时使用相机元数据或传感器实际的深度比例。
5. **检测结果。** 查看 2D 输出和所选物体 ID。如果该 ID 下没有实例通过筛选，字典中可能没有对应条目，读取位姿前需检查。

[结果字段](results-api.md)说明处理后图像坐标、可选字段和位姿输出；[RGB 推理](rgb-inference.md)介绍示例阈值。

## 可选后端

| 现象 | 检查与处理 |
| --- | --- |
| ONNX 初始化时运行 pip 或改变包版本 | 设置程序可能在首次运行时安装依赖。建议使用独立环境，涉及的包见[推理加速](acceleration.md)。 |
| ONNX Runtime 能导入但没有 CUDA provider | ONNX 和 ONNX Runtime 能成功导入时，辅助函数就会返回，即使安装的是 CPU runtime。需要在当前环境中安装适合的 GPU runtime，并查看 session/provider 日志。 |
| 缺少 TensorRT execution provider | 选用的 ONNX Runtime 构建必须包含 `TensorrtExecutionProvider`；仅安装 TensorRT Python 包不能给 ORT 补上缺失的 provider。 |
| 列出了 TensorRT provider，但无法加载 `libnvinfer` | 安装匹配的 TensorRT 动态库，在启动 Python 前使库目录可见，并核对 CUDA／驱动／动态库组合。排查可选后端时可继续使用已经跑通的 PyTorch 路径。 |
| MegaPose 首次运行长时间准备文件 | 准备流程可能克隆 `third_party_megapose6d/`，在 `.envs/megapose/` 创建 Python 3.9 环境，安装独立依赖并下载模型。需要 conda、网络、可写目录和足够磁盘空间。见 [MegaPose](rgbd-megapose.md)。 |
| MegaPose 缺少配置文件或 checkpoint | 检查目录是否包含上游配置文件和权重。有些镜像只提供 checkpoint。根据下载器或 rclone 的报错补齐缺失文件。 |

反馈问题时，请附上脚本名、运行命令、Python 解释器路径、所选后端、相关文件路径，以及日志中的第一处异常。如果报错来自 MegaPose 子进程，请保留完整日志，方便查看子进程的调用栈。

## WAPR 输入与下载缓存

WAPR 的包、编译环境和资源准备见[安装说明](installation.md)。本项目使用独立的 `estimate_from_masks` 接口。遇到错误先检查原图掩码、配准米制深度、相机内参和米制网格；空掩码、无正深度和输出数量错误会报错。

Hugging Face 离线回退可能返回已有目录，因此资源页仍逐项检查七文件大小与两个权重摘要。发现损坏时按[资源页](assets.md)重新下载；额外旧权重或说明文件需检查后移出 `best_score/`，不会自动删除。

`pnp_success=False` 表示初始 PnP 未成功，不能把回退单位矩阵当成可靠位姿。默认组合 PnP 缺少 `keypoints_` 先验时，会报告 checkpoint 路径；先核对权重与物体、网络架构是否匹配。
