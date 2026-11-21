# 文档导航

HccePose(BF) 从 RGB 图像估计已知物体的 6D 位姿。先跑通预训练示例，再按需要接入自己的图像、深度数据或新物体。

## 概览

- [方法与流程](method.md)
- [项目结构](project-structure.md)

## 快速入门

- [环境安装](installation.md)
- [模型与输入](assets.md)
- [第一次预测](quickstart.md)
- [常见问题](troubleshooting.md)

## 使用案例

- [RGB 图片推理](rgb-inference.md)
- [视频推理](video.md)
- [WAPR 接入](rgbd-wapr.md)
- [WAPR 评分筛选与 NMS](wapr-postprocessing.md)
- [MegaPose 位姿精修](rgbd-megapose.md)
- [位姿精修与深度对比](depth-comparison.md)
- [ONNX 与 TensorRT](acceleration.md)

## 为新物体训练

- [网格与对称信息](object-preparation.md)
- [PBR 场景渲染](rendering.md)
- [检测器训练](detector-training.md)
- [前后表面坐标标签](coordinate-labels.md)
- [位姿网络训练](pose-training.md)

## 接口文档

- [Tester 接口](tester-api.md)
- [结果与坐标](results-api.md)
- [位姿精修与 RGB-D 接口](refinement-api.md)
- [数据加载接口](data-api.md)
- [工具与进阶接口](utilities-api.md)

## 项目

- [BOP 评测](bop-evaluation.md)
- [公开更新记录](changelog.md)
- [引用与许可](citation.md)
