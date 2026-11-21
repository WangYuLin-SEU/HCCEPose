# 更新记录

### 2026 年 6 月 2 日

BOP Writer 改用 pyrender 多进程生成真值掩码和场景元数据。[源码提交](https://github.com/WangYuLin-SEU/HCCEPose/commit/f75cc4d5db2fc298fc7b853f8c75273757872382)

### 2026 年 4 月 6 日

安装文档新增了最小推理环境、首次运行耗时说明、故障排查和可选的 [`requirements-inference.txt`](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/requirements-inference.txt)。ONNX 脚本可能自动安装依赖，建议在独立的 venv 或 conda 环境中运行。

### 2026 年 4 月 4 日

- 新增 MegaPose 位姿精修接口及 RGB-D 示例。
- HccePose 支持通过 `hccepose_acceleration` 选择 ONNX 或 TensorRT；`time_dict` 提供分阶段计时。
- 发布 RGB-D 示例帧 `000000` 至 `000003`。每帧含 `{stem}_rgb.png`、`{stem}_depth.png` 和 `{stem}_camK.json`。参见[示例文件](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/4e482e540314622981ccd4fce5beddac7f9d3e1d/test_imgs_RGBD)和[下载说明](assets.md)。
- OpenCV 返回的 BGR 图像直接传给 `Tester.predict`；MegaPose 调试图使用 BGR 保存。

### 2025 年 10 月 28 日

更新了标签制备流程。PBR 渲染入口可以自动生成默认的 `camera.json`。[渲染脚本](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/s2_p1_gen_pbr_data.py)

### 2025 年 10 月 27 日

发布 [cc0textures-512](https://huggingface.co/datasets/SEU-WYL/HccePose/blob/4e482e540314622981ccd4fce5beddac7f9d3e1d/cc0textures-512.zip)。这套材质库约 600 MB，可在渲染示例中替代约 44 GB 的原版 CC0Textures。

## 历史路线图

2025 年的路线图以当年年底为目标。当时列为已完成的有七个 BOP 核心数据集的权重、BOP 挑战评测流程，以及物体预处理、数据渲染、YOLOv11 标签与训练、HccePose(BF) 标签与训练的教程。

当时仍在计划中的三项工作是帧间位姿跟踪、用 HccePose(BF) 制备真实场景位姿数据集，以及联合使用 PBR 和真实数据训练。仓库里的[视频示例](video.md)展示的是逐帧独立推理。
