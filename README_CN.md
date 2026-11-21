# HCCEPose

![HCCEPose](docs/assets/hccepose-readme-banner.webp)

[English](README.md) · 中文

从图像中的像素，到物体在三维空间中的位置。HccePose(BF) 联合预测已知物体的前、后表面坐标，通过超密集 2D–3D 对应求解物体相对于相机的旋转和平移。


## 从这里开始

| 你的目标 | 对应文档 |
| --- | --- |
| 跑通预训练示例 | [环境安装](docs/zh/installation.md) → [模型与输入](docs/zh/assets.md) → [第一次预测](docs/zh/quickstart.md) |
| 处理图片或视频 | [RGB 图片](docs/zh/rgb-inference.md) · [视频逐帧](docs/zh/video.md) |
| 使用深度估计或精修 | [WAPR](docs/zh/rgbd-wapr.md) · [MegaPose](docs/zh/rgbd-megapose.md) · [深度对比](docs/zh/depth-comparison.md) |
| 为新物体训练 | [网格与对称性](docs/zh/object-preparation.md) → [PBR 渲染](docs/zh/rendering.md) → [检测器](docs/zh/detector-training.md) → [表面标签](docs/zh/coordinate-labels.md) → [位姿训练](docs/zh/pose-training.md) |
| 在程序中接入 | [Tester](docs/zh/tester-api.md) · [结果与坐标](docs/zh/results-api.md) · [精修接口](docs/zh/refinement-api.md) · [数据加载](docs/zh/data-api.md) · [工具](docs/zh/utilities-api.md) |
| 加速、评测和查阅 | [ONNX/TensorRT](docs/zh/acceleration.md) · [BOP 评测](docs/zh/bop-evaluation.md) · [公开更新](docs/zh/changelog.md) · [项目结构](docs/zh/project-structure.md) |

## 开始使用

项目提供 RGB 推理、WAPR 接入，以及可选的外部 MegaPose 接口。先准备匹配的检测器与物体位姿权重、物体网格、相机内参和输入图像，再运行推理示例。具体步骤见[安装说明](docs/zh/installation.md)、[资源约定](docs/zh/assets.md)和[第一次预测](docs/zh/quickstart.md)。

```bash
python examples/inference/predict_image.py
```

修改入口末尾的资源路径与物体编号后，在项目根目录运行上述命令。输出保存于 `outputs/rgb/`。

## WAPR 接入

[WAPR](https://github.com/WangYuLin-SEU/WAPR/tree/4e591be318a828c91980c58ef8bbdc34e6599881) 需单独安装。示例将 HccePose 从当前图像预测的实例掩码映射回原图，结合配准深度、相机内参及各物体的米制网格和直径估计位姿。每个实例输出 `pose_4x4_m`（物体到相机，平移单位为米）和针对本次输入计算的 `score_6d`，保持输入实例顺序。该接口从掩码估计位姿，无需 HccePose 位姿初值。详见 [WAPR 使用说明](docs/zh/rgbd-wapr.md)。

可选的分数筛选与同类 CAD 渲染掩码 NMS 见[后处理教程](docs/zh/wapr-postprocessing.md)及 `examples/inference/postprocess_wapr.py`，原始推理 JSON 单独保留。

## 方法

![HccePose(BF) 论文方法图](show_vis/fig2.jpg)

HccePose(BF) 网络结构与位姿求解流程。参阅[方法与流程](docs/zh/method.md)和[结果与坐标](docs/zh/results-api.md)。


## 新物体与完整训练流程

公开上游提供模型居中、旋转对称标注、BOP 元数据、PBR 渲染、YOLO 训练、前后表面标签、逐物体位姿训练和 BOP 评测脚本。双语 docs 保留逐阶段操作、公开参数、输出目录和示例图；阶段脚本仍位于仓库根目录，复现原始公开版本时参阅[安装说明](docs/zh/installation.md#公开训练与渲染源码)，应用推理使用本项目 `examples/inference/`。

公开 BOP 权重、历史结果和榜单见[BOP 评测](docs/zh/bop-evaluation.md)。这些历史结果对应原公开设置，不应套用于其他数据、模型或配置。

## 文档网站

[项目网站](https://wangyulin-seu.github.io/hccepose/?lang=zh) · [使用文档](https://wangyulin-seu.github.io/hccepose/docs/index.html?lang=zh) · [全部项目](https://wangyulin-seu.github.io/?lang=zh)


```bash
python -m pip install -r docs/requirements.txt
python docs/build.py
python -m http.server 8768 --directory pages
```

首页：`http://127.0.0.1:8768/`；文档：`http://127.0.0.1:8768/docs/index.html?lang=zh`。

源码目录与脚本位置见[项目结构](docs/zh/project-structure.md)。许可见 [LICENSE](LICENSE)，公开来源见 [PUBLIC_SOURCES.md](PUBLIC_SOURCES.md)。第三方组件遵循其各自许可证。

[HccePose(BF), ICCV 2025](https://arxiv.org/abs/2510.10177)

## 引用

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
