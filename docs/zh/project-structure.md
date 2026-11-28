# 项目结构

本项目包含应用推理、公开 WAPR 接入和双语网站源码。训练与渲染的阶段脚本见下方公开上游索引。

```text
HccePose/                 RGB 推理、网络与 BOP 加载器
s1_*.py ... s4_*.py       公开准备、训练与评测阶段
Refinement/              共用 WAPR/MegaPose 接口与 RGB-D 读取
s4_p3_*RGBD_wapr.py      WAPR 预测及自动评分筛选、NMS
scripts/                 HF 下载与资源校验
docs/zh/                 中文文档
docs/en/                 英文文档
docs/theme/              共享视觉主题
docs/assets/             图片与展示素材
docs/assets/figures/     论文、教程与效果图片
pages/                   构建后的静态网站
media/                   公开视频预览
sample_data/images/      自带 RGB 输入图片
sample_data/rgbd/        自带 RGB-D 帧与相机标定
sample_data/videos/      自带输入视频
tests/                   接口与文档检查
```

## 公开阶段脚本

以下文件位于[公开上游固定版本](installation.md#公开训练与渲染源码)根目录；点击可查看源码，参数在脚本中显式设置。

| 脚本 | 用途 |
| --- | --- |
| [s1_p1_obj_rename_center.py](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/s1_p1_obj_rename_center.py) | 模型居中与编号 |
| [s1_p2_obj_symmetry.py](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/s1_p2_obj_symmetry.py) | 旋转对称信息 |
| [s1_p3_obj_infos.py](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/s1_p3_obj_infos.py) | BOP 模型元数据 |
| [s2_p1_gen_pbr_data.py](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/s2_p1_gen_pbr_data.py) | PBR 场景渲染 |
| [s3_p1_prepare_yolo_label.py](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/s3_p1_prepare_yolo_label.py) | 检测标签转换 |
| [s3_p2_train_yolo.py](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/s3_p2_train_yolo.py) | 检测器训练 |
| [s4_p1_gen_bf_labels.py](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/s4_p1_gen_bf_labels.py) | 前后表面坐标标签 |
| [s4_p2_train_bf_pbr.py](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/s4_p2_train_bf_pbr.py) | 逐物体位姿训练 |
| [s4_p2_test_bf_pbr.py](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/s4_p2_test_bf_pbr.py) | 裁剪图验证 |
| [s4_p2_test_bf_pbr_bop_challenge.py](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/s4_p2_test_bf_pbr_bop_challenge.py) | BOP 预测导出 |

## 数据与权重目录

网格和元数据位于 `<dataset>/models/`，检测器位于 `<dataset>/yolo11/`，位姿权重位于 `<dataset>/HccePose/obj_XX/best_score/`。训练场景和两套表面标签分别位于 `train_pbr/`、`train_pbr_xyz_GT_front/`、`train_pbr_xyz_GT_back/`。

推理下载布局与相机文件见[资源说明](assets.md)。脚本组织不改变 BOP 物体编号、像素坐标和毫米模型单位。

## 构建网站

```bash
python -m pip install -r docs/requirements.txt
python docs/build.py
python -m http.server 8768 --directory pages
```

网站构建无需 GPU，也不运行推理。其他使用说明见[文档导航](index.md)。
