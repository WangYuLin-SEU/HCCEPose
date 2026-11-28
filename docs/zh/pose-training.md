# 为每个物体训练 HccePose(BF)

本页使用本仓库根目录中的准备、训练或评测脚本。请按[安装说明](installation.md#公开训练与渲染源码)准备对应依赖，再从仓库根目录执行命令。应用推理使用根目录的 `s4_p3_*.py` 示例。

HccePose(BF) 为每个物体单独训练一套权重。`s4_p2_train_bf_pbr.py` 可以依次训练多个物体，每个物体的训练可以使用一张或多张 GPU。

## 配置训练脚本

先准备[物体模型与元数据](object-preparation.md)、[PBR 场景](rendering.md)和[正背面坐标标签](coordinate-labels.md)。打开[训练脚本](https://github.com/WangYuLin-SEU/HCCEPose/blob/main/s4_p2_train_bf_pbr.py)，在 `__main__` 中将 `dataset_path` 换成自己的数据集路径，并设置物体 ID 范围。

| 参数 | 示例设置 | 用途 |
| --- | --- | --- |
| `start_obj_id`、`end_obj_id` | `1`、`5` | 依次训练物体 1 至 5。只训练一个物体时，两者填同一 ID。 |
| `total_iteration` | `50001` | 每个物体的训练迭代设置。 |
| `batch_size`、`lr` | `24`、`0.0002` | 每张 GPU 的批量大小及学习率。 |
| `efficientnet_key` | `None` | 训练脚本的骨干网络设置。 |

DDP 训练处理的样本总数按下式计算：

```
total samples = total iteration × batch size × GPU number
```

## 单卡调试或多卡训练

默认 `ide_debug=True`，适合在一张 GPU 上运行或调试：

```
nohup python -u s4_p2_train_bf_pbr.py > log4.file 2>&1 &
```

多卡训练时，将 `ide_debug` 设为 `False`，并根据实际 GPU 数量设置 `--nproc_per_node`。下面的命令使用六张卡。在独立终端会话中启动，有助于避免暂停 IDE 导致各进程的通信中断。

```
screen -S train_ddp
nohup python -u -m torch.distributed.launch --nproc_per_node 6 s4_p2_train_bf_pbr.py > log4.file 2>&1 &
```

## 检查权重和训练进度

权重按物体 ID 保存在数据集的 `HccePose/` 目录下。以 `demo-tex-objs` 为例：

```
demo-tex-objs
|--- HccePose
    |--- obj_01
    ...
    |--- obj_10
|--- models
|--- train_pbr
|--- train_pbr_xyz_GT_back
|--- train_pbr_xyz_GT_front
```

训练过程中的 `test()` 从同一个 `train_pbr` 划分中抽取 1% 的样本，可以用来观察训练进度。要报告基准成绩，请在对应的测试集上进行 [BOP 评测](bop-evaluation.md)。

本地监控指标对存在非平凡对称变换的物体使用 ADD-S，其余使用 ADD；空对称列表或仅单位变换仍使用 ADD。通过阈值保持物体直径的 10%。该指标不代替官方 BOP 评分。

续训时，已有 checkpoint 无法恢复会停止并报告文件和原因，不会静默重置训练步数。只有 checkpoint 目录不存在或为空才开始新训练。
