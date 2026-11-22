# 工具与进阶接口

用 [`Tester`](tester-api.md) 完成推理后，可以用这里的工具保存图像、打印分阶段耗时。需要直接调用网络或读取 checkpoint 时，参见下方接口。推理输出的形状和单位见[结果字典](results-api.md)。

## 保存可视化

```python
from HccePose.test_script_utils import save_visual_artifacts

save_visual_artifacts(artifacts, enabled=True)
```

`artifacts` 是 `(path, image)` 二元组的可迭代对象，图像采用 OpenCV BGR `H × W × 3`；保存测试器输出时使用 uint8 图像。`enabled=False` 时不写文件、返回 `[]`。函数跳过路径或图像为 `None` 的条目，创建父目录，再调用 `cv2.imwrite()`。

```python
# output_dir 是选定的本地输出目录。
import os

saved_paths = save_visual_artifacts([
    (os.path.join(output_dir, 'pose.jpg'), results.get('show_6D_vis2')),
    (os.path.join(output_dir, 'detection.jpg'), results.get('show_2D_results')),
])
```

函数允许覆盖同名文件，并会记录每个尝试写入的路径。它不检查 `cv2.imwrite()` 的返回值；需要确认写入成功时，请检查目标文件。

源码：[`save_visual_artifacts`](https://github.com/WangYuLin-SEU/HCCEPose/blob/main/HccePose/test_script_utils.py)。

## 打印阶段耗时

```python
from HccePose.test_script_utils import print_stage_time_breakdown

print_stage_time_breakdown(results_dict, enabled=False, prefix='')
```

设置 `enabled=True` 后打印 `time`、各阶段秒数和占比，返回 `None`。耗时字典缺失或格式不符时打印 unavailable 提示。百分比以 `results_dict['time']` 为分母，但可视化可能位于这一计时区间之外，所以不保证加总为 100%。见[计时定义](results-api.md)。

源码：[`print_stage_time_breakdown`](https://github.com/WangYuLin-SEU/HCCEPose/blob/main/HccePose/test_script_utils.py)。

## 底层可视化

```python
vis_rgb_mask_Coord(
    rgb_c, pred_mask, pred_front_code, pred_back_code, img_path=None,
)

vis_rgb_mask_Coord_origin(
    cam_K, obj_ids_l, obj_ids_all, BBox_3d_l, Rts_l, conf_s,
    rgb_c, pred_mask, pred_front_code, pred_back_code, img_path=None,
)
```

函数位于 `HccePose.visualization`。第一个返回裁剪及编码展示图，第二个返回 `(combined_canvas, compact_canvas)`，并投影模型包围盒。输入是归一化 BGR 张量和编码、掩码张量。第二个函数要求掩码与编码已映射到图像画布，位姿与 `BBox_3d_l` 使用相同模型坐标约定。

`vis_rgb_mask_Coord` 返回可视化图像，传入 `img_path` 时还会把图像保存到该路径。`vis_rgb_mask_Coord_origin` 返回两幅图像，保留了 `img_path` 参数，但不使用它写文件。常规推理可让 `Tester` 准备输入，再用上面的工具保存 `show_*` 图像。

源码：[裁剪图可视化](https://github.com/WangYuLin-SEU/HCCEPose/blob/main/HccePose/visualization.py)、[完整画布可视化](https://github.com/WangYuLin-SEU/HCCEPose/blob/main/HccePose/visualization.py)。

## 网络与 checkpoint

```python
from HccePose.network_model import HccePose_BF_Net

HccePose_BF_Net(
    efficientnet_key=None,
    input_channels=3,
    min_xyz=None,
    size_xyz=None,
)

network.inference_batch(inputs, Bbox, thershold=0.5)
```

`inputs` 是归一化 BGR 裁剪张量，通常为 `N × 3 × 256 × 256`。`Bbox` 为 `N × 4` 张量，内容是对应图像画布中的 `[x, y, width, height]`。`min_xyz` 和 `size_xyz` 应描述物体轴对齐模型边界，并与运算张量的设备匹配。

调用时按接口的拼写传入 **`thershold`**，用于对 sigmoid 后的掩码和编码设阈值。该接口返回稠密预测、解码后的模型坐标及 logits。要获得最终 `Rts`，还需要检测和 PnP，见[结果字段](results-api.md)。`efficientnet_key=None` 选择 ResNet，其余值选择 EfficientNet。

Checkpoint 工具也位于 `HccePose.network_model`：

```python
get_checkpoint(path)
load_checkpoint(
    check_point_path, net, optimizer=None, local_rank=0, CUDA_DEVICE='0',
)
```

两处路径都指向 checkpoint **目录**。`get_checkpoint()` 解析文件名 `step` 前面的分数，返回最高分文件路径。`load_checkpoint()` 加载 CUDA 权重及可选优化器状态，返回 `best_score`、`iteration_step`、`keypoints_`。传入优化器时必须同时恢复其状态；只有目录不存在或为空才从零开始新训练。已有 checkpoint 损坏或不兼容会停止并报告文件和原因。旧权重可缺少 keypoints_，但空先验只能用于不依赖先验的 PnP 模式。使用架构兼容的 checkpoint；不要将说明文件或旧权重混放在 `best_score/` 中。

源码：[网络构造](https://github.com/WangYuLin-SEU/HCCEPose/blob/main/HccePose/network_model.py)、[批推理](https://github.com/WangYuLin-SEU/HCCEPose/blob/main/HccePose/network_model.py)、[checkpoint 选择](https://github.com/WangYuLin-SEU/HCCEPose/blob/main/HccePose/network_model.py)、[checkpoint 加载](https://github.com/WangYuLin-SEU/HCCEPose/blob/main/HccePose/network_model.py)。

## 训练与评测边界

训练和评测各有独立的 Python 脚本，参数在脚本内设置。仓库没有统一的 `train()` 或 `evaluate()` 接口。下表脚本位于[仓库根目录](installation.md#公开训练与渲染源码)。

| 脚本 | 用途 |
| --- | --- |
| `s4_p2_train_bf_pbr.py` | 逐物体训练；`ide_debug=True` 为单卡，`False` 启用 DDP。数据集路径仍为占位值，需编辑。 |
| `s4_p2_test_bf_pbr.py` | 在已裁剪的 PBR 图像上验证网络预测和 PnP。 |
| `s4_p2_test_bf_pbr_bop_challenge.py` | 默认 LM-O 示例，使用外部 GDRNPP 检测，写入分割 JSON 和位姿 CSV，不调用官方评分器、不上传结果。 |

训练脚本使用 `total_iteration=50001`、`batch_size=24`。训练时从同一 `train_pbr` 目录抽取 `ratio=0.01` 的样本监控进度。评测独立测试集时请使用相应的评测脚本。

BOP 脚本使用 `efficientnet_key=1`、脚本列出的 LM-O 物体 ID 及该数据集的检测 JSON。它输出 `seg2d_<dataset>.json` 和 `det6d_<dataset>.csv` 供后续评分。两份文件的 `time=-1` 均为占位值；运行耗时需单独测量。

源码：[训练设置](https://github.com/WangYuLin-SEU/HCCEPose/blob/main/s4_p2_train_bf_pbr.py)、[PBR 验证](https://github.com/WangYuLin-SEU/HCCEPose/blob/main/s4_p2_test_bf_pbr.py)、[挑战设置](https://github.com/WangYuLin-SEU/HCCEPose/blob/main/s4_p2_test_bf_pbr_bop_challenge.py)、[挑战输出](https://github.com/WangYuLin-SEU/HCCEPose/blob/main/s4_p2_test_bf_pbr_bop_challenge.py)。
