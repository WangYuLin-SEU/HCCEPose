# 结果与坐标

## HccePose

`Tester.predict` 按物体编号返回结果。每组的 `pnp_success[i]` 表示第 `i` 个检测实例的 HccePose PnP 是否求解成功。读取 RGB 位姿前先检查这个布尔值。可用对应点不足时，求解器可能返回单位矩阵，此时 `pnp_success[i]` 为 `False`。启用可选精修时，这个标志仍只记录初始 PnP 状态。

`Rts[i]` 为物体到相机的 4×4 变换，平移单位为毫米。`pred_mask[i]` 位于裁剪图像坐标系。还原到原图时，使用已有辅助函数完成裁剪逆变换、去除补边和分辨率恢复：

```python
from Refinement.wapr_public import restore_full_frame_masks

mask_i = restore_full_frame_masks(results[obj_id], image_bgr.shape)[i]
```

`image_bgr` 为传入 `predict` 的原始图像，`i` 为该物体组内的实例索引。函数使用匹配的 `Detect_Bbox_tfs_128`，返回原图大小的布尔掩码。

## WAPR

`pose_4x4_m` 表示物体到相机的位姿，平移单位为米。接入示例的 `score_6d` 由 WAPR 根据本次输入计算，并非 YOLO 检测置信度，不保证位于概率区间。

输入与输出保持相同实例顺序；示例保存的 `instance_index` 是物体组内索引，不代表检测器全局框编号或真实物体标签。

坐标和颜色顺序变换均在接入边界显式进行，原始 `Tester` 的毫米约定不变。

## 每个物体的完整字段

以下形状使用默认 `crop_size=256`；`N` 为当前物体的实例数。张量可能位于 GPU，转为 NumPy 前先 `detach().cpu()`。

| 字段 | 类型 / 形状 | 含义 |
| --- | --- | --- |
| `Rts` | NumPy `N × 4 × 4` | 物体到相机位姿，平移为毫米；MegaPose 可覆盖成功实例。 |
| `pred_mask` | torch `N × 128 × 128` | 网络裁剪空间内的二值掩码。 |
| `coord_2d_image` | torch `N × 128 × 128 × 2` | 缩放及补边后输入画布中的像素坐标。 |
| `pred_front_code_obj`、`pred_back_code_obj` | torch `N × 128 × 128 × 3` | 物体坐标系下的正背面解码坐标，使用模型单位。 |
| `pred_front_code`、`pred_back_code` | torch `N × 128 × 128 × 3` | 解码并归一化到 `[0, 1]` 的坐标。 |
| `pred_front_code_raw`、`pred_back_code_raw` | torch `N × 128 × 128 × 24` | 将 `(logits + 1) / 2` 截断到 `[0, 1]` 的展示值，不是未经修改的 logits。 |
| `pred_mask_logits` | torch `N × 1 × 128 × 128` | 掩码 logits。 |
| `pred_front_back_code_logits` | torch `N × 48 × 128 × 128` | 正背面编码 logits。 |
| `Detect_Bbox_tfs_128` | torch `N × 3 × 3` | 从处理后画布映射到网络输出裁剪图的单应变换。 |
| `crop_rgbs` | torch `N × 3 × 256 × 256` | BGR 裁剪图，使用按 BGR 重排的 ImageNet 统计量归一化。 |
| `pnp_success` | NumPy bool `N` | 初始 HccePose PnP 的成功状态。 |
| `conf` | torch `N` | YOLO 检测分数；不是 WAPR 评分。 |

## 可视化与耗时

| 顶层键 | 含义 |
| --- | --- |
| `show_2D_results` | YOLO 检测框图。 |
| `show_6D_vis0` | 实例裁剪、掩码与前后表面编码拼图。 |
| `show_6D_vis1` | 处理后整图上的坐标和位姿组合展示。 |
| `show_6D_vis2` | 紧凑的整图位姿展示。 |
| `show_megapose` | 请求并生成 MegaPose 调试图时出现。 |
| `time` | 从检测前到位姿计算完成的秒数，不含模型初始化和之后的可视化。 |
| `time_dict` | `yolo`、`hccepose`、`megapose`、`visualization`、`other` 的墙钟计时。 |

四个基础图像字段需要 `hccepose_vis=True`。图像为 BGR，可含多个拼接面板，尺寸不一定等于原图。`time_dict` 包含可视化，而 `time` 在可视化前结束，不能要求两者相加一致；这些计时也不等同于同步 GPU 基准。

MegaPose 成功返回评分时，`Rts_hccepose` 保留初值，`Rts` 更新成功的位姿；`megapose_scores` 与 `megapose_valid_pose_ids` 对齐，不应按全部实例下标直接索引。`pnp_success` 仍只表示原始 PnP 状态。见[精修接口](refinement-api.md)。

```python
# 转为米制平移时显式换算，旋转保持不变。
entry = results.get(1)
if entry is not None:
    translations_m = entry['Rts'][:, :3, 3] * 1e-3
```
