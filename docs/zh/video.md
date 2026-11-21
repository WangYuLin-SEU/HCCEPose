# 视频推理

视频中的每一帧独立检测物体、预测表面坐标并求解位姿。模型只加载一次，各帧之间不传递位姿和跟踪状态。先用[RGB 单图](rgb-inference.md)确认模型可用，再处理视频。

## 视频、权重与相机

公开测试视频位于 [Hugging Face 的 test_videos](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/4e482e540314622981ccd4fce5beddac7f9d3e1d/test_videos)。物体模型和权重的目录要求与单图一致。视频可能包含多个目标物体，应加载相应编号表中的全部模型。

下面脚本的 `input.mp4` 和 `video_camera_matrix.json` 是用户需要准备的文件。相机 JSON 应为该视频分辨率对应的 3×3 数组；不要直接套用另一相机或另一帧的内参。将脚本保存在项目根目录。

## 逐帧推理与保存

```python
import json
import os
import sys

project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

import cv2
import numpy as np
from HccePose.bop_loader import bop_dataset
from HccePose.tester import Tester

if __name__ == '__main__':
    data_root = os.path.join(project_dir, 'data')
    dataset_root = os.path.join(data_root, 'demo-bin-picking')
    video_path = os.path.join(data_root, 'test_videos', 'input.mp4')
    camera_path = os.path.join(data_root, 'video_camera_matrix.json')
    output_path = os.path.join(project_dir, 'outputs', 'video', 'poses.mp4')
    object_ids = [1]
    with open(camera_path, encoding='utf-8') as handle:
        cam_K = np.asarray(json.load(handle), dtype=np.float32)
    dataset = bop_dataset(dataset_root)
    tester = Tester(dataset, CUDA_DEVICE='0', hccepose_vis=True)
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise OSError(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError('Invalid source frame rate')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = None
    try:
        while True:
            ok, image_bgr = capture.read()
            if not ok:
                break
            results = tester.predict(cam_K, image_bgr, object_ids)
            canvas = results['show_6D_vis1']
            height, width = canvas.shape[:2]
            if writer is None:
                output_size = (width, height)
                writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, output_size)
                if not writer.isOpened():
                    raise OSError(output_path)
            if (width, height) != output_size:
                raise ValueError('Visualization size changed within the video')
            writer.write(canvas)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
```

模型在循环外创建。`VideoCapture` 返回 BGR 帧，可直接传给 `predict`。写入器根据第一帧实际拼图尺寸创建，并检查后续尺寸不变；视频输出编码可按浏览器和系统支持另行转换。

如果视频分辨率改变，先同步更新内参。模型内部的缩放补边由 `Tester` 处理，不要额外用输出画布尺寸替换原图内参。需要逐实例结果时，读取每帧结果中的整数物体键与 `pnp_success`，见[结果字典](results-api.md)。

## 理解播放帧率与运行速度

输出视频沿用输入 FPS，表示回放时的时间尺度，不代表模型能以该速度在线处理。`results['time']` 不含模型加载、视频读取和写入，实际整段耗时还应包括这些操作。阶段计时见[工具接口](utilities-api.md)。

## 公开示例预览

下面按“输入视频 → 对应预测”的顺序展示料箱和纹理物体两个场景，方便对照输入及输出中的物体区域。视频均由同一公开源码版本的动图转换为 H.264，保留完整画面和原始比例，静音自动循环；不是另行采集的数据。

预测预览显示掩码、位姿、前后表面坐标与 C1–C6 六层编码，用于展示预测形式，不是本机速度测量。

### 料箱输入

<video controls autoplay muted loop playsinline preload="none" poster="../../media/public-bin-input-poster.jpg"><source src="../../media/public-bin-input.mp4" type="video/mp4"></video>

### 料箱预测

<video controls autoplay muted loop playsinline preload="none" width="600" height="240" poster="../../media/public-poster.jpg"><source src="../../media/public-result.mp4" type="video/mp4"></video>

### 纹理物体输入

<video controls autoplay muted loop playsinline preload="none" poster="../../media/public-textured-input-poster.jpg"><source src="../../media/public-textured-input.mp4" type="video/mp4"></video>

### 纹理物体预测

<video controls autoplay muted loop playsinline preload="none" poster="../../media/public-textured-result-poster.jpg"><source src="../../media/public-textured-result.mp4" type="video/mp4"></video>

公开原始脚本：[料箱视频](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/s4_p3_test_mi10_bin_picking_video.py) · [纹理物体视频](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/s4_p3_test_mi10_tex_objs_video.py)。这些脚本位于上游根目录，输入路径和标定需按实际视频设置。
