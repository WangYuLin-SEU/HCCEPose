# Video inference

Each frame is processed independently: detect objects, predict surface coordinates, and solve poses. Models are loaded once; no pose or tracking state passes between frames. First confirm the model with an [RGB image](rgb-inference.md), then process video.

## Video, weights and camera

Public input videos are available under [test_videos on Hugging Face](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/4e482e540314622981ccd4fce5beddac7f9d3e1d/test_videos). Model and checkpoint layouts match image inference. Videos may contain multiple target categories; prepare every model listed in the corresponding ID table.

The script below expects caller-supplied `input.mp4` and `video_camera_matrix.json`. The latter must be a 3×3 array calibrated for the video’s resolution. Do not reuse intrinsics from another camera or unrelated frame. Save the script in the project root.

## Predict and save frames

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

Models are constructed outside the loop. `VideoCapture` produces BGR frames suitable for `predict`. The writer uses the first actual visualization size and checks that subsequent dimensions remain stable. Convert the output codec separately if needed for your browser or system.

Update intrinsics whenever video resolution changes. `Tester` handles its internal resizing and padding; do not replace original-image intrinsics using output-canvas dimensions. For instance results, read each frame’s integer object keys and `pnp_success`; see [results](results-api.md).

## Playback rate and processing speed

The output retains source FPS for playback timing. This does not imply online inference at that speed. `results['time']` excludes model loading, video decoding and writing; whole-video runtime includes those operations. See [timing utilities](utilities-api.md).

## Public preview

Each scene below shows its input video followed by the corresponding predictions, first for bin picking and then for textured objects. All videos are converted from animations in the same public source revision to browser-friendly H.264. They preserve complete frames and original aspect ratios and loop automatically without sound; no additional data was captured.

Prediction previews show masks, poses, front/back surface coordinates and C1–C6 code levels. They illustrate predictions rather than measuring this machine’s speed.

### Bin-picking input

<video controls autoplay muted loop playsinline preload="none" poster="../../media/public-bin-input-poster.jpg"><source src="../../media/public-bin-input.mp4" type="video/mp4"></video>

### Bin-picking predictions

<video controls autoplay muted loop playsinline preload="none" width="600" height="240" poster="../../media/public-poster.jpg"><source src="../../media/public-result.mp4" type="video/mp4"></video>

### Textured-object input

<video controls autoplay muted loop playsinline preload="none" poster="../../media/public-textured-input-poster.jpg"><source src="../../media/public-textured-input.mp4" type="video/mp4"></video>

### Textured-object predictions

<video controls autoplay muted loop playsinline preload="none" poster="../../media/public-textured-result-poster.jpg"><source src="../../media/public-textured-result.mp4" type="video/mp4"></video>

Public source recipes: [bin-picking video](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/s4_p3_test_mi10_bin_picking_video.py) · [textured-object video](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/s4_p3_test_mi10_tex_objs_video.py). These scripts live in the upstream root; set their paths and calibration for the actual video.
