# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

import os

import cv2


def save_visual_artifacts(artifacts, enabled=True):
    '''
    ---
    ---
    Save a batch of visualization images only when saving is enabled.

    Each array must be BGR uint8 (or float castable) in HxWx3 layout, matching
    ``cv2.imwrite`` so JPEG/PNG colors match common viewers.
    ---
    ---
    The helper keeps `s4_p3_test_*.py` scripts concise and makes it easy to
    disable all visualization outputs from one flag without changing the
    inference logic itself.

    Args:
        - artifacts: Iterable of `(path, image)` pairs.
        - enabled: Whether image files should actually be written.

    Returns:
        - saved_paths: Paths that were written to disk.
    ---
    ---
    仅在启用保存时，批量保存一组可视化图像。
    ---
    ---
    该辅助函数用于简化 `s4_p3_test_*.py` 脚本中的保存逻辑，
    让脚本可以通过一个总开关关闭所有可视化文件输出，而不影响推理本身。

    参数:
        - artifacts: `(path, image)` 二元组的可迭代对象。
        - enabled: 是否真的把图像写入磁盘。

    返回:
        - saved_paths: 实际写入磁盘的路径列表。
    '''
    if not enabled:
        return []

    saved_paths = []
    for path, image in artifacts:
        if path is None or image is None:
            continue
        parent = os.path.dirname(path)
        if len(parent) > 0:
            os.makedirs(parent, exist_ok=True)
        cv2.imwrite(path, image)
        saved_paths.append(path)
    return saved_paths


def print_stage_time_breakdown(results_dict, enabled=False, prefix=''):
    '''
    ---
    ---
    Print the time share of each inference stage from one results dictionary.
    ---
    ---
    The function expects the lightweight `time_dict` attached by `Tester` and
    prints both absolute time and percentage relative to the total inference
    time. It is designed for quick script-side benchmarking rather than formal
    profiling.

    Args:
        - results_dict: Output dictionary returned by `Tester.predict(...)`.
        - enabled: Whether the timing breakdown should be printed.
        - prefix: Optional label prepended to the printed block.
    ---
    ---
    根据一次推理的结果字典，打印各阶段耗时占比。
    ---
    ---
    该函数读取 `Tester` 附带的轻量级 `time_dict`，
    并输出各阶段的绝对耗时与相对总推理时间的百分比。
    它面向脚本侧的快速测速，而不是正式 profiler。

    参数:
        - results_dict: `Tester.predict(...)` 返回的结果字典。
        - enabled: 是否打印耗时占比。
        - prefix: 打印时附带的可选标签。
    '''
    if not enabled:
        return

    if not isinstance(results_dict, dict):
        print('%s stage timing: unavailable' % prefix if prefix else 'stage timing: unavailable')
        return

    total_time = float(results_dict.get('time', 0.0))
    time_dict = results_dict.get('time_dict', {})
    if not isinstance(time_dict, dict) or len(time_dict) == 0:
        print('%s stage timing: unavailable' % prefix if prefix else 'stage timing: unavailable')
        return

    title = '%s stage timing' % prefix if prefix else 'stage timing'
    print(title)
    print('  total: %.4f s' % total_time)
    for key in ['yolo', 'hccepose', 'foundationpose', 'megapose', 'visualization', 'other']:
        value = float(time_dict.get(key, 0.0))
        ratio = 0.0 if total_time <= 1e-8 else value / total_time * 100.0
        print('  %s: %.4f s (%.1f%%)' % (key, value, ratio))
