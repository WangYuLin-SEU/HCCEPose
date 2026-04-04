# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

import cv2, os, sys, tempfile, shutil
import numpy as np
from HccePose.bop_loader import bop_dataset
from HccePose.test_script_utils import print_stage_time_breakdown, save_visual_artifacts
from HccePose.tester import Tester
from Refinement.refinement_test_utils import build_depth_comparison_visual, load_capture_frame, list_capture_frame_names


def rotation_error_deg(R1, R2):
    delta = R1 @ R2.T
    cos_theta = (np.trace(delta) - 1.0) * 0.5
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def build_pose_comparison(results_fp, results_mp):
    summary = {}
    fp_obj_ids = sorted(key for key in results_fp.keys() if isinstance(key, int))
    mp_obj_ids = sorted(key for key in results_mp.keys() if isinstance(key, int))
    common_obj_ids = sorted(set(fp_obj_ids) & set(mp_obj_ids))
    for obj_id in common_obj_ids:
        fp_rts = np.asarray(results_fp[obj_id].get('Rts', []), dtype=np.float32)
        mp_rts = np.asarray(results_mp[obj_id].get('Rts', []), dtype=np.float32)
        pair_count = int(min(len(fp_rts), len(mp_rts)))
        pairs = []
        for pose_id in range(pair_count):
            rot_err = rotation_error_deg(fp_rts[pose_id, :3, :3], mp_rts[pose_id, :3, :3])
            trans_err_mm = float(np.linalg.norm(fp_rts[pose_id, :3, 3] - mp_rts[pose_id, :3, 3]))
            pairs.append({
                'pose_id': pose_id,
                'rotation_error_deg': rot_err,
                'translation_error_mm': trans_err_mm,
            })
        summary[str(obj_id)] = {
            'foundationpose_count': int(len(fp_rts)),
            'megapose_count': int(len(mp_rts)),
            'pair_count': pair_count,
            'pairs': pairs,
        }
    return summary


def summarize_pose_alignment(results_base, results_candidate, obj_id):
    base_rts = np.asarray(results_base.get(obj_id, {}).get('Rts', []), dtype=np.float32)
    candidate_rts = np.asarray(results_candidate.get(obj_id, {}).get('Rts', []), dtype=np.float32)
    pair_count = int(min(len(base_rts), len(candidate_rts)))
    if pair_count == 0:
        return {
            'pair_count': 0,
            'translation_mean_mm': None,
            'rotation_mean_deg': None,
        }

    translation_errors = []
    rotation_errors = []
    for pose_id in range(pair_count):
        translation_errors.append(float(np.linalg.norm(base_rts[pose_id, :3, 3] - candidate_rts[pose_id, :3, 3])))
        rotation_errors.append(rotation_error_deg(base_rts[pose_id, :3, :3], candidate_rts[pose_id, :3, :3]))

    return {
        'pair_count': pair_count,
        'translation_mean_mm': float(np.mean(translation_errors)),
        'rotation_mean_deg': float(np.mean(rotation_errors)),
    }


def add_title(image, title, height=28):
    if image is None:
        return None
    header = np.full((height, image.shape[1], 3), 18, dtype=np.uint8)
    cv2.putText(header, str(title), (8, int(height * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2, cv2.LINE_AA)
    return np.concatenate([header, image], axis=0)


def hstack_images(images):
    images = [image for image in images if image is not None]
    if len(images) == 0:
        return None
    max_h = max(image.shape[0] for image in images)
    padded = []
    for image in images:
        pad_h = max_h - image.shape[0]
        padded.append(cv2.copyMakeBorder(image, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=(18, 18, 18)))
    return np.concatenate(padded, axis=1)


def get_foundationpose_providers(tester_item):
    foundationpose_item = tester_item.FoundationPose_Item
    refiner_provider = ['PyTorch']
    scorer_provider = ['PyTorch']
    if foundationpose_item is None:
        return refiner_provider, scorer_provider

    refiner_runtime = getattr(foundationpose_item.refiner, 'runtime', None)
    if refiner_runtime is not None and hasattr(refiner_runtime, 'session'):
        refiner_provider = list(refiner_runtime.session.get_providers())

    scorer_runtime = getattr(foundationpose_item.scorer, 'runtime', None)
    if scorer_runtime is not None:
        sessions = getattr(scorer_runtime, '_sessions', {})
        if len(sessions) > 0:
            first_key = sorted(sessions.keys())[0]
            session_info = sessions[first_key]
            if isinstance(session_info, dict) and 'session' in session_info:
                scorer_provider = list(session_info['session'].get_providers())

    return refiner_provider, scorer_provider


def print_foundationpose_benchmark(current_dir, bop_dataset_item, cam_K, image, depth_m, obj_id,
                                   foundationpose_refine_dir, foundationpose_score_dir, CUDA_DEVICE,
                                   print_stage_timing=False):
    print('\n[FoundationPose Benchmark]')
    trt_cache_dir = tempfile.mkdtemp(prefix='.foundationpose_trt_bench_', dir=current_dir)
    try:
        backend_settings = [
            ('pytorch', 'pytorch', None),
            ('onnx', 'onnx', None),
            ('tensorrt first-run', 'tensorrt', trt_cache_dir),
            ('tensorrt warm-run', 'tensorrt', trt_cache_dir),
        ]
        benchmark_results = {}
        pytorch_results = None

        for label, backend, cache_dir in backend_settings:
            runner = Tester(
                bop_dataset_item,
                hccepose_vis=False,
                CUDA_DEVICE=CUDA_DEVICE,
                foundationpose_refine_dir=foundationpose_refine_dir,
                foundationpose_score_dir=foundationpose_score_dir,
                foundationpose_acceleration=backend,
                foundationpose_acceleration_cache_dir=cache_dir,
            )
            results = runner.predict(
                cam_K,
                image,
                [obj_id],
                conf=0.85,
                confidence_threshold=0.85,
                depth=depth_m,
                use_foundationpose=True,
                foundationpose_vis=False,
            )
            refiner_provider, scorer_provider = get_foundationpose_providers(runner)
            benchmark_results[label] = {
                'results': results,
                'time': float(results.get('time', 0.0)),
                'refiner_provider': refiner_provider,
                'scorer_provider': scorer_provider,
            }
            if backend == 'pytorch' and pytorch_results is None:
                pytorch_results = results

        for label in ['pytorch', 'onnx', 'tensorrt first-run', 'tensorrt warm-run']:
            item = benchmark_results[label]
            print('%s: %.4f s' % (label, item['time']))
            print('  refiner provider: %s' % ', '.join(item['refiner_provider']))
            print('  scorer provider: %s' % ', '.join(item['scorer_provider']))
            print_stage_time_breakdown(item['results'], enabled=print_stage_timing, prefix='FoundationPose %s' % label)
            if label != 'pytorch' and pytorch_results is not None:
                alignment = summarize_pose_alignment(pytorch_results, item['results'], obj_id)
                if alignment['pair_count'] == 0:
                    print('  pose alignment: no overlapping poses')
                else:
                    print('  pose alignment: pairs=%d, trans_mean=%.4f mm, rot_mean=%.4f deg' % (
                        alignment['pair_count'],
                        alignment['translation_mean_mm'],
                        alignment['rotation_mean_deg'],
                    ))
    finally:
        shutil.rmtree(trt_cache_dir, ignore_errors=True)


if __name__ == '__main__':

    sys.path.insert(0, os.getcwd())
    current_dir = os.path.dirname(sys.argv[0])
    dataset_path = os.path.join(current_dir, 'demo-bin-picking')
    capture_dir = os.path.join(current_dir, 'test_imgs_RGBD')
    foundationpose_refine_dir = os.path.join(current_dir, '2023-10-28-18-33-37')
    foundationpose_score_dir = os.path.join(current_dir, '2024-01-11-20-02-45')
    bop_dataset_item = bop_dataset(dataset_path)
    obj_id = 1
    obj_index = list(bop_dataset_item.obj_id_list).index(obj_id)
    obj_model_path = bop_dataset_item.obj_model_list[obj_index]
    CUDA_DEVICE = '0'
    hccepose_vis = True
    hccepose_acceleration = 'pytorch'
    foundationpose_vis = True
    foundationpose_vis_stages = [1, 2, 3, 4, 5, 'score']
    foundationpose_acceleration = 'onnx'
    megapose_vis = True
    megapose_vis_stages = [1, 2, 3, 4, 5]
    megapose_variants = [('rgbd', True), ('rgb', False)]
    save_visualizations = hccepose_vis or foundationpose_vis or megapose_vis
    print_stage_timing = False

    foundationpose_runner = Tester(
        bop_dataset_item,
        hccepose_vis=hccepose_vis,
        CUDA_DEVICE=CUDA_DEVICE,
        foundationpose_refine_dir=foundationpose_refine_dir,
        foundationpose_score_dir=foundationpose_score_dir,
        hccepose_acceleration=hccepose_acceleration,
        foundationpose_acceleration=foundationpose_acceleration,
    )
    megapose_runner = Tester(
        bop_dataset_item,
        hccepose_vis=hccepose_vis,
        CUDA_DEVICE=CUDA_DEVICE,
        hccepose_acceleration=hccepose_acceleration,
    )

    frame_names = list_capture_frame_names(capture_dir)
    if len(frame_names) > 0:
        benchmark_name = frame_names[0]
        benchmark_image, benchmark_depth, benchmark_depth_m, benchmark_cam_K = load_capture_frame(capture_dir, benchmark_name)
        print_foundationpose_benchmark(
            current_dir,
            bop_dataset_item,
            benchmark_cam_K,
            benchmark_image,
            benchmark_depth_m,
            obj_id,
            foundationpose_refine_dir,
            foundationpose_score_dir,
            CUDA_DEVICE,
            print_stage_timing=print_stage_timing,
        )

    for name in frame_names:
        image, depth, depth_m, cam_K = load_capture_frame(capture_dir, name)

        results_hccepose = foundationpose_runner.predict(
            cam_K,
            image,
            [obj_id],
            conf=0.85,
            confidence_threshold=0.85,
            depth=depth_m,
        )
        results_fp = foundationpose_runner.predict(
            cam_K,
            image,
            [obj_id],
            conf=0.85,
            confidence_threshold=0.85,
            depth=depth_m,
            use_foundationpose=True,
            foundationpose_vis=foundationpose_vis,
            foundationpose_vis_stages=foundationpose_vis_stages,
        )

        print_stage_time_breakdown(results_hccepose, enabled=print_stage_timing, prefix='%s | HccePose' % name)
        print_stage_time_breakdown(results_fp, enabled=print_stage_timing, prefix='%s | FoundationPose' % name)

        save_visual_artifacts([
            (os.path.join(capture_dir, '%s_hccepose_show_2d.jpg' % name), results_hccepose.get('show_2D_results')),
            (os.path.join(capture_dir, '%s_hccepose_show_6d_vis0.jpg' % name), results_hccepose.get('show_6D_vis0')),
            (os.path.join(capture_dir, '%s_hccepose_show_6d_vis1.jpg' % name), results_hccepose.get('show_6D_vis1')),
            (os.path.join(capture_dir, '%s_hccepose_show_6d_vis2.jpg' % name), results_hccepose.get('show_6D_vis2')),
            (os.path.join(capture_dir, '%s_foundationpose_show_2d.jpg' % name), results_fp.get('show_2D_results')),
            (os.path.join(capture_dir, '%s_foundationpose_show_6d_vis0.jpg' % name), results_fp.get('show_6D_vis0')),
            (os.path.join(capture_dir, '%s_foundationpose_show_6d_vis1.jpg' % name), results_fp.get('show_6D_vis1')),
            (os.path.join(capture_dir, '%s_foundationpose_show_6d_vis2.jpg' % name), results_fp.get('show_6D_vis2')),
            (os.path.join(capture_dir, '%s_show_foundationpose.jpg' % name), results_fp.get('show_foundationpose')),
        ], enabled=save_visualizations)

        for megapose_variant_name, megapose_use_depth in megapose_variants:
            megapose_depth = depth_m if megapose_use_depth else None
            results_mp = megapose_runner.predict(
                cam_K,
                image,
                [obj_id],
                conf=0.85,
                confidence_threshold=0.85,
                depth=megapose_depth,
                use_megapose=True,
                megapose_vis=megapose_vis,
                megapose_vis_stages=megapose_vis_stages,
            )

            if 'show_megapose' in results_fp or any(isinstance(v, dict) and 'megapose_vis' in v for v in results_fp.values()):
                raise RuntimeError('FoundationPose branch unexpectedly produced MegaPose visualization keys.')
            if 'show_foundationpose' in results_mp or any(isinstance(v, dict) and 'foundationpose_vis' in v for v in results_mp.values()):
                raise RuntimeError('MegaPose branch unexpectedly produced FoundationPose visualization keys.')

            print_stage_time_breakdown(results_mp, enabled=print_stage_timing, prefix='%s | MegaPose %s' % (name, megapose_variant_name.upper()))

            save_visual_artifacts([
                (os.path.join(capture_dir, '%s_show_megapose_%s.jpg' % (name, megapose_variant_name)), results_mp.get('show_megapose')),
                (os.path.join(capture_dir, '%s_megapose_%s_show_2d.jpg' % (name, megapose_variant_name)), results_mp.get('show_2D_results')),
                (os.path.join(capture_dir, '%s_megapose_%s_show_6d_vis0.jpg' % (name, megapose_variant_name)), results_mp.get('show_6D_vis0')),
                (os.path.join(capture_dir, '%s_megapose_%s_show_6d_vis1.jpg' % (name, megapose_variant_name)), results_mp.get('show_6D_vis1')),
                (os.path.join(capture_dir, '%s_megapose_%s_show_6d_vis2.jpg' % (name, megapose_variant_name)), results_mp.get('show_6D_vis2')),
            ], enabled=save_visualizations)

            pose_sets_mm = {}
            if obj_id in results_hccepose and 'Rts' in results_hccepose[obj_id]:
                pose_sets_mm['HccePose'] = results_hccepose[obj_id]['Rts']
            if obj_id in results_fp and 'Rts' in results_fp[obj_id]:
                pose_sets_mm['FoundationPose'] = results_fp[obj_id]['Rts']
            if obj_id in results_mp and 'Rts' in results_mp[obj_id]:
                pose_sets_mm['MegaPose'] = results_mp[obj_id]['Rts']
            if save_visualizations:
                depth_compare_vis, depth_compare_summary = build_depth_comparison_visual(
                    depth,
                    cam_K,
                    obj_model_path,
                    pose_sets_mm,
                    device=str(foundationpose_runner.device),
                    max_items=4,
                )
                save_visual_artifacts([
                    (os.path.join(capture_dir, '%s_compare_depth_hccepose_foundationpose_megapose_%s.jpg' % (name, megapose_variant_name)), depth_compare_vis),
                ], enabled=True)
