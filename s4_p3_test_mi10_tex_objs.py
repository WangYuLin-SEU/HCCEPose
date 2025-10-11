import cv2
import numpy as np
from HccePose.tester import Tester
from HccePose.bop_loader import bop_dataset

if __name__ == '__main__':
    
    dataset_path = '/root/xxxxxx/demo-tex-objs'
    bop_dataset_item = bop_dataset(dataset_path)
    CUDA_DEVICE = '0'
    # show_op = False
    show_op = True
    Tester_item = Tester(bop_dataset_item, show_op = show_op, CUDA_DEVICE=CUDA_DEVICE)
    obj_ids = bop_dataset_item.obj_id_list
    for name in ['IMG_20251009_142305',
                 'IMG_20251009_142310',
                 'IMG_20251009_142316',
                 'IMG_20251009_142319']:
        file_name = '/root/xxxxxx/test_imgs/%s.jpg'%name
        image = cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_RGB2BGR)
        cam_K = np.array([
            [2.83925618e+03, 0.00000000e+00, 2.02288638e+03],
            [0.00000000e+00, 2.84037288e+03, 1.53940473e+03],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
        ])
        results_dict = Tester_item.perdict(cam_K, image, obj_ids,
                                                        conf = 0.85, confidence_threshold = 0.85)
        cv2.imwrite(file_name.replace('.jpg','_show_2d.jpg'), results_dict['show_2D_results'])
        cv2.imwrite(file_name.replace('.jpg','_show_6d_vis0.jpg'), results_dict['show_6D_vis0'])
        cv2.imwrite(file_name.replace('.jpg','_show_6d_vis1.jpg'), results_dict['show_6D_vis1'])
        cv2.imwrite(file_name.replace('.jpg','_show_6d_vis2.jpg'), results_dict['show_6D_vis2'])
    pass