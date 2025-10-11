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
    for name in ['VID_20251009_141606',
                 'VID_20251009_141731',
                 'VID_20251009_141754',
                 'VID_20251009_141902',
                 'VID_20251009_141948',]:
        file_name = '/root/xxxxxx/test_videos/%s.mp4'%name
        
        
        cap = cv2.VideoCapture(file_name)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4格式
        out_1 = None
        out_2 = None
        
        
        cam_K = np.array([
            [1.63235512e+03, 0.00000000e+00, 9.74032712e+02],
            [0.00000000e+00, 1.64159967e+03, 5.14229781e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
        ])
        
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            results_dict = Tester_item.perdict(cam_K, frame, obj_ids,
                                                            conf = 0.85, confidence_threshold = 0.85)
            
            fps_hccepose = 1 / results_dict['time']
            
            show_6D_vis1 = results_dict['show_6D_vis1']
            show_6D_vis1[show_6D_vis1 < 0] = 0
            show_6D_vis1[show_6D_vis1 > 255] = 255
            # frame = cv2.resize(frame, (int(frame.shape[1]/frame.shape[0] * show_6D_vis1.shape[0]), show_6D_vis1.shape[0]))
            # show_6D_vis1 = np.concatenate([frame, show_6D_vis1.astype(np.uint8)], axis = 1)
            if out_1 is None:
                out_1 = cv2.VideoWriter(
                    file_name.replace('.mp4', '_show_1.mp4'),
                    fourcc,
                    fps,
                    (show_6D_vis1.shape[1], show_6D_vis1.shape[0])
                )
            # cv2.putText(show_6D_vis1, "FPS: {0:.2f}".format(fps_hccepose), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
            out_1.write(show_6D_vis1.astype(np.uint8))
                
            show_6D_vis2 = results_dict['show_6D_vis2']
            show_6D_vis2[show_6D_vis2 < 0] = 0
            show_6D_vis2[show_6D_vis2 > 255] = 255
            if out_2 is None:
                out_2 = cv2.VideoWriter(
                    file_name.replace('.mp4', '_show_2.mp4'),
                    fourcc,
                    fps,
                    (show_6D_vis2.shape[1], show_6D_vis2.shape[0])
                )
            cv2.putText(show_6D_vis2, "FPS: {0:.2f}".format(fps_hccepose), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4, cv2.LINE_AA)
            out_2.write(show_6D_vis2.astype(np.uint8))
        cap.release()
        out_1.release()
        out_2.release()
        
    pass