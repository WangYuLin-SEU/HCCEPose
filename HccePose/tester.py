import os, torch, kornia, time, cv2, random
import numpy as np 
from ultralytics import YOLO
from HccePose.bop_loader import bop_dataset
from HccePose.network_model import HccePose_BF_Net
from HccePose.network_model import load_checkpoint
import torchvision.transforms as transforms
from HccePose.visualization import vis_rgb_mask_Coord, vis_rgb_mask_Coord_origin
from HccePose.PnP_solver import solve_PnP, solve_PnP_comb

composed_transforms_img = transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

def draw_annotations_on_image_yolo(image, boxes, confidences, cls, class_names, confidence_threshold=0.5):
    for box, confidence, cl in zip(boxes, confidences, cls):
        if confidence >= confidence_threshold:
            x_min, y_min, x_max, y_max = box[0], box[1], box[0] + box[2], box[1] + box[3]
            random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), random_color, 2)
            text = f"{'obj_%s'%str(class_names[cl]).rjust(2, '0')}: {confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            font_color = random_color
            thickness = 2
            text_position = (x_min, y_min - 2)
            cv2.putText(image, text, text_position, font, font_scale, font_color, thickness)
    return image

def crop_trans_batch_hccepose(xywh, out_size = [128, 128], padding_ratio = 1.2):

    def compute_tf_batch(left, right, top, bottom):
        B = len(left)
        left = left.round()
        right = right.round()
        top = top.round()
        bottom = bottom.round()

        tf = torch.eye(3, device=xywh.device)[None].expand(B,-1,-1).contiguous()
        tf[:,0,2] = -left
        tf[:,1,2] = -top
        new_tf = torch.eye(3, device=xywh.device)[None].expand(B,-1,-1).contiguous()
        new_tf[:,0,0] = out_size[0]/(right-left)
        new_tf[:,1,1] = out_size[1]/(bottom-top)
        tf = new_tf@tf
        
        boxes = torch.cat([left[...,None], top[...,None], right[...,None] - left[...,None], bottom[...,None] - top[...,None]], dim=1)
        return tf, boxes

    center = xywh[:, :2].clone().detach()
    center[:, 0] += xywh[:, 2] / 2
    center[:, 1] += xywh[:, 3] / 2
    
    radius, _ = torch.max(xywh[:, 2:] * padding_ratio / 2, dim=1) #
    left = center[:,0]-radius
    right = center[:,0]+radius
    top = center[:,1]-radius
    bottom = center[:,1]+radius
    tfs, boxes = compute_tf_batch(left, right, top, bottom)
    return tfs, boxes

class Tester():
    
    def __init__(self, bop_dataset_item : bop_dataset, show_op = True, CUDA_DEVICE='0', top_K = None, crop_size = 256, efficientnet_key=None):
        self.bop_dataset_item = bop_dataset_item
        self.crop_size = crop_size
        self.top_K = top_K
        self.show_op = show_op
        self.CUDA_DEVICE = CUDA_DEVICE
        if torch.cuda.is_available():
            self.device = device = torch.device("cuda:%s"%CUDA_DEVICE)
            print("GPU is available. Using GPU.")
        else:
            self.device = device = torch.device("cpu")
            print("GPU is not available. Using CPU.")
            
        self.model_yolo = YOLO(os.path.join(bop_dataset_item.dataset_path, 'yolo11', 
                                            'train_obj_s', 'detection', 'obj_s', 
                                            'yolo11-detection-obj_s.pt')).to(device).eval()

        BBox_3d = []
        for key_i in self.bop_dataset_item.model_info:
            model_info_i = self.bop_dataset_item.model_info[key_i]
            min_x = model_info_i['min_x']
            min_y = model_info_i['min_y']
            min_z = model_info_i['min_z']
            size_x = model_info_i['size_x']
            size_y = model_info_i['size_y']
            size_z = model_info_i['size_z']
            max_x = min_x + size_x
            max_y = min_y + size_y
            max_z = min_z + size_z
            pts_3d = np.array([
                [min_x, min_y, min_z],
                [max_x, min_y, min_z],
                [max_x, max_y, min_z],
                [min_x, max_y, min_z],
                [min_x, min_y, max_z],
                [max_x, min_y, max_z],
                [max_x, max_y, max_z],
                [min_x, max_y, max_z]
            ])
            edges = np.array([
                [0, 1], [1, 2], [2, 3], [3, 0],
                [4, 5], [5, 6], [6, 7], [7, 4],
                [0, 4], [1, 5], [2, 6], [3, 7]
            ])
            BBox_3d.append(pts_3d[edges])
        self.BBox_3d = np.array(BBox_3d)
            
            
        self.HccePose_Item = {}
        self.HccePose_Item_info = {}
        for obj_id in bop_dataset_item.obj_id_list:
            obj_info = bop_dataset_item.obj_info_list[bop_dataset_item.obj_id_list.index(obj_id)]
            min_xyz = torch.from_numpy(np.array([obj_info['min_x'], obj_info['min_y'], obj_info['min_z']],dtype=np.float32)).to('cuda:'+CUDA_DEVICE)
            size_xyz = torch.from_numpy(np.array([obj_info['size_x'], obj_info['size_y'], obj_info['size_z']],dtype=np.float32)).to('cuda:'+CUDA_DEVICE)

            HccePose_BF_Net_i = HccePose_BF_Net(efficientnet_key=efficientnet_key, 
                                                min_xyz = min_xyz, 
                                                size_xyz = size_xyz)
            if torch.cuda.is_available():
                HccePose_BF_Net_i=HccePose_BF_Net_i.to('cuda:'+CUDA_DEVICE)
                HccePose_BF_Net_i.eval()
            best_save_path = os.path.join(bop_dataset_item.dataset_path, 'HccePose', 'obj_%s'%str(obj_id).rjust(2, '0'), 'best_score')
            info_i = load_checkpoint(best_save_path, HccePose_BF_Net_i, CUDA_DEVICE=CUDA_DEVICE)
            self.HccePose_Item[obj_id] = HccePose_BF_Net_i
            self.HccePose_Item_info[obj_id] = info_i
            

        pass


    @torch.inference_mode()
    def perdict(self, cam_K, img, obj_ids, confidence_threshold=0.75, conf=0.75, iou=0.50, max_det=200, pnp_op='ransac+comb'):
        pnp_op_l = [['epnp', 'ransac', 'ransac+vvs', 'ransac+comb', 'ransac+vvs+comb'],[0,2,1]]
        height, width = img.shape[:2]
        ratio_ = 1
        ratio_ = max(height / (640*1), width / (640*1))
        pad_s = 32
        if ratio_ > 1:
            height_new = int(height / ratio_)
            width_new = int(width / ratio_)
            img = cv2.resize(img, (width_new, height_new), cv2.INTER_LINEAR)
            cam_K = cam_K.copy()
            cam_K[:2, :] /= ratio_
        height, width = img.shape[:2]
        if height % pad_s == 0:
            height_pad = height
            h_move = 0
        else:
            height_pad = (int(height / pad_s) + 1) * pad_s
            h_move = int((height_pad - height) / 2)
        if width % pad_s == 0:
            width_pad = width
            w_move = 0
        else:
            width_pad = (int(width / pad_s) + 1) * pad_s
            w_move = int((width_pad - width) / 2)
        if w_move > 0 or h_move > 0:
            img_new = np.zeros((height_pad, width_pad, 3), dtype=img.dtype)
            img_new[h_move:h_move + height, w_move:w_move + width, :] = img
            img = img_new
        cam_K = cam_K.copy()
        cam_K[0, 2] += w_move
        cam_K[1, 2] += h_move
        
        img_torch = torch.from_numpy(img.astype(np.float32)).to(self.device)
        
        img_torch = img_torch[None]

        with torch.amp.autocast('cuda'):
            t1 = time.time()
            
            det_results = {}
            results_dict = {}
            
            scaled_img_torch = kornia.geometry.transform.resize(img_torch.permute(0,3,1,2), (int(img_torch.shape[1] * 1.0), int(img_torch.shape[2] * 1.0)), interpolation='bilinear')
            det_yolo = self.model_yolo(scaled_img_torch.clamp(0,255)/255, 
                                       conf=conf, iou=iou, max_det=max_det, 
                                    #    imgsz=640 * 1
                                       )
            xywh = det_yolo[0].boxes.xywh.clone().detach()
            cls = det_yolo[0].boxes.cls.clone().detach()
            xywh[:, 0] -= xywh[:, 2] / 2 # left, top, width, height
            xywh[:, 1] -= xywh[:, 3] / 2 # left, top, width, height
            det_results['xywh'] = xywh
            det_results['confs'] = confs = det_yolo[0].boxes.conf.clone().detach()
            det_results['cls'] = cls
            
            

            for key_i in self.bop_dataset_item.obj_id_list:
                if key_i in obj_ids:
                    obj_id = key_i
                    Rt_list = []
                    padding_ratio = 1.5
                    xywh_s = xywh[cls == self.bop_dataset_item.obj_id_list.index(obj_id)]
                    conf_s = confs[cls == self.bop_dataset_item.obj_id_list.index(obj_id)]
                    if xywh_s.shape[0] > 0:
                        crop_size = self.crop_size
                        Detect_Bbox_tfs, _ = crop_trans_batch_hccepose(xywh_s, out_size=[crop_size,crop_size], padding_ratio=padding_ratio)
                        Detect_Bbox_tfs_128, boxes_128 = crop_trans_batch_hccepose(xywh_s, out_size=[int(crop_size/2),int(crop_size/2)], padding_ratio=padding_ratio)
                        img_torch_hccepose = composed_transforms_img(img_torch.permute(0,3,1,2) / 255).flip(dims = [1])
                        crop_rgbs = kornia.geometry.transform.warp_perspective(img_torch_hccepose.repeat(xywh_s.shape[0],1,1,1), Detect_Bbox_tfs, dsize=[crop_size,crop_size], mode='bilinear', align_corners=False)
                        pred_results = self.HccePose_Item[obj_id].inference_batch(crop_rgbs, boxes_128)
                        
                        
                        pred_results['Detect_Bbox_tfs_128'] = Detect_Bbox_tfs_128
                        pred_results['conf'] = conf_s
                        pred_results['crop_rgbs'] = crop_rgbs
                        pred_mask = pred_results['pred_mask']
                        coord_image = pred_results['coord_2d_image']
                        pred_front_code_0 = pred_results['pred_front_code_obj']
                        pred_back_code_0 = pred_results['pred_back_code_obj']
                        pred_front_code = pred_results['pred_front_code']
                        pred_back_code = pred_results['pred_back_code']
                        
                        pred_mask_np = pred_mask.detach().cpu().numpy()
                        pred_front_code_0_np = pred_front_code_0.detach().cpu().numpy()
                        pred_back_code_0_np = pred_back_code_0.detach().cpu().numpy()
                        results = []
                        coord_image_np = coord_image.detach().cpu().numpy()
                        
                        if pnp_op in ['epnp', 'ransac', 'ransac+vvs']:
                            pred_m_f_c_np = [(pred_mask_np[i], pred_front_code_0_np[i], coord_image_np[i], cam_K) for i in range(pred_mask_np.shape[0])]
                            for pred_m_f_c_np_i in pred_m_f_c_np:
                                result_i = solve_PnP(pred_m_f_c_np_i, pnp_op=pnp_op_l[1][pnp_op_l[0].index(pnp_op)])
                                results.append(result_i)
                                Rt_i = np.eye(4)
                                Rt_i[:3, :3] = result_i['rot']
                                Rt_i[:3, 3:] = result_i['tvecs']
                                Rt_list.append(Rt_i)
                        else:
                            pred_m_bf_c_np = [(pred_mask_np[i], pred_front_code_0_np[i], pred_back_code_0_np[i], coord_image_np[i], cam_K) for i in range(pred_mask_np.shape[0])]
                            for pred_m_bf_c_np_i in pred_m_bf_c_np:
                                if pnp_op == 'ransac+comb':
                                    pnp_op_0 = 2
                                else:
                                    pnp_op_0 = 1
                                result_i = solve_PnP_comb(pred_m_bf_c_np_i, self.HccePose_Item_info[obj_id]['keypoints_'], pnp_op=pnp_op_0)
                                results.append(result_i)
                                Rt_i = np.eye(4)
                                Rt_i[:3, :3] = result_i['rot']
                                Rt_i[:3, 3:] = result_i['tvecs']
                                Rt_list.append(Rt_i)
                                
                        pred_results['Rts'] = np.array(Rt_list)
                        results_dict[obj_id] = pred_results
            
            t2 = time.time()
            
            if self.show_op:
                draw_image = draw_annotations_on_image_yolo(img_torch.clone().detach().cpu().numpy().astype(np.uint8)[0], 
                                                    det_results['xywh'].clone().detach().cpu().numpy().astype(np.int32), 
                                                    det_results['confs'].clone().detach().cpu().numpy().astype(np.float32),
                                                    det_results['cls'].clone().detach().cpu().numpy().astype(np.int32),
                                                    self.bop_dataset_item.obj_id_list,
                                                    confidence_threshold = confidence_threshold,
                                                    )
                draw_image = cv2.cvtColor(draw_image, cv2.COLOR_RGB2BGR)
                
            if self.show_op is not None:
                pred_front_code_l = []
                pred_back_code_l = []
                crop_rgbs_l = []
                pred_mask_l = []
                Detect_Bbox_tfs_128_l = []
                obj_ids_l = []
                Rts_l = []
                conf_s_l = []
                for obj_id in results_dict:
                    pred_results = results_dict[obj_id]
                    pred_front_code_raw = pred_results['pred_front_code_raw'].reshape((-1,128,128,3,8)).permute((0,1,2,4,3)).reshape((-1,128,128,24))
                    pred_back_code_raw = pred_results['pred_back_code_raw'].reshape((-1,128,128,3,8)).permute((0,1,2,4,3)).reshape((-1,128,128,24))
                    pred_front_code_l.append(torch.cat([pred_results['pred_front_code'], pred_front_code_raw], dim=-1))
                    pred_back_code_l.append(torch.cat([pred_results['pred_back_code'], pred_back_code_raw], dim=-1))
                    crop_rgbs_l.append(pred_results['crop_rgbs'])
                    pred_mask_l.append(pred_results['pred_mask'])
                    Detect_Bbox_tfs_128_l.append(pred_results['Detect_Bbox_tfs_128'])
                    obj_ids_l.append(np.ones((pred_results['pred_mask'].shape[0])) * obj_id)
                    Rts_l.append(pred_results['Rts'])
                    conf_s_l.append(pred_results['conf'])
                    
                crop_rgbs = torch.cat(crop_rgbs_l, dim = 0)
                pred_mask = torch.cat(pred_mask_l, dim = 0)
                pred_front_code = torch.cat(pred_front_code_l, dim = 0)
                pred_back_code = torch.cat(pred_back_code_l, dim = 0)
                Detect_Bbox_tfs_128 = torch.cat(Detect_Bbox_tfs_128_l, dim = 0)
                conf_s = torch.cat(conf_s_l, dim = 0)
                obj_ids_l = np.concatenate(obj_ids_l, axis = 0)
                Rts_l = np.concatenate(Rts_l, axis = 0)
                
                vis0 = vis_rgb_mask_Coord(crop_rgbs, pred_mask, pred_front_code, pred_back_code)
                
                pred_mask_origin = kornia.geometry.transform.warp_perspective(pred_mask[:,None,...], 
                                                                        torch.linalg.inv(Detect_Bbox_tfs_128.to(torch.float32)), 
                                                                        dsize=[img_torch[0].shape[0],img_torch[0].shape[1]], mode='nearest', align_corners=False)
                pred_front_code_origin = kornia.geometry.transform.warp_perspective(pred_front_code.permute(0,3,1,2), 
                                                                        torch.linalg.inv(Detect_Bbox_tfs_128.to(torch.float32)), 
                                                                        dsize=[img_torch[0].shape[0],img_torch[0].shape[1]], mode='nearest', align_corners=False)
                pred_back_code_origin = kornia.geometry.transform.warp_perspective(pred_back_code.permute(0,3,1,2), 
                                                                        torch.linalg.inv(Detect_Bbox_tfs_128.to(torch.float32)), 
                                                                        dsize=[img_torch[0].shape[0],img_torch[0].shape[1]], mode='nearest', align_corners=False)
                vis1, vis2 = vis_rgb_mask_Coord_origin(cam_K, obj_ids_l, self.bop_dataset_item.obj_id_list, self.BBox_3d, Rts_l, conf_s, 
                                          img_torch_hccepose, pred_mask_origin, 
                                          pred_front_code_origin, pred_back_code_origin)
            results_dict['time'] = t2 - t1
            if self.show_op:
                results_dict['show_2D_results'] = draw_image
                results_dict['show_6D_vis0'] = vis0
                results_dict['show_6D_vis1'] = vis1
                results_dict['show_6D_vis2'] = vis2

        return results_dict
