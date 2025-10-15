import os, cv2, sys, json, copy, torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
import imgaug.augmenters as iaa
from torch.utils.data import Dataset
import platform
sys0 = platform.system()
if sys0 == "Linux":
    os.environ["PYOPENGL_PLATFORM"] = "egl"
sys.path.insert(0, os.getcwd())
current_directory = sys.argv[0]
pa_ = os.path.join(os.path.dirname(current_directory), 'bop_toolkit')
sys.path.append(pa_)
from bop_toolkit.bop_toolkit_lib import inout, renderer, misc, pose_error

def load_json2dict(path):
    with open(path, 'r') as f:
        dict_ = json.load(f)
    f.close()
    return dict_

def aug_square_fp32(GT_Bbox, padding_ratio):
    GT_Bbox = GT_Bbox.copy()
    center_x = GT_Bbox[0] + 0.5 * GT_Bbox[2]
    center_y = GT_Bbox[1] + 0.5 * GT_Bbox[3]
    width = GT_Bbox[2]
    height = GT_Bbox[3]
    scale_ratio = 1 + 0.25 * (2 * np.random.random_sample() - 1)
    shift_ratio = 0.25 * (2 * np.random.random_sample(2) - 1) 
    bbox_center = np.array([center_x + width * shift_ratio[0], center_y + height * shift_ratio[1]]) 
    augmented_width = width * scale_ratio * padding_ratio
    augmented_height = height * scale_ratio * padding_ratio
    w = max(augmented_width, augmented_height) 
    augmented_Box = np.array([bbox_center[0]-w/2, bbox_center[1]-w/2, w, w])
    return augmented_Box

def pad_square_fp32(GT_Bbox, padding_ratio):
    GT_Bbox = GT_Bbox.copy()
    center_x = GT_Bbox[0] + 0.5 * GT_Bbox[2]
    center_y = GT_Bbox[1] + 0.5 * GT_Bbox[3]
    width = GT_Bbox[2]
    height = GT_Bbox[3]
    scale_ratio = 1 + 0.0
    shift_ratio = 0.25 * 0.0
    bbox_center = np.array([center_x + width * shift_ratio, center_y + height * shift_ratio]) 
    augmented_width = width * scale_ratio * padding_ratio
    augmented_height = height * scale_ratio * padding_ratio
    w = max(augmented_width, augmented_height) 
    augmented_Box = np.array([bbox_center[0]-w/2, bbox_center[1]-w/2, w, w])
    return augmented_Box

def crop_square_resize(img, Bbox, crop_size=None, interpolation=None):
    Bbox = Bbox.copy()
    center_x = Bbox[0] + 0.5 * Bbox[2]
    center_y = Bbox[1] + 0.5 * Bbox[3]
    w_2 = Bbox[2] / 2
    pts1 = np.float32([[center_x - w_2, center_y - w_2], [center_x - w_2, center_y + w_2], [center_x + w_2, center_y - w_2]])
    pts2 = np.float32([[0, 0], [0, crop_size], [crop_size, 0]])
    M = cv2.getAffineTransform(pts1, pts2)
    roi_img = cv2.warpAffine(img, M, (crop_size, crop_size), flags=interpolation)
    return roi_img

class bop_dataset():
    
    def __init__(self, dataset_path, model_name = 'models', local_rank=0):
        self.local_rank = local_rank
        self.dataset_path = dataset_path
        if not os.path.exists(self.dataset_path):
            if local_rank == 0:
                print()
                print('dataset_path is not existed: ', self.dataset_path)
                print()
            return
        self.dataset_name = os.path.basename(dataset_path)
        if local_rank == 0:
            print()
            print('-*-' * 30)
            print('dataset name: ', self.dataset_name)
            print()
        self.model_path = os.path.join(dataset_path, model_name)
        if not os.path.exists(self.model_path):
            if local_rank == 0:
                print()
                print('model_name is not existed: ', self.model_path)
                print()
            return
        if local_rank == 0:
            print('obj model path: ', self.model_path)
        self.model_info = load_json2dict(os.path.join(self.model_path, 'models_info.json'))
        self.obj_id_list = []
        self.obj_model_list = []
        self.obj_info_list = []
        for key_i in self.model_info:
            self.obj_id_list.append(int(key_i)) 
            ply_path = os.path.join(self.model_path, 'obj_%s.ply'%str(int(key_i)).rjust(6, '0'))
            if not os.path.exists(ply_path):
                if local_rank == 0:
                    print()
                    print('%s is not existed'%ply_path)
                    print()
            self.obj_model_list.append(ply_path)
            self.obj_info_list.append(self.model_info[key_i])
        for i in range(len(self.obj_id_list)):
            if local_rank == 0:
                print('obj id: %s        obj model: %s'%(str(self.obj_id_list[i]).rjust(4, ' '), self.obj_model_list[i]))
        if local_rank == 0:
            print('-*-' * 30)
            print()
        pass
    
    def load_folder(self, folder_name, scene_num = 200, vis = 0.0):
        if self.local_rank == 0:
            print()
            print('-*-' * 30)
            print('folder name: ', folder_name)
            print()
        folder_path = os.path.join(self.dataset_path, folder_name)
        if not os.path.exists(folder_path):
            if self.local_rank == 0:
                print()
                print('folder_path is not existed: ', folder_path)
                print()
            return None
        
        scene_path_list = []
        for i in range(scene_num):
            scene_name = str(i).rjust(6, '0')
            scene_path = os.path.join(folder_path, scene_name)
            if os.path.exists(scene_path):
                scene_path_list.append(scene_path)
                if self.local_rank == 0:
                    print(scene_path)
        if self.local_rank == 0:
            print('-*-' * 30)
            print()
        
        img_info = {}
        
        obj_info = {}
        
        for scene_path_i in scene_path_list:
            if self.local_rank == 0:
                print('loading: ', scene_path_i)
            scene_camera_path = os.path.join(scene_path_i, 'scene_camera.json')
            scene_gt_info_path = os.path.join(scene_path_i, 'scene_gt_info.json')
            scene_gt_path = os.path.join(scene_path_i, 'scene_gt.json')
            
            scene_gt_info_dict = None
            if os.path.exists(scene_gt_info_path):
                scene_gt_info_dict = load_json2dict(scene_gt_info_path)
            else:
                if self.local_rank == 0:
                    print()
                    print('scene_gt_info_path is not existed: ', scene_gt_info_path)
                    print()
            
            scene_gt_dict = None
            if os.path.exists(scene_gt_path):
                scene_gt_dict = load_json2dict(scene_gt_path)
            else:
                if self.local_rank == 0:
                    print()
                    print('scene_gt_path is not existed: ', scene_gt_path)
                    print()
            if not os.path.exists(scene_camera_path):
                if self.local_rank == 0:
                    print()
                    print('scene_camera_path is not existed: ', scene_camera_path)
                    print()
                continue
                
            scene_camera_dict = load_json2dict(scene_camera_path)
            
            rgb_folder_name = 'rgb'
            dep_folder_name = 'depth'
            mask_folder_name = 'mask'
            mask_vis_folder_name = 'mask_visib'
            
            rgb_suffix = None
            dep_suffix = None
            mask_suffix = None
            mask_vis_suffix = None
            
            for camera_key in tqdm(scene_camera_dict):
                camera_i = scene_camera_dict[camera_key]
                
                scene_gt_info_i = None
                scene_gt_i = None
                if scene_gt_info_dict is not None:
                    scene_gt_info_i = scene_gt_info_dict[camera_key]
                if scene_gt_dict is not None:
                    scene_gt_i = scene_gt_dict[camera_key]
                camera_key_pad = camera_key.rjust(6, '0')
                if rgb_suffix is None:
                    for suffix_i in ['.jpg', '.jpeg', '.bmp', '.png', '.tif', '.tiff']:
                        if os.path.exists(os.path.join(scene_path_i, rgb_folder_name, camera_key_pad + suffix_i)):
                            rgb_suffix = suffix_i
                if dep_suffix is None:
                    for suffix_i in ['.jpg', '.jpeg', '.bmp', '.png', '.tif', '.tiff']:
                        if os.path.exists(os.path.join(scene_path_i, dep_folder_name, camera_key_pad + suffix_i)):
                            dep_suffix = suffix_i
                if mask_suffix is None:
                    for suffix_i in ['.jpg', '.jpeg', '.bmp', '.png', '.tif', '.tiff']:
                        if os.path.exists(os.path.join(scene_path_i, mask_folder_name, camera_key_pad + '_000000' + suffix_i)):
                            mask_suffix = suffix_i
                if mask_vis_suffix is None:
                    for suffix_i in ['.jpg', '.jpeg', '.bmp', '.png', '.tif', '.tiff']:
                        if os.path.exists(os.path.join(scene_path_i, mask_vis_folder_name, camera_key_pad + '_000000' + suffix_i)):
                            mask_vis_suffix = suffix_i
                rgb_suffix
                img_info_i = {}
                if rgb_suffix is not None:
                    img_info_i['rgb'] = os.path.join(scene_path_i, rgb_folder_name, camera_key_pad + rgb_suffix)
                    img_info_i['depth'] = os.path.join(scene_path_i, dep_folder_name, camera_key_pad + dep_suffix)
                    if scene_gt_info_i is not None:
                        for j in range(len(scene_gt_info_i)):
                            scene_gt_i_j_for_obj = {
                                'scene' : os.path.basename(scene_path_i),
                                'image' : camera_key_pad,
                                'rgb' : os.path.join(scene_path_i, rgb_folder_name, camera_key_pad + rgb_suffix),
                                'depth' : os.path.join(scene_path_i, dep_folder_name, camera_key_pad + dep_suffix),
                            }
                            scene_gt_info_i_j = scene_gt_info_i[j]
                            scene_gt_i_j = copy.deepcopy(scene_gt_i[j])
                            scene_gt_i_j.update(scene_gt_info_i_j)
                            scene_gt_i_j.update(camera_i)
                            
                            obj_id = scene_gt_i_j['obj_id']
                            if scene_gt_i_j['visib_fract'] >= vis:
                                scene_gt_i_j['mask_path'] = os.path.join(scene_path_i, mask_folder_name, camera_key_pad + '_' + str(j).rjust(6, '0') + mask_suffix)
                                scene_gt_i_j['mask_visib_path'] = os.path.join(scene_path_i, mask_vis_folder_name, camera_key_pad + '_' + str(j).rjust(6, '0') + mask_vis_suffix)
                                obj_id_key = 'obj_' + str(obj_id).rjust(6, '0')
                                if obj_id_key in img_info_i:
                                    img_info_i[obj_id_key].append(scene_gt_i_j)
                                else:
                                    img_info_i[obj_id_key] = [scene_gt_i_j]
                                scene_gt_i_j_for_obj.update(scene_gt_i_j)
                                scene_gt_i_j_for_obj.update(camera_i)
                                if obj_id_key in obj_info:
                                    obj_info[obj_id_key].append(scene_gt_i_j_for_obj)
                                else:
                                    obj_info[obj_id_key] = [scene_gt_i_j_for_obj]
                img_info['%s_%s'%(os.path.basename(scene_path_i), camera_key_pad)] = img_info_i
        if self.local_rank == 0:
            print('-*-' * 30)
            print()
        return {
            'img_info' : img_info,
            'obj_info' : obj_info,
            'scene_path_list' : scene_path_list,
        }
    
class rendering_bop_dataset_back_front(Dataset):
    
    def __init__(self, bop_dataset_item : bop_dataset, folder_name):
        self.bop_dataset_item = bop_dataset_item
        self.dataset_info = bop_dataset_item.load_folder(folder_name)
        self.folder_name = folder_name
        self.nSamples = 0
        if self.dataset_info is None:
            return

        target_dir_front = os.path.join(bop_dataset_item.dataset_path, folder_name + '_xyz_GT_front')
        try:
            os.mkdir(target_dir_front)
        except:
            1
        self.target_dir_front = target_dir_front
            
        target_dir_back = os.path.join(bop_dataset_item.dataset_path, folder_name + '_xyz_GT_back')
        try:
            os.mkdir(target_dir_back)
        except:
            1
        self.target_dir_back = target_dir_back
        
        for scene_path_i in self.dataset_info['scene_path_list']:
            try:
                os.mkdir(os.path.join(self.target_dir_front, os.path.basename(scene_path_i)))
            except:
                1
            try:
                os.mkdir(os.path.join(self.target_dir_back, os.path.basename(scene_path_i)))
            except:
                1
            
    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        info_ = self.dataset_info['obj_info']['obj_%s'%str(self.current_obj_id).rjust(6, '0')][index]
        scene_id = info_['scene']
        label_image_name = os.path.basename(info_['mask_path']).split('.')[0]
        front_label_image_path = os.path.join(self.target_dir_front, scene_id, label_image_name + '.png')
        back_label_image_path = os.path.join(self.target_dir_back, scene_id, label_image_name + '.png')
        R = np.array(info_['cam_R_m2c']).reshape((3, 3))
        t = np.array(info_['cam_t_m2c']).reshape((3, 1))
        RT = [R, t]
        cam_K = np.array(info_['cam_K']).reshape((3, 3))
        fx, fy, cx, cy = cam_K[0,0], cam_K[1,1], cam_K[0,2], cam_K[1,2]
        self.renderer_vispy.render_object(0, R, t, fx, fy, cx, cy, draw_back = False)
        depth = self.renderer_vispy.depth
        pose_4 = np.eye(4)
        pose_4[:3,:3] = R
        pose_4[:3,3] = t[:,0]
        model_info_obj = self.model_info_obj
        if 'symmetries_discrete' in model_info_obj:
            pose_4_re = self.pnp_solve_re_4p(pose_4, depth, fx, fy, cx, cy)
            RT[0], RT[1] = self.modified_sym_Rt(R, t, model_info_obj, e_rot = pose_4_re[:3,:3])
        mask_n = depth.copy()
        mask_n[mask_n>0] = 255
        mask_n = mask_n.astype(np.uint8)
        grid_row, grid_column = np.nonzero(mask_n.astype(np.int64)>0)
        p2dxy = np.empty((len(grid_row), 2))
        p2dxy[:, 1] = grid_row
        p2dxy[:, 0] = grid_column
        p2dxy[:, 0] -= cx
        p2dxy[:, 1] -= cy
        p2dxy[:, 0] /= fx
        p2dxy[:, 1] /= fy
        p2z = depth[mask_n>0]
        p2z = p2z.reshape((-1,1)).repeat(2, axis=1)
        T = RT[1]
        p2dxy *= p2z
        p2dxy[:, 0] -= T[0]
        p2dxy[:, 1] -= T[1]
        p2z -= T[2]
        p3xyz = np.empty((len(grid_row), 3))
        p3xyz[:, :2] = p2dxy
        p3xyz[:, 2] = p2z[:,0]
        R = RT[0]
        p3xyz = np.dot(p3xyz, R)
        p3xyz[:,0] = (p3xyz[:,0]-self.min_xyz[0]) / self.div_v[0]
        p3xyz[:,1] = (p3xyz[:,1]-self.min_xyz[1]) / self.div_v[1]
        p3xyz[:,2] = (p3xyz[:,2]-self.min_xyz[2]) / self.div_v[2]
        p3xyz[p3xyz>1] = 1
        p3xyz[p3xyz<0] = 0
        p3xyz = np.round(p3xyz * 255).astype(np.uint8)
        rgb_xyz = np.zeros((*depth.shape, 3), dtype=np.uint8)
        rgb_xyz[:,:, 0][mask_n>0] = p3xyz[:,0]
        rgb_xyz[:,:, 1][mask_n>0] = p3xyz[:,1]
        rgb_xyz[:,:, 2][mask_n>0] = p3xyz[:,2]
        cv2.imwrite(front_label_image_path, rgb_xyz)
        
        self.renderer_vispy.render_object(0, R, t, fx, fy, cx, cy, draw_back = True)
        depth = self.renderer_vispy.depth
        grid_row, grid_column = np.nonzero(mask_n.astype(np.int64)>0)
        p2dxy = np.empty((len(grid_row), 2))
        p2dxy[:, 1] = grid_row
        p2dxy[:, 0] = grid_column
        p2dxy[:, 0] -= cx
        p2dxy[:, 1] -= cy
        p2dxy[:, 0] /= fx
        p2dxy[:, 1] /= fy
        p2z = depth[mask_n>0]
        p2z = p2z.reshape((-1,1)).repeat(2, axis=1)
        T = RT[1]
        p2dxy *= p2z
        p2dxy[:, 0] -= T[0]
        p2dxy[:, 1] -= T[1]
        p2z -= T[2]
        p3xyz = np.empty((len(grid_row), 3))
        p3xyz[:, :2] = p2dxy
        p3xyz[:, 2] = p2z[:,0]
        R = RT[0]
        p3xyz = np.dot(p3xyz, R)
        p3xyz[:,0] = (p3xyz[:,0]-self.min_xyz[0]) / self.div_v[0]
        p3xyz[:,1] = (p3xyz[:,1]-self.min_xyz[1]) / self.div_v[1]
        p3xyz[:,2] = (p3xyz[:,2]-self.min_xyz[2]) / self.div_v[2]
        p3xyz[p3xyz>1] = 1
        p3xyz[p3xyz<0] = 0
        p3xyz = np.round(p3xyz * 255).astype(np.uint8)
        rgb_xyz = np.zeros((*depth.shape, 3), dtype=np.uint8)
        rgb_xyz[:,:, 0][mask_n>0] = p3xyz[:,0]
        rgb_xyz[:,:, 1][mask_n>0] = p3xyz[:,1]
        rgb_xyz[:,:, 2][mask_n>0] = p3xyz[:,2]
        cv2.imwrite(back_label_image_path, rgb_xyz)
        
        return 1

    def pnp_solve_re_4p(self, pose_, depth, fx, fy, cx, cy):
        
        '''
            The idea comes from : Yulin Wang, Hongli Li, and Chen Luo. Object Pose Estimation Based on Multi-precision Vectors and Seg-Driven PnP, International Journal of Computer Vision (IJCV), 2025, 133: 2620-2634.
        '''
        
        def perspective_unknown_kp_2D(kp, RT ):
            keypoints = kp
            Rot_m, T_m = RT
            keypoints_dot_mat = np.array(keypoints)
            mat = Rot_m
            for i in range(keypoints_dot_mat.shape[0]):
                keypoints_dot_mat[i]=np.dot(mat, keypoints_dot_mat[i])
            for i in range(keypoints_dot_mat.shape[0]):
                keypoints_dot_mat[i][2] = keypoints_dot_mat[i][2] + T_m[2]
                keypoints_dot_mat[i][0] = (keypoints_dot_mat[i][0] + T_m[0]) / keypoints_dot_mat[i][2] * fx + cx #- c_x
                keypoints_dot_mat[i][1] = (keypoints_dot_mat[i][1] + T_m[1]) / keypoints_dot_mat[i][2] * fy + cy #- c_y
            keypoints_xy = np.array(keypoints_dot_mat[:,:2])
            keypoints_xyz = np.array(keypoints_dot_mat[:,:3])
            return keypoints_xy, keypoints_xyz, keypoints

        def depth2kp(depth, RT):
            mask = depth.copy()
            mask[mask>0] = 255
            mask = mask.astype(np.uint8)
            grid_row, grid_column = np.nonzero(mask.astype(np.int64)>0)
            p2dxy = np.empty((len(grid_row), 2))
            p2dxy[:, 1] = grid_row
            p2dxy[:, 0] = grid_column
            p2dxy_r = p2dxy.copy()
            p2dxy[:, 0] -= cx
            p2dxy[:, 1] -= cy
            p2dxy[:, 0] /= fx
            p2dxy[:, 1] /= fy
            p2z = depth[mask>0]
            p2z = p2z.reshape((-1,1)).repeat(2, axis=1)
            T = RT[1]
            p2dxy *= p2z
            p2dxy[:, 0] -= T[0]
            p2dxy[:, 1] -= T[1]
            p2z -= T[2]
            p3xyz = np.empty((len(grid_row), 3))
            p3xyz[:, :2] = p2dxy
            p3xyz[:, 2] = p2z[:,0]
            R = RT[0]
            p3xyz = np.dot(p3xyz, R)
            return p2dxy_r, p3xyz

        def solvePnP(cam, image_points, object_points, return_inliers=False, ransac_iter=5000):
            dist_coeffs = np.zeros((5, 1)) 
            inliers = None
            if image_points.shape[0] < 4:
                pose = np.eye(4)
                inliers = []
            else:
                object_points = np.expand_dims(object_points, 1)
                image_points = np.expand_dims(image_points, 1)
                success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(object_points, image_points.astype(float), cam.astype(float),
                                                                                dist_coeffs, iterationsCount=ransac_iter,
                                                                                reprojectionError=1.5,
                                                                                confidence = 0.9995,)[:4]
                pose = np.eye(4)
                if success:
                    pose[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
                    pose[:3, 3] = np.squeeze(translation_vector)
            if return_inliers:
                return pose, len(inliers)
            else:
                return pose
        
        p2dxy_r, p3xyz = depth2kp(depth, [pose_[:3,:3], pose_[:3,3].reshape((3))])
        kp_ = np.array([[0,0,0]])
        kp_xy, _, _ = perspective_unknown_kp_2D(kp_, [pose_[:3, :3],pose_[:3, 3]], )
        T_0 = pose_[:3, 3].copy()
        T_0[:2] = 0
        kp_xy0, _, _ = perspective_unknown_kp_2D(kp_, [pose_[:3, :3],T_0], )
        dT_ = kp_xy0[0] - kp_xy[0]
        p2dxy_r[:, 0] += dT_[0]
        p2dxy_r[:, 1] += dT_[1]
        K = np.eye(3)
        K[0,0] = fx
        K[1,1] = fy
        K[0,2] = cx
        K[1,2] = cy
        pnp_pose = solvePnP(K, p2dxy_r, p3xyz, False)
        return pnp_pose

    def modified_sym_Rt(self, rot_pose, tra_pose, model_info, e_rot=None):
        '''
            Part of the idea comes from ZebraPose
        '''
        trans_disc = [{'R': np.eye(3), 't': np.array([[0, 0, 0]]).T}]  # Identity.
        for sym in model_info['symmetries_discrete']:
            sym_4x4 = np.reshape(sym, (4, 4))
            R = sym_4x4[:3, :3]
            t = sym_4x4[:3, 3].reshape((3, 1))
            trans_disc.append({'R': R, 't': t})
        best_R = None
        best_t = None
        froebenius_norm = 1e8
        for sym in trans_disc:
            R = sym['R']
            t = sym['t']
            if e_rot is None:
                tmp_froebenius_norm = np.linalg.norm(rot_pose.dot(R)-np.eye(3))
            else:
                tmp_froebenius_norm = np.linalg.norm(e_rot.dot(R)-np.eye(3))
            if tmp_froebenius_norm < froebenius_norm:
                froebenius_norm = tmp_froebenius_norm
                best_R = R
                best_t = t
        tra_pose = rot_pose.dot(best_t) + tra_pose
        rot_pose = rot_pose.dot(best_R)
        return rot_pose, tra_pose
    
    def update_obj_id(self, obj_id, obj_path):
        
        self.current_obj_id = obj_id
        self.current_obj_path = obj_path
        self.nSamples = len(self.dataset_info['obj_info']['obj_%s'%str(self.current_obj_id).rjust(6, '0')])
        self.model_info_obj = copy.deepcopy(self.bop_dataset_item.model_info[str(self.current_obj_id)])
        
        if 'symmetries_continuous' in self.model_info_obj:
            if len(self.model_info_obj['symmetries_continuous']):
                if "axis" in self.model_info_obj['symmetries_continuous'][0]:
                    self.model_info_obj['symmetries_discrete'] = misc.get_symmetry_transformations(self.model_info_obj, np.pi / 180)
                   
            self.model_info_obj.pop("symmetries_continuous")
        
        if 'symmetries_discrete' in self.model_info_obj:
            if len(self.model_info_obj['symmetries_discrete']) == 0:
                self.model_info_obj.pop("symmetries_discrete")
        return
    
    def worker_init_fn(self, worker_id):
        print(worker_id)
        self.worker_id = worker_id
        self.img_shape = cv2.imread(self.dataset_info['obj_info']['obj_%s'%str(self.current_obj_id).rjust(6, '0')][0]['rgb']).shape[:2]
        self.renderer_vispy = renderer.create_renderer(self.img_shape[1], self.img_shape[0], 'vispy', mode='depth', shading='flat')
        self.renderer_vispy.add_object(0, self.current_obj_path)
        vertices = inout.load_ply(self.current_obj_path)["pts"]
        div_v = np.max(vertices,axis=0) - np.min(vertices,axis=0)
        self.div_v = div_v
        self.min_xyz = np.min(vertices,axis=0)

class train_bop_dataset_back_front(Dataset):
    
    def __init__(self, bop_dataset_item : bop_dataset, folder_name, padding_ratio=1.5, crop_size_img=256, aug_op = 'imgaug', ):

        self.bop_dataset_item = bop_dataset_item
        self.dataset_info = bop_dataset_item.load_folder(folder_name, vis = 0.2)
        self.folder_name = folder_name
        self.nSamples = 0
        self.aug_op = aug_op
        self.padding_ratio = padding_ratio
        self.crop_size_img = crop_size_img
        self.crop_size_gt = int(crop_size_img / 2)
        if self.dataset_info is None:return
        target_dir_front = os.path.join(bop_dataset_item.dataset_path, folder_name + '_xyz_GT_front')
        try:os.mkdir(target_dir_front)
        except:1
        self.target_dir_front = target_dir_front
        target_dir_back = os.path.join(bop_dataset_item.dataset_path, folder_name + '_xyz_GT_back')
        try:os.mkdir(target_dir_back)
        except:1
        self.target_dir_back = target_dir_back
        for scene_path_i in self.dataset_info['scene_path_list']:
            try:os.mkdir(os.path.join(self.target_dir_front, os.path.basename(scene_path_i)))
            except:1
            try:os.mkdir(os.path.join(self.target_dir_back, os.path.basename(scene_path_i)))
            except:1
        self.composed_transforms_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        pass
    
    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        info_ = self.dataset_info['obj_info']['obj_%s'%str(self.current_obj_id).rjust(6, '0')][index]
        rgb = cv2.imread(info_['rgb'])
        mask_vis = cv2.imread(info_['mask_visib_path'], 0)
        label_image_name = os.path.basename(info_['mask_path']).split('.')[0]
        front_label_image_path = os.path.join(self.target_dir_front, info_['scene'], label_image_name + '.png')
        back_label_image_path = os.path.join(self.target_dir_back, info_['scene'], label_image_name + '.png')
        GT_Front = cv2.imread(front_label_image_path)
        GT_Back = cv2.imread(back_label_image_path)
        if GT_Front is None: GT_Front = np.zeros((self.crop_size_gt, self.crop_size_gt, 3))
        if GT_Back is None: GT_Back = np.zeros((self.crop_size_gt, self.crop_size_gt, 3))
        if self.aug_op == 'imgaug': rgb = self.apply_augmentation(rgb)
        Bbox = aug_square_fp32(np.array(info_['bbox_visib']), padding_ratio=self.padding_ratio)
        rgb_c = crop_square_resize(rgb, Bbox, self.crop_size_img, interpolation=cv2.INTER_LINEAR)
        mask_vis_c = crop_square_resize(mask_vis, Bbox, self.crop_size_gt, interpolation=cv2.INTER_NEAREST)
        GT_Front_c = crop_square_resize(GT_Front, Bbox, self.crop_size_gt, interpolation=cv2.INTER_NEAREST)
        GT_Front_hcce = self.hcce_encode(GT_Front_c)
        GT_Back_c = crop_square_resize(GT_Back, Bbox, self.crop_size_gt, interpolation=cv2.INTER_NEAREST)
        GT_Back_hcce = self.hcce_encode(GT_Back_c)
        rgb_c, mask_vis_c, GT_Front_hcce, GT_Back_hcce = self.preprocess(rgb_c, mask_vis_c, GT_Front_hcce, GT_Back_hcce)
        return rgb_c, mask_vis_c, GT_Front_hcce, GT_Back_hcce

    def hcce_encode(self, code_img, iteration=8):
        code_img = code_img.copy()
        
        code_img = [code_img[:, :, 0], code_img[:, :, 1], code_img[:, :, 2]]
        hcce_images = np.zeros((code_img[0].shape[0], code_img[0].shape[1], iteration * 3))
        for i in range(iteration):
            temp1 = np.array(code_img[0] % (2**(iteration-i)), dtype='int') / (2**(iteration-i)-1)
            hcce_images[:,:,i] = temp1
            temp1 = np.array(code_img[1] % (2**(iteration-i)), dtype='int') / (2**(iteration-i)-1)
            hcce_images[:,:,i+iteration] = temp1
            temp1 = np.array(code_img[2] % (2**(iteration-i)), dtype='int') / (2**(iteration-i)-1)
            hcce_images[:,:,i+iteration*2] = temp1
        check_hcce_images = hcce_images.copy()
        k_ = iteration
        for i in range(k_-1):
            temp = hcce_images[:,:,i+1].copy()
            temp[hcce_images[:,:,i] >= 0.5] = -temp[hcce_images[:,:,i] >= 0.5] + 1
            check_hcce_images[:,:,i+1]=temp
        for i in range(k_-1):
            temp = hcce_images[:,:,i+1+k_].copy()
            temp[hcce_images[:,:,i+k_] >= 0.5] = -temp[hcce_images[:,:,i+k_] >= 0.5] + 1
            check_hcce_images[:,:,i+k_+1]=temp
        for i in range(k_-1):
            temp = hcce_images[:,:,i+1+k_*2].copy()
            temp[hcce_images[:,:,i+k_*2] >= 0.5] = -temp[hcce_images[:,:,i+k_*2] >= 0.5] + 1
            check_hcce_images[:,:,i+k_*2+1]=temp
        
        
        return check_hcce_images

    def update_obj_id(self, obj_id, obj_path):
        
        self.current_obj_id = obj_id
        self.current_obj_path = obj_path
        self.nSamples = len(self.dataset_info['obj_info']['obj_%s'%str(self.current_obj_id).rjust(6, '0')])
        self.model_info_obj = copy.deepcopy(self.bop_dataset_item.model_info[str(self.current_obj_id)])
        
        if 'symmetries_continuous' in self.model_info_obj:
            if len(self.model_info_obj['symmetries_continuous']):
                if "axis" in self.model_info_obj['symmetries_continuous'][0]:
                    self.model_info_obj['symmetries_discrete'] = misc.get_symmetry_transformations(self.model_info_obj, np.pi / 180)
                   
            self.model_info_obj.pop("symmetries_continuous")
        
        if 'symmetries_discrete' in self.model_info_obj:
            if len(self.model_info_obj['symmetries_discrete']) == 0:
                self.model_info_obj.pop("symmetries_discrete")
        return

    def apply_augmentation(self, x):
        def build_augmentations_depth():
            augmentations = []
            augmentations.append(iaa.Sometimes(0.3, iaa.SaltAndPepper(0.05)))
            augmentations.append(iaa.Sometimes(0.2, iaa.MotionBlur(k=5)))
            augmentations = augmentations + [iaa.Sometimes(0.4, iaa.CoarseDropout( p=0.1, size_percent=0.05) ),
                                            iaa.Sometimes(0.5, iaa.GaussianBlur(np.random.rand())),
                                            iaa.Sometimes(0.5, iaa.Add((-20, 20), per_channel=0.3)),
                                            iaa.Sometimes(0.4, iaa.Invert(0.20, per_channel=True)),
                                            iaa.Sometimes(0.5, iaa.Multiply((0.7, 1.4), per_channel=0.8)),
                                            iaa.Sometimes(0.5, iaa.Multiply((0.7, 1.4))),
                                            iaa.Sometimes(0.5, iaa.ContrastNormalization((0.5, 2.0), per_channel=0.3))
                                            ]
            image_augmentations=iaa.Sequential(augmentations, random_order = False)
            return image_augmentations
        self.augmentations = build_augmentations_depth()
        color_aug_prob = 0.8
        if np.random.rand() < color_aug_prob:
            x = self.augmentations.augment_image(x)
        return x
    
    def preprocess(self, rgb_c, mask_vis_c, GT_Front_hcce, GT_Back_hcce):
        rgb_c_pil = Image.fromarray(np.uint8(rgb_c)).convert('RGB')
        mask_vis_c = mask_vis_c / 255.
        mask_vis_c = torch.from_numpy(mask_vis_c).type(torch.float)
        GT_Front_hcce = torch.from_numpy(GT_Front_hcce).permute(2, 0, 1)
        GT_Back_hcce = torch.from_numpy(GT_Back_hcce).permute(2, 0, 1)
        return self.composed_transforms_img(rgb_c_pil), mask_vis_c, GT_Front_hcce, GT_Back_hcce

class test_bop_dataset_back_front(Dataset):
    
    def __init__(self, bop_dataset_item : bop_dataset, folder_name, bbox_2D = None, padding_ratio=1.5, crop_size_img=256, ratio = 1.0 ):
        
        self.ratio = ratio
        self.bbox_2D = bbox_2D
        self.bop_dataset_item = bop_dataset_item
        self.dataset_info = bop_dataset_item.load_folder(folder_name, vis = 0.2)
        self.folder_name = folder_name
        self.nSamples = 0
        self.padding_ratio = padding_ratio
        self.crop_size_img = crop_size_img
        self.crop_size_gt = int(crop_size_img / 2)
        if self.dataset_info is None:return
        target_dir_front = os.path.join(bop_dataset_item.dataset_path, folder_name + '_xyz_GT_front')
        try:os.mkdir(target_dir_front)
        except:1
        self.target_dir_front = target_dir_front
        target_dir_back = os.path.join(bop_dataset_item.dataset_path, folder_name + '_xyz_GT_back')
        try:os.mkdir(target_dir_back)
        except:1
        self.target_dir_back = target_dir_back
        for scene_path_i in self.dataset_info['scene_path_list']:
            try:os.mkdir(os.path.join(self.target_dir_front, os.path.basename(scene_path_i)))
            except:1
            try:os.mkdir(os.path.join(self.target_dir_back, os.path.basename(scene_path_i)))
            except:1
        self.composed_transforms_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        pass
    
    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        info_ = self.dataset_info['obj_info']['obj_%s'%str(self.current_obj_id).rjust(6, '0')][index]
        
        cam_K = np.array(info_['cam_K']).reshape((3,3))
        if 'cam_R_m2c' in info_ and 'cam_t_m2c' in info_:
            cam_R_m2c = np.array(info_['cam_R_m2c']).reshape((3,3))
            cam_t_m2c = np.array(info_['cam_t_m2c']).reshape((3,1))
        else:
            cam_R_m2c = np.eye(3)
            cam_t_m2c = np.zeros((3, 1))
        rgb = cv2.imread(info_['rgb'])
        mask_vis = cv2.imread(info_['mask_visib_path'], 0)
        label_image_name = os.path.basename(info_['mask_path']).split('.')[0]
        front_label_image_path = os.path.join(self.target_dir_front, info_['scene'], label_image_name + '.png')
        back_label_image_path = os.path.join(self.target_dir_back, info_['scene'], label_image_name + '.png')
        GT_Front = cv2.imread(front_label_image_path)
        GT_Back = cv2.imread(back_label_image_path)
        if GT_Front is None: GT_Front = np.zeros((self.crop_size_gt, self.crop_size_gt, 3))
        if GT_Back is None: GT_Back = np.zeros((self.crop_size_gt, self.crop_size_gt, 3))
        if self.bbox_2D is not None:Bbox = pad_square_fp32(np.array(self.bbox_2D[index]), padding_ratio=self.padding_ratio)
        else:Bbox = pad_square_fp32(np.array(info_['bbox_visib']), padding_ratio=self.padding_ratio)
        mask_vis_b = cv2.boundingRect(mask_vis)
        rgb_c = crop_square_resize(rgb, Bbox, self.crop_size_img, interpolation=cv2.INTER_LINEAR)
        mask_vis_c = crop_square_resize(mask_vis, Bbox, self.crop_size_gt, interpolation=cv2.INTER_NEAREST)
        GT_Front_c = crop_square_resize(GT_Front, Bbox, self.crop_size_gt, interpolation=cv2.INTER_NEAREST)
        GT_Front_hcce = self.hcce_encode(GT_Front_c)
        GT_Back_c = crop_square_resize(GT_Back, Bbox, self.crop_size_gt, interpolation=cv2.INTER_NEAREST)
        GT_Back_hcce = self.hcce_encode(GT_Back_c)
        rgb_c, mask_vis_c, GT_Front_hcce, GT_Back_hcce = self.preprocess(rgb_c, mask_vis_c, GT_Front_hcce, GT_Back_hcce)
        return rgb_c, mask_vis_c, GT_Front_hcce, GT_Back_hcce, Bbox, cam_K, cam_R_m2c, cam_t_m2c

    def hcce_encode(self, code_img, iteration=8):
        code_img = [code_img[:, :, 0], code_img[:, :, 1], code_img[:, :, 2]]
        hcce_images = np.zeros((code_img[0].shape[0], code_img[0].shape[1], iteration * 3))
        for i in range(iteration):
            temp1 = np.array(code_img[0] % (2**(iteration-i)), dtype='int') / (2**(iteration-i)-1)
            hcce_images[:,:,i] = temp1
            temp1 = np.array(code_img[1] % (2**(iteration-i)), dtype='int') / (2**(iteration-i)-1)
            hcce_images[:,:,i+iteration] = temp1
            temp1 = np.array(code_img[2] % (2**(iteration-i)), dtype='int') / (2**(iteration-i)-1)
            hcce_images[:,:,i+iteration*2] = temp1
        check_hcce_images = hcce_images.copy()
        k_ = iteration
        for i in range(k_-1):
            temp = hcce_images[:,:,i+1].copy()
            temp[hcce_images[:,:,i] >= 0.5] = -temp[hcce_images[:,:,i] >= 0.5] + 1
            check_hcce_images[:,:,i+1]=temp
        for i in range(k_-1):
            temp = hcce_images[:,:,i+1+k_].copy()
            temp[hcce_images[:,:,i+k_] >= 0.5] = -temp[hcce_images[:,:,i+k_] >= 0.5] + 1
            check_hcce_images[:,:,i+k_+1]=temp
        for i in range(k_-1):
            temp = hcce_images[:,:,i+1+k_*2].copy()
            temp[hcce_images[:,:,i+k_*2] >= 0.5] = -temp[hcce_images[:,:,i+k_*2] >= 0.5] + 1
            check_hcce_images[:,:,i+k_*2+1]=temp
        return check_hcce_images

    def update_obj_id(self, obj_id, obj_path):
        
        
        self.current_obj_id = obj_id
        self.current_obj_path = obj_path
        
        self.nSamples = len(self.dataset_info['obj_info']['obj_%s'%str(self.current_obj_id).rjust(6, '0')])
        
        if self.ratio != 1.0:
            len_ = int(self.ratio * len(self.dataset_info['obj_info']['obj_%s'%str(self.current_obj_id).rjust(6, '0')])) + 0
            
            self.nSamples = len_
        
        self.model_info_obj = copy.deepcopy(self.bop_dataset_item.model_info[str(self.current_obj_id)])
        
        if 'symmetries_continuous' in self.model_info_obj:
            if len(self.model_info_obj['symmetries_continuous']):
                if "axis" in self.model_info_obj['symmetries_continuous'][0]:
                    self.model_info_obj['symmetries_discrete'] = misc.get_symmetry_transformations(self.model_info_obj, np.pi / 180)
                   
            self.model_info_obj.pop("symmetries_continuous")
        
        if 'symmetries_discrete' in self.model_info_obj:
            if len(self.model_info_obj['symmetries_discrete']) == 0:
                self.model_info_obj.pop("symmetries_discrete")
        return

    def preprocess(self, rgb_c, mask_vis_c, GT_Front_hcce, GT_Back_hcce):
        rgb_c_pil = Image.fromarray(np.uint8(rgb_c)).convert('RGB')
        mask_vis_c = mask_vis_c / 255.
        mask_vis_c = torch.from_numpy(mask_vis_c).type(torch.float)
        GT_Front_hcce = torch.from_numpy(GT_Front_hcce).permute(2, 0, 1)
        GT_Back_hcce = torch.from_numpy(GT_Back_hcce).permute(2, 0, 1)
        return self.composed_transforms_img(rgb_c_pil), mask_vis_c, GT_Front_hcce, GT_Back_hcce


