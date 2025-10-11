import os

if __name__ == '__main__':

    dataset_name = 'demo-bin-picking'
    gpu_num = 6
    task_suffix = 'detection'
    dataset_path = '/root/xxxxxx/%s/train_pbr'%dataset_name
    train_multi_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_multi.py')
    data_objs_path = os.path.join(os.path.dirname(dataset_path), 'yolo11', 'train_obj_s', 'yolo_configs', 'data_objs.yaml')
    save_dir = os.path.join(os.path.dirname(os.path.dirname(data_objs_path)), task_suffix, f"obj_s")
    model_name = f"yolo11-{task_suffix}-obj_s.pt"
    final_model_path = os.path.join(os.path.dirname(os.path.dirname(data_objs_path)), save_dir, model_name)
    obj_s_path = os.path.dirname(final_model_path)
    while 1:
        if not os.path.exists('%s'%obj_s_path):
            os.system("python %s --data_path '%s' --epochs 100 --imgsz 640 --batch 60 --gpu_num %s --task %s"%(train_multi_path, data_objs_path, str(gpu_num), task_suffix))
        if os.path.exists('%s'%obj_s_path):
            break
    pass
