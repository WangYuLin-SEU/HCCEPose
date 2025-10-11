
import torch
from HccePose.bop_loader import bop_dataset, rendering_bop_dataset_back_front


if __name__ == '__main__':
    
    # dataset_path = '/root/xxxxxx/demo-bin-picking'
    dataset_path = '/root/xxxxxx/demo-tex-objs'
    bop_dataset_item = bop_dataset(dataset_path)
    folder_name = 'train_pbr'
    
    rendering_bop_dataset_back_front_item = rendering_bop_dataset_back_front(bop_dataset_item, folder_name)
    
    for (obj_id, obj_path) in zip(bop_dataset_item.obj_id_list, bop_dataset_item.obj_model_list):
        print(obj_path)
        
        rendering_bop_dataset_back_front_item.update_obj_id(obj_id, obj_path)
        
        batch_size = 16
        
        data_gen_loader = torch.utils.data.DataLoader(rendering_bop_dataset_back_front_item, 
                                                batch_size=batch_size, shuffle=False, num_workers=16, drop_last=False, 
                                                worker_init_fn=rendering_bop_dataset_back_front_item.worker_init_fn)
        for batch_idx, (cc_) in enumerate(data_gen_loader):
            if int(batch_idx%5) == 0:
                print(batch_idx)
            if batch_idx == int(rendering_bop_dataset_back_front_item.nSamples / batch_size) + 1:
                break
    pass