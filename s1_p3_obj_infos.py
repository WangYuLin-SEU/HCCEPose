import os
import numpy as np
from kasal.utils import load_ply_model, load_json2dict, get_all_ply_obj, write_dict2json

if __name__ == '__main__':
    
    dataset_path = 'demo-tex-objs'


    models_path = get_all_ply_obj(dataset_path)

    models_info = {}

    for model_path in models_path:
        
        ply_info = load_ply_model(model_path)
        
        model_info = {
            "diameter" : float(ply_info['diameter']),
            "max_x" : float(np.max(ply_info['vertices'], axis = 0)[0]),
            "max_y" : float(np.max(ply_info['vertices'], axis = 0)[1]),
            "max_z" : float(np.max(ply_info['vertices'], axis = 0)[2]),
            "min_x" : float(np.min(ply_info['vertices'], axis = 0)[0]),
            "min_y" : float(np.min(ply_info['vertices'], axis = 0)[1]),
            "min_z" : float(np.min(ply_info['vertices'], axis = 0)[2]),
            "size_x" : float(np.max(ply_info['vertices'], axis = 0)[0] - np.min(ply_info['vertices'], axis = 0)[0]),
            "size_y" : float(np.max(ply_info['vertices'], axis = 0)[1] - np.min(ply_info['vertices'], axis = 0)[1]),
            "size_z" : float(np.max(ply_info['vertices'], axis = 0)[2] - np.min(ply_info['vertices'], axis = 0)[2]),
        }
        
        symmetry_type_dict = None
        sym_type_file = os.path.join(os.path.dirname(model_path), os.path.basename(model_path).split('.')[0]+'_sym_type.json')
        if os.path.exists(sym_type_file):
            symmetry_type_dict = load_json2dict(sym_type_file)
        if symmetry_type_dict is not None:
            if 'symmetries_continuous' in symmetry_type_dict['current_obj_info']:
                model_info["symmetries_continuous"] = symmetry_type_dict['current_obj_info']["symmetries_continuous"]
            if 'symmetries_discrete' in symmetry_type_dict['current_obj_info']:
                model_info["symmetries_discrete"] = symmetry_type_dict['current_obj_info']["symmetries_discrete"]
            
        model_id = str(int(os.path.basename(model_path).split('.')[0].split('obj_')[1]))
        
        models_info[model_id] = model_info

    if len(models_path) > 0:
        models_info_path = os.path.join(os.path.dirname(model_path), 'models_info.json')
        write_dict2json(models_info_path, models_info)
    
    pass