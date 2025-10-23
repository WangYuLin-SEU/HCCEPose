# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

'''
HCCEPose
|--- s2_p0_download_cc0textures.py
|--- cc0textures
'''

import os
from huggingface_hub import snapshot_download
script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(script_path)
snapshot_download(
    repo_id="SEU-WYL/HccePose",
    repo_type="dataset",
    allow_patterns=["cc0textures/**"],  
    local_dir=current_dir,  
    local_dir_use_symlinks=False   
)
