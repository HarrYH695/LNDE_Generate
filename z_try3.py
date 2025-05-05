import numpy as np
import torch
import os

dir = "rD_Trial_2c"

file_path = "/nfs/turbo/coe-mcity/hanhy/LNDE_Results/" + dir + "/1/"
print(len(os.listdir(file_path)))

file_path = "/nfs/turbo/coe-mcity/hanhy/LNDE_Results/" + dir + "/scene_videos/"
print(len(os.listdir(file_path)))