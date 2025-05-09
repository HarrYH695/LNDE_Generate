import numpy as np
import torch
import os

# dir = "rD_Trial_2_vali_only"

# file_path = "/nfs/turbo/coe-mcity/hanhy/LNDE_Results/" + dir + "/1/"
# print(len(os.listdir(file_path)))

# # file_path = "/nfs/turbo/coe-mcity/hanhy/LNDE_Results/" + dir + "/scene_videos/"
# # print(len(os.listdir(file_path)))

a = torch.ones((3,5,2))
b = a[:,:,0].unsqueeze(-1)
c = 2 * a[:,:,1].unsqueeze(-1)

d = torch.cat([b, c], dim=2)
print(d.shape)
print(d)
