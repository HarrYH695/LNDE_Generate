import numpy as np
import torch
import os

# dir = "rD_Trial_2_vali_only"

# file_path = "/nfs/turbo/coe-mcity/hanhy/LNDE_Results/" + dir + "/1/"
# print(len(os.listdir(file_path)))

# # file_path = "/nfs/turbo/coe-mcity/hanhy/LNDE_Results/" + dir + "/scene_videos/"
# # print(len(os.listdir(file_path)))

# L = torch.zeros((4,5,3,2,2))
# a = torch.ones((4,5,3,2))
# a[:,:,:,1] *= 3
# b = torch.ones((4,5,3))*0.5

# std_x = a[:,:,:,0]
# std_y = a[:,:,:,1]
# print(std_x.shape, std_y.shape)

# # L[:,:,:,0,0] = std_x
# # L[:,:,:,1,0] = std_y
# # L[:,:,:,1,1] = std_y*b
# # print(L)

# row1 = torch.stack([std_x, torch.zeros_like(std_x)], dim=-1)
# print(row1.shape)
# # print(row1)
# row2 = torch.stack([std_y, std_y*b], dim=-1)
# print(row2.shape)

# L2 = torch.stack([row1, row2], dim=-2)
# print(L2.shape)
# print(L2)

a = 5.0e-5
print(a==0.00005)