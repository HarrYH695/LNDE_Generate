import numpy as np
import torch
import os

# file_path = "/nfs/turbo/coe-mcity/hanhy/LNDE_Results/rD_Trial_4/1/"
# print(len(os.listdir(file_path)))

a = np.ones((2,3,4))
b = a[0,:,:]
print(b.shape)
c = np.sqrt(b)
print(c)