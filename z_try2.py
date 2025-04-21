import os
import numpy as np
import pickle

# file_vis = "/nfs/turbo/coe-mcity/hanhy/LNDE_Results/AA_Trial_1/2_vis/"
# files = os.listdir(file_vis)
# print(len(files))

file_save = "/nfs/turbo/coe-mcity/hanhy/LNDE_Results/AA_Trial_1/4_check_hascarinfo/"
processed_files = os.listdir(file_save)

for file in processed_files:
    data = pickle.load(open(file_save+file, "rb"))
    len_tb = data["tb_len"]
    wrong_sigma_info = data["wrong_and_sigma"] #(car_num, time, 4)

    wrong_time = np.argwhere(wrong_sigma_info[:, :, 0] > 0.5)
    
    if len(wrong_time) > 1:
        print(wrong_time.shape)
        print(wrong_sigma_info[:, :, 0])
        print(wrong_time[:,1])
        break