import os
from tqdm import tqdm
import pickle
import numpy as np

save_txt_dir = "/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/AA_check_txts/"
if not os.path.exists(save_txt_dir):
    os.makedirs(save_txt_dir)

file_save = "/nfs/turbo/coe-mcity/hanhy/LNDE_Results/AA_Trial_1/4_check_hascarinfo/"
processed_files = os.listdir(file_save)
print(len(processed_files)) #4427

file_case = "/nfs/turbo/coe-mcity/hanhy/LNDE_Results/AA_Trial_1/2/"
case_files = os.listdir(file_case)
print(len(case_files)) 

exist_wrong_num = 0
exist_wrong_num_gen = 0
exist_3_sigma_gen = 0
exist_2_sigma_gen = 0

all_3_sigma = 0
all_2_sigma = 0
all_1_sigma = 0

for file in tqdm(processed_files):
    data = pickle.load(open(file_save+file, "rb"))
    len_tb = data["tb_len"]
    wrong_sigma_info = data["wrong_and_sigma"] #(car_num, time, 4)
    #dis_info = data["distance_info"]

    wrong_time = np.argwhere(wrong_sigma_info[:, :, 0] > 0.5)
    # all_3_sigma += np.sum(wrong_sigma_info[:, 1])
    # all_2_sigma += np.sum(wrong_sigma_info[:, 2])
    # all_1_sigma += np.sum(wrong_sigma_info[:, 3])

    if len(wrong_time) > 0:
        exist_wrong_num += 1

        if np.max(wrong_time) > 4:
            exist_wrong_num_gen += 1
            flag1 = False
            flag2 = False
            for i in range(len(wrong_time)):
                if wrong_time[i] > 4:
                    if np.max(wrong_sigma_info[:wrong_time[i]+1, 1]) > 0.5:
                        flag1 = True

                    if np.max(wrong_sigma_info[:wrong_time[i]+1, 2]) > 0.5:
                        flag2 = True

            if flag1:
                exist_3_sigma_gen += 1

            if flag2:
                exist_2_sigma_gen += 1

print(exist_wrong_num)
print(exist_wrong_num_gen)
print(exist_3_sigma_gen)
print(exist_2_sigma_gen)
print(all_3_sigma)
print(all_2_sigma)
print(all_1_sigma)