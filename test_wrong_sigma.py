import os
from tqdm import tqdm
import pickle
import numpy as np

save_txt_dir = "/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/AA_check_txts/"
if not os.path.exists(save_txt_dir):
    os.makedirs(save_txt_dir)

file_save = "/nfs/turbo/coe-mcity/hanhy/LNDE_Results/AA_Trial_1/5_check_remove_small_dis_when_angle/" 
file_save = "/nfs/turbo/coe-mcity/hanhy/LNDE_Results/AA_Trial_1/4_check_hascarinfo/"
processed_files = os.listdir(file_save)
print(len(processed_files)) #4427

file_case = "/nfs/turbo/coe-mcity/hanhy/LNDE_Results/AA_Trial_1/2/"
case_files = os.listdir(file_case)
print(len(case_files)) 

exist_wrong_num = 0
exist_wrong_num_gen = 0
exist_wrong_num_gen_dis = 0
exist_wrong_num_gen_ang = 0
exist_wrong_num_gen_poc = 0
exist_3_sigma_gen_dis = 0
exist_2_sigma_gen_dis = 0
exist_3_sigma_gen_ang = 0
exist_2_sigma_gen_ang = 0
exist_3_sigma_gen_poc = 0
exist_2_sigma_gen_poc = 0
exist_3_sigma_gen_allcase = 0
exist_2_sigma_gen_allcase = 0

for file in tqdm(processed_files):
    data = pickle.load(open(file_save+file, "rb"))
    len_tb = data["tb_len"]
    wrong_sigma_info = data["wrong_and_sigma"] #(car_num, time, 4)
    #dis_info = data["distance_info"]

    wrong_time_1 = np.argwhere(wrong_sigma_info[:, :, 0] > 0.5)
    wrong_time_2 = np.argwhere(wrong_sigma_info[:, :, 1] > 0.5)
    wrong_time_3 = np.argwhere(wrong_sigma_info[:, :, 2] > 0.5)

    if len(wrong_time_1) > 0 or len(wrong_time_2) > 0 or len(wrong_time_3) > 0:
        exist_wrong_num += 1
        k3 = 0
        k2 = 0
        ex_flag = 0

        if len(wrong_time_1) > 0 and np.max(wrong_time_1[:, 1]) > 4:
            exist_wrong_num_gen_dis += 1
            ex_flag = 1
            flag1 = False
            flag2 = False
            for i in range(wrong_time_1.shape[0]):
                #如果某辆车在某时刻是wrong的，检测这辆车在之前有没有3sigma之外
                if wrong_time_1[i,1] > 4:
                    if np.max(wrong_sigma_info[wrong_time_1[i,0],:wrong_time_1[i,1]+1, 3]) > 0.5:
                        flag1 = True

                    if np.max(wrong_sigma_info[wrong_time_1[i,0],:wrong_time_1[i,1]+1, 4]) > 0.5:
                        flag2 = True

            if flag1:
                exist_3_sigma_gen_dis += 1
                k3 = 1

            if flag2:
                exist_2_sigma_gen_dis += 1
                k2 = 1


        if len(wrong_time_2) > 0 and np.max(wrong_time_2[:, 1]) > 4:
            exist_wrong_num_gen_ang += 1
            ex_flag = 1
            flag1 = False
            flag2 = False
            for i in range(wrong_time_2.shape[0]):
                #如果某辆车在某时刻是wrong的，检测这辆车在之前有没有3sigma之外
                if wrong_time_2[i,1] > 4:
                    if np.max(wrong_sigma_info[wrong_time_2[i,0],:wrong_time_2[i,1]+1, 3]) > 0.5:
                        flag1 = True

                    if np.max(wrong_sigma_info[wrong_time_2[i,0],:wrong_time_2[i,1]+1, 4]) > 0.5:
                        flag2 = True

            if flag1:
                exist_3_sigma_gen_ang += 1
                k3 = 1

            if flag2:
                exist_2_sigma_gen_ang += 1
                k2 = 1

        if len(wrong_time_3) > 0:
            exist_wrong_num_gen_poc += 1
            ex_flag = 1
            flag1 = False
            flag2 = False
            #如果此前有车在3sigma或2sigma之外
            if np.max(wrong_sigma_info[:, -1, 3]) > 0.5:
                flag1 = True

            if np.max(wrong_sigma_info[:, -1, 4]) > 0.5:
                flag2 = True

            if flag1:
                exist_3_sigma_gen_poc += 1
                k3 = 1

            if flag2:
                exist_2_sigma_gen_poc += 1
                k2 = 1

        if k3 == 1:
            exist_3_sigma_gen_allcase += 1

        if k2 == 1:
            exist_2_sigma_gen_allcase += 1

        if ex_flag == 1:
            exist_wrong_num_gen += 1

print(f"exist_wrong_num: {exist_wrong_num}")
print(f"exist_wrong_num_gen:{exist_wrong_num_gen}")
print(f"exist_wrong_num_gen_dis:{exist_wrong_num_gen_dis}")
print(f"exist_wrong_num_gen_ang:{exist_wrong_num_gen_ang}")
print(f"exist_wrong_num_gen_poc:{exist_wrong_num_gen_poc}")
print(f"exist_3_sigma_gen_dis:{exist_3_sigma_gen_dis}")
print(f"exist_2_sigma_gen_dis:{exist_2_sigma_gen_dis}")
print(f"exist_3_sigma_gen_ang:{exist_3_sigma_gen_ang}")
print(f"exist_2_sigma_gen_ang:{exist_2_sigma_gen_ang}")
print(f"exist_3_sigma_gen_poc:{exist_3_sigma_gen_poc}")
print(f"exist_2_sigma_gen_poc:{exist_2_sigma_gen_poc}")
print(f"exist_3_sigma_gen_allcase:{exist_3_sigma_gen_allcase}")
print(f"exist_2_sigma_gen_allcase:{exist_2_sigma_gen_allcase}")





# print(exist_wrong_num)
# print(exist_wrong_num_gen)
# print(exist_3_sigma_gen)
# print(exist_2_sigma_gen)
# print(all_3_sigma)
# print(all_2_sigma)
# print(all_1_sigma)


