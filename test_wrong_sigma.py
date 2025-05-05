import os
from tqdm import tqdm
import pickle
import numpy as np

dir_name = "rD_baseline_3"
dir_name = "rD_Trial_2r_woD"

file_save = "/nfs/turbo/coe-mcity/hanhy/LNDE_Results/" + dir_name + "/check/"
processed_files = os.listdir(file_save)

file_case = "/nfs/turbo/coe-mcity/hanhy/LNDE_Results/" + dir_name + "/2/"
# case_files = os.listdir(file_case)
# print(len(case_files)) 

dir_txt_save = "/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/rD_check_txts/"

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
            with open(dir_txt_save + dir_name + "_no_poc.txt", "a+") as ft:
                ft.write(file)
                ft.write("\n")

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
            with open(dir_txt_save + dir_name + "_no_poc.txt", "a+") as ft:
                ft.write(file)
                ft.write("\n")

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
            # with open(dir_txt_save + dir_name + "_no_poc.txt", "a+") as ft:
            #     ft.write(file)
            #     ft.write("\n")

print(dir_name)
print(f"All data:{len(processed_files)}")
print("----------------------------------")
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


# python test_wrong_sigma.py