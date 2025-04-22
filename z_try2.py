import os
import numpy as np
import pickle

# file_vis = "/nfs/turbo/coe-mcity/hanhy/LNDE_Results/AA_Trial_1/2_vis/"
# files = os.listdir(file_vis)
# print(len(files))

file_save = "/nfs/turbo/coe-mcity/hanhy/LNDE_Results/AA_Trial_1/4_check_hascarinfo/"
processed_files = os.listdir(file_save)

for file in processed_files:
    file = "27.pkl"
    data = pickle.load(open(file_save+file, "rb"))
    len_tb = data["tb_len"]
    distance_info = data["distance_info"] #(car_num, time, 4)

    car_dis_simu = distance_info[:, :, :2]

    car_num = 6
    disall = []
    for i in range(6,11):
        dis = np.linalg.norm(car_dis_simu[car_num, i + 1, :] - car_dis_simu[car_num, i, :])
        disall.append(dis)

    print(disall)
    break