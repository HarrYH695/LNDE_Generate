import numpy as np
import torch
import pickle
from tqdm import tqdm
import time
import os
import copy

# TIME_BUFF: [idx - history_length, idx)时间段内的车辆位置信息,TIME_BUFF[i]代表i时刻的所有车辆的相关信息，具体请看readme中的data format。
# 注意TIME_BUFF已经被去除了out of bound的车以及可能的crash

# file_idx = 3153
# start_i = 19
# file_t = "/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/LNDE_Results/Trial_6/2/"
# data = pickle.load(open(file_t+f"{file_idx}.pkl", "rb"))
# timeb = data["states_considered"]
# timeb_considered = timeb[start_i:start_i+6]
# tao = len(timeb_considered)
# print(tao)
# car_nums = [[len(timeb_considered[i]) for i in range(tao)]]
# print(car_nums)

# #剔除掉最后一帧中不存在的车
# print("---------------------------")
# for j in range(tao):
#     car_id_in_last_step = [int(timeb_considered[j][i].id) for i in range(len(timeb_considered[j]))]
#     print(car_id_in_last_step)

# timb_new = []
# for timeb_t in timeb_considered:
#     timb_single = []
#     for car in timeb_t:
#         if int(car.id) in car_id_in_last_step:
#             timb_single.append(car)

#     timb_new.append(timb_single)

# print("---------------------------")
# for j in range(tao):
#     car_id_in_last_step = [int(timeb_considered[j][i].id) for i in range(len(timeb_considered[j]))]
#     print(car_id_in_last_step)

# print("---------------------------")
# for j in range(tao):
#     car_id_in_last_step = [int(timb_new[j][i].id) for i in range(len(timb_new[j]))]
#     print(car_id_in_last_step)



file_t = "/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/LNDE_Results/Trial_7/2/" # num: 7258
print(len(os.listdir(file_t)))

#scp -r mtl@35.3.214.134:~/Documents/conflict_detection/Conflict-Identifier-Network/behavior_net/ /home/hanhy/ondemand/data/sys/myjobs/Identifier_Update/