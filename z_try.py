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



# file_t = "/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/LNDE_Results/AA_Trial_1/1/" # num: 4427
# print(len(os.listdir(file_t)))

# file_t2 = "/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/LNDE_Results/AA_Trial_1/2/" # num: 33954
# print(len(os.listdir(file_t2)))
# file_o = "/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/LNDE_Results/AA_Trial_1/2/"
# print(len(os.listdir(file_o)))

position = np.array([[1, 0],[1, 3],[1, 2]], dtype=np.float32)
print(position.shape)
heading = []
for j in range(1, len(position)):
    if position[j][0] == -1 or position[j-1][0] == -1:
        continue
    heading.append([np.arctan2(position[j][1] - position[j - 1][1], position[j][0] - position[j - 1][0]), j])

for j in range(len(heading) - 1):
    if abs(heading[j][0] - heading[j + 1][0]) < np.pi / 4 or abs(2 * np.pi - abs(heading[j][0] - heading[j + 1][0])) < np.pi / 4:
        continue
    print(heading[j + 1][1])
    print(abs(heading[j][0] - heading[j + 1][0]))
    print(abs(2 * np.pi - abs(heading[j][0] - heading[j + 1][0])))


#scp -r mtl@35.3.214.134:~/Documents/conflict_detection/Conflict-Identifier-Network/behavior_net/ /home/hanhy/ondemand/data/sys/myjobs/Identifier_Update/