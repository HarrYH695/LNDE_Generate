import numpy as np
import torch
import pickle
from tqdm import tqdm
import time
# TIME_BUFF: [idx - history_length, idx)时间段内的车辆位置信息,TIME_BUFF[i]代表i时刻的所有车辆的相关信息，具体请看readme中的data format。
# 注意TIME_BUFF已经被去除了out of bound的车以及可能的crash

# test_file_path = "/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/data/inference/rounD/simulation_initialization/initial_clips/rounD-09/01/00000001.pickle"
# v_list = pickle.load(open(test_file_path, "rb"))
# v_1 = v_list[0]
# print(type(v_1))
# print(v_1.id)

# with open('0.pkl', 'rb') as f:
#     loaded_data = pickle.load(f)
    
# print(loaded_data)

a = [1,2,3,4,5,0,10,9,22,6]
print(a[1:7])
# a.append(-1)
# print(a)
# a.append(0)
# print(a)


#data['agent']['valid_mask'] = torch.tensor(data['agent']['valid_mask'], dtype=torch.bool)
#data['agent']['position'] = torch.tensor(data['agent']['position'], dtype=torch.float32)
#data['agent']['heading'] = torch.tensor(data['agent']['heading'], dtype=torch.float32)
#data['agent']['heading'] = torch.where(data['agent']['heading'] > torch.pi, data['agent']['heading'] - 2 * torch.pi, data['agent']['heading'])
#data['agent']['velocity'] = torch.tensor(velocity_3d, dtype=torch.float32)
