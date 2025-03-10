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

def f():
    return True
print(not f())
