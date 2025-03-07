import numpy as np
import torch

# TIME_BUFF: [idx - history_length, idx)时间段内的车辆位置信息,TIME_BUFF[i]代表i时刻的所有车辆的相关信息，具体请看readme中的data format。
# 注意TIME_BUFF已经被去除了out of bound的车以及可能的crash

#