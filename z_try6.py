import cv2
import numpy as np
import os
import pickle
from matplotlib import pyplot as plt
from tqdm import tqdm

if __name__=="__main__":
    # path = '/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/Data_Gen/data_ignore_new_all_0805/train/'
    # path_2 = '/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/Data_Gen/data_ignore_new_single_0805/train_2/'
    # files = os.listdir(path)

    # os.makedirs(path_2, exist_ok=True)

    # for file in tqdm(files):
    #     data = pickle.load(open(path + file, 'rb'))

    #     for i in range(len(data) - 5):
    #         with open(path_2 + file[:-4] + f'_{i}.pkl', 'wb') as f1:
    #             pickle.dump(data[i:i+6], f1)

    # print(len(os.listdir(path)))

    print(min(1, 0))