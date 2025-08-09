import cv2
import numpy as np
import os
import pickle
from matplotlib import pyplot as plt
from tqdm import tqdm
import random
import math

if __name__=="__main__":
    path = '/nfs/turbo/coe-mcity/hanhy/LNDE_Data/Data_Gen/data_ignore_new_all_0805/train/'
    path_2 = '/nfs/turbo/coe-mcity/hanhy/LNDE_Data/Data_Gen/data_ignore_new_single_0805/train_part_p60/'
    files = os.listdir(path)

    os.makedirs(path_2, exist_ok=True)

    len_origin = math.floor(len(files) * 0.6)
    new_files = random.sample(files, len_origin)

    for file in tqdm(new_files):
        data = pickle.load(open(path + file, 'rb'))

        for i in range(len(data) - 6):
            with open(path_2 + file[:-4] + f'_{i}.pkl', 'wb') as f1:
                pickle.dump(data[i:i+6], f1)

    print(len(os.listdir(path_2)))
