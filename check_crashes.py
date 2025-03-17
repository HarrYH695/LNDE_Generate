import numpy as np
import torch
import pickle

result_dir = "/nfs/turbo/coe-mcity/hanhy/LNDE_Results/Trial_3/"

for i in range(3000):
    file = result_dir + f"{i}.pkl"
    with open(file, 'rb') as f:
        infos = pickle.load(f)
        print(infos["inference_step"])
        TB = infos["inital_state"][0]
        ids = [TB[j].id for j in range(len(TB))]
        print(ids)
        TB = infos["inital_state"][-1]
        ids = [TB[j].id for j in range(len(TB))]
        print(ids)