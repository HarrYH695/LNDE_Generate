import numpy as np
import torch
import os
import pickle

if __name__ == '__main__':
    # dir_path = '/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/LNDE_Training_Res/results_gmn_ignore_0726_2/training/behavior_net/'
    # pickle_path = dir_path + 'rounD_nG3_trial_3_3/vis_training/branch_num.pkl'

    # with open(pickle_path, 'rb') as f1:
    #     data = pickle.load(f1)

    # print(data)

    # dataset_file = '/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/LNDE_Data/data/training/behavior_net/rounD/rounD-filtered-VRUs-no-trailer-local-heading-size-36-18/train/'
    # sub_dir = os.listdir(dataset_file)

    # num_all = 0

    # for sub in sub_dir:
    #     subsub_dir = os.listdir(os.path.join(dataset_file, sub))
    #     for subsub in subsub_dir:
    #         files = os.listdir(os.path.join(dataset_file, sub, subsub))
    #         num_all += len(files)

    # print(num_all)

    # vehicle_list = pickle.load(open('/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/LNDE_Data/data/training/behavior_net/rounD/rounD-filtered-VRUs-no-trailer-local-heading-size-36-18/train/rounD-17/08/00011311.pickle', "rb"))
    # print(vehicle_list)

    dir_path = '/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/LNDE_inference_data/LNDE_ignore_0726_2/rD_trial_2_4/1_2'
    files = os.listdir(dir_path)
    print(len(files))

    # python z_try3.py