import numpy as np
import torch
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import cv2

if __name__ == '__main__':
    dir_path = '/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/LNDE_Training_Res/results_gmn_ignore_0809/training/behavior_net/'
    pickle_path = dir_path + 'rounD_nG3_t1/vis_training/branch_num.pkl'

    with open(pickle_path, 'rb') as f1:
        data = pickle.load(f1)

    print(data)

    dir_path = '/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/LNDE_Training_Res/results_gmn_ignore_0809/training/behavior_net/'
    pickle_path = dir_path + 'rounD_nG3_t3/vis_training/branch_num.pkl'

    with open(pickle_path, 'rb') as f1:
        data = pickle.load(f1)

    print(data)

    dir_path = '/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/LNDE_Training_Res/results_gmn_ignore_0809/training/behavior_net/'
    pickle_path = dir_path + 'rounD_nG3_t5/vis_training/branch_num.pkl'

    with open(pickle_path, 'rb') as f1:
        data = pickle.load(f1)

    print(data)

    # dataset_file = '/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/LNDE_Data/data/training/behavior_net/rounD/rounD-filtered-VRUs-no-trailer-local-heading-size-36-18/train/'
    # sub_dir = os.listdir(dataset_file)

    # num_all = 0

    # for sub in sub_dir:
    #     subsub_dir = os.listdir(os.path.join(dataset_file, sub))
    #     for subsub in subsub_dir:
    #         files = os.listdir(os.path.join(dataset_file, sub, subsub))
    #         num_all += len(files)

    # print(num_all)

    # # vehicle_list = pickle.load(open('/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/LNDE_Data/data/training/behavior_net/rounD/rounD-filtered-VRUs-no-trailer-local-heading-size-36-18/train/rounD-17/08/00011311.pickle', "rb"))
    # # print(vehicle_list)

    # dir_path = '/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/Data_Gen/data_ignore_new_single_0805_2/train/'
    # files = os.listdir(dir_path)
    # print(len(files))


    # dir_path = '/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/Data_Gen/data_ignore_new_single_0805/train/'
    # files = os.listdir(dir_path)
    # print(len(files))


    # dir_path = '/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/Data_Gen/data_ignore_new_single_0805/train_2/'
    # files = os.listdir(dir_path)
    # print(len(files))



    # dir_path = '/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/Data_Gen/data_ignore_new_single_0805/val/'
    # files = os.listdir(dir_path)
    # print(len(files))

    # dir_path = '/nfs/turbo/coe-mcity/hanhy/LNDE_Data/Data_Gen/data_ignore_new_single_0809/train/'
    # files = os.listdir(dir_path)
    # print(len(files))

    # dir_path = '/nfs/turbo/coe-mcity/hanhy/LNDE_Data/Data_Gen/data_ignore_new_single_0805/train/'
    # files = os.listdir(dir_path)
    # print(len(files))

    # dir_path = '/nfs/turbo/coe-mcity/hanhy/LNDE_Data/Data_Gen/data_ignore_new_single_0805/train_2/'
    # files = os.listdir(dir_path)
    # print(len(files))

    # dir_path = '/nfs/turbo/coe-mcity/hanhy/LNDE_Data/Data_Gen/data_ignore_new_single_0805/val/'
    # files = os.listdir(dir_path)
    # print(len(files))


    # data = pickle.load(open('/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/heat_matrices.pkl', 'rb'))

    # h_matrix = data['heat_matrix']
    # # h_matrix[h_matrix >= 4] = 4
    # print(np.max(h_matrix))
    # print(np.min(h_matrix))
    # print(np.where(h_matrix == np.max(h_matrix)))
    # values = h_matrix.flatten()
    # print(values.shape)
    # print(np.sum(values <= 5))
    # print('111', np.sum(values <= 4) / values.shape[0])

    # h_matrix[h_matrix >= 4] = 4
    # # plt.figure(figsize=(7, 5))
    # # plt.hist(values, bins=100)

    # # plt.title("Value Distribution of Matrix")
    # # plt.xlabel("Value")
    # # plt.ylabel("Frequency")
    # # plt.grid(True)
    # # plt.tight_layout()

    # # plt.savefig('z_fenbu.png')

    # # plt.figure(figsize=(16, 9))
    # # sns.heatmap(h_matrix, cmap='YlOrRd', cbar=True, annot=True)  # annot=True 会显示具体数值
    # # plt.title("Heatmap of GMM Std")
    # B = cv2.imread('/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/LNDE_Data/data/inference/rounD/basemap/rounD-official-map.png')
    # print(B.shape)
    # print(np.max(B))
    # # plt.savefig('z.png')
    # M_norm = (h_matrix - h_matrix.min()) / (h_matrix.max() - h_matrix.min() + 1e-8)
    # B = B.astype(np.float32) / 255.0

    # # Step 2: 构造红色图层，红通道 = 1，其它通道 = 0
    # R_layer = np.zeros_like(B)
    # R_layer[..., 0] = 1.0  # 只开启 R 通道
    # R_layer[..., 0] = 1
    # # Step 3: 将红色图层按热度矩阵加权后与背景图融合
    # alpha = M_norm[..., np.newaxis]  # 扩展为 (h,w,1)
    # blended = np.clip((1 - alpha) * B + alpha * R_layer, 0, 1)

    # plt.imsave("blended_heatmap_t.png", np.uint8(blended * 255))

    # # h_matrix /= np.max(h_matrix)
    # # cv2.imwrite('zzz.png', np.uint(h_matrix*255))



    # python z_try3.py