import numpy as np
import pickle
import os
import glob
import random
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, RandomSampler
from trajectory_pool import TrajectoryPool

# import simulation_modeling.utils as utils
from . import utils


class MTLTrajectoryPredictionDataset(Dataset):
    """
    Pytorch Dataset Loader...
    """

    def __init__(self, path_to_traj_data, history_length, pred_length, max_num_vehicles, is_train, dataset='AA_rdbt'):
        self.history_length = history_length
        self.pred_length = pred_length
        self.max_num_vehicles = max_num_vehicles
        self.is_train = is_train
        self.dataset = dataset

        if self.dataset == 'rounD' or self.dataset == 'AA_rdbt':
            split = 'train' if is_train else 'val'
            subfolders = sorted(os.listdir(os.path.join(path_to_traj_data, split)))
            subsubfolders = [sorted(os.listdir(os.path.join(path_to_traj_data, split, subfolders[i]))) for i in
                             range(len(subfolders))]

            self.traj_dirs = []
            self.each_subfolder_size = []
            self.each_subsubfolder_size = []
            for i in range(len(subfolders)):
                one_video = []
                each_subsubfolder_size_tmp = []
                for j in range(len(subsubfolders[i])):
                    files_list = sorted(glob.glob(os.path.join(path_to_traj_data, split, subfolders[i], subsubfolders[i][j], '*.pickle')))
                    one_video.append(files_list)
                    each_subsubfolder_size_tmp.append(len(files_list))
                self.traj_dirs.append(one_video)
                self.each_subfolder_size.append(sum([len(listElem)for listElem in one_video]))
                self.each_subsubfolder_size.append(each_subsubfolder_size_tmp)

            self.subfolder_data_proportion = [self.each_subfolder_size[i]/sum(self.each_subfolder_size) for i in range(len(self.each_subfolder_size))]
            self.subsubfolder_data_proportion = [[self.each_subsubfolder_size[i][j]/sum(self.each_subsubfolder_size[i]) for j in range(len(self.each_subsubfolder_size[i]))] for i in range(len(self.each_subsubfolder_size))]

        else:
            raise NotImplementedError( 'Wrong dataset name %s (choose one from [AA_rdbt, rounD,...])' % self.dataset)

    def __len__(self):
        if self.dataset == 'rounD' or self.dataset == 'AA_rdbt':
            return sum(self.each_subfolder_size)
        else:
            raise NotImplementedError( 'Wrong dataset name %s (choose one from [AA_rdbt, rounD,...])' % self.dataset)

    def __getitem__(self, idx):

        if self.dataset == 'rounD' or self.dataset == 'AA_rdbt':
            subfolder_id = random.choices(range(len(self.each_subfolder_size)), weights=self.subfolder_data_proportion)[0]
            subsubfolder_id = random.choices(range(len(self.traj_dirs[subfolder_id])), weights=self.subsubfolder_data_proportion[subfolder_id])[0]
            datafolder_dirs = self.traj_dirs[subfolder_id][subsubfolder_id]

            idx_start = self.history_length + 1
            idx_end = len(datafolder_dirs) - self.pred_length - 1
            idx = random.randint(idx_start, idx_end)

        else:
            raise NotImplementedError( 'Wrong dataset name %s (choose one from [AA_rdbt, rounD,...])' % self.dataset)

        traj_pool = self.fill_in_traj_pool(t0=idx, datafolder_dirs=datafolder_dirs)
        buff_lat, buff_lon, buff_cos_heading, buff_sin_heading = traj_pool.flatten_trajectory(
            max_num_vehicles=self.max_num_vehicles, time_length=self.history_length+self.pred_length)

        input_matrix, gt_matrix = self.make_training_data_pair(buff_lat, buff_lon, buff_cos_heading, buff_sin_heading)

        input_matrix = torch.tensor(input_matrix, dtype=torch.float32)
        gt_matrix = torch.tensor(gt_matrix, dtype=torch.float32)
        data = {'input': input_matrix, 'gt': gt_matrix}

        return data

    def fill_in_traj_pool(self, t0, datafolder_dirs):
        # read frames within a time interval
        traj_pool = TrajectoryPool()
        for i in range(t0-self.history_length+1, t0+self.pred_length+1):
            vehicle_list = pickle.load(open(datafolder_dirs[i], "rb"))
            traj_pool.update(vehicle_list)
        return traj_pool


    def make_training_data_pair(self, buff_lat, buff_lon, buff_cos_heading, buff_sin_heading):

        buff_lat_in = buff_lat[:, 0:self.history_length]
        buff_lat_out = buff_lat[:, self.history_length:]
        buff_lon_in = buff_lon[:, 0:self.history_length]
        buff_lon_out = buff_lon[:, self.history_length:]
        buff_cos_heading_in = buff_cos_heading[:, 0:self.history_length]
        buff_cos_heading_out = buff_cos_heading[:, self.history_length:]
        buff_sin_heading_in = buff_sin_heading[:, 0:self.history_length]
        buff_sin_heading_out = buff_sin_heading[:, self.history_length:]

        buff_lat_in = utils.nan_intep_2d(buff_lat_in, axis=1)
        buff_lon_in = utils.nan_intep_2d(buff_lon_in, axis=1)
        buff_cos_heading_in = utils.nan_intep_2d(buff_cos_heading_in, axis=1)
        buff_sin_heading_in = utils.nan_intep_2d(buff_sin_heading_in, axis=1)

        input_matrix = np.concatenate([buff_lat_in, buff_lon_in, buff_cos_heading_in, buff_sin_heading_in], axis=1)
        gt_matrix = np.concatenate([buff_lat_out, buff_lon_out, buff_cos_heading_out, buff_sin_heading_out], axis=1)

        # # mask-out those output traj whose input is nan
        gt_matrix[np.isnan(input_matrix).sum(1) > 0, :] = np.nan

        # shuffle the order of input tokens
        input_matrix, gt_matrix = self._shuffle_tokens(input_matrix, gt_matrix)

        # data augmentation
        input_matrix = self._data_augmentation(input_matrix, pos_scale=0.05, heading_scale=0.001)

        return input_matrix, gt_matrix

    @staticmethod
    def _shuffle_tokens(input_matrix, gt_matrix):

        max_num_vehicles = input_matrix.shape[0]
        shuffle_id = list(range(0, max_num_vehicles))
        random.shuffle(shuffle_id)

        input_matrix = input_matrix[shuffle_id, :]
        gt_matrix = gt_matrix[shuffle_id, :]

        return input_matrix, gt_matrix

    def _data_augmentation(self, input_matrix, pos_scale=1.0, heading_scale=1.0):
        pos_mask, heading_mask = np.ones_like(input_matrix), np.ones_like(input_matrix)
        pos_mask[:, 2*self.history_length:] = 0
        heading_mask[:, :2*self.history_length] = 0

        pos_rand = pos_scale * utils.randn_like(input_matrix) * pos_mask
        heading_rand = heading_scale * utils.randn_like(input_matrix) * heading_mask

        augmented_input = input_matrix + pos_rand + heading_rand

        return augmented_input



class MTLTrajectoryPredictionDataset_Ignore(Dataset):
    """
    Pytorch Dataset Loader...
    """

    def __init__(self, path_to_traj_data_new_train, path_to_traj_data_new_val, history_length, pred_length, max_num_vehicles, is_train, dataset='AA_rdbt'):
        self.history_length = history_length
        self.pred_length = pred_length
        self.max_num_vehicles = max_num_vehicles
        self.is_train = is_train
        self.dataset = dataset

        if self.dataset == 'rounD' or self.dataset == 'AA_rdbt':
            if is_train:
                self.path_ignore = path_to_traj_data_new_train
            else:
                self.path_ignore = path_to_traj_data_new_val

            self.data_files_ignore = os.listdir(self.path_ignore)
            self.path_ignore_len = len(self.data_files_ignore)
        else:
            raise NotImplementedError( 'Wrong dataset name %s (choose one from [AA_rdbt, rounD,...])' % self.dataset)

    def __len__(self):
        if self.dataset == 'rounD' or self.dataset == 'AA_rdbt':
            return self.path_ignore_len
        else:
            raise NotImplementedError( 'Wrong dataset name %s (choose one from [AA_rdbt, rounD,...])' % self.dataset)

    def __getitem__(self, idx):

        if self.dataset == 'rounD' or self.dataset == 'AA_rdbt':
            datafolder_dirs = ''
            idx = random.randint(0, self.path_ignore_len - 2)

        else:
            raise NotImplementedError( 'Wrong dataset name %s (choose one from [AA_rdbt, rounD,...])' % self.dataset)

        traj_pool = self.fill_in_traj_pool(t0=idx, datafolder_dirs=datafolder_dirs)
        buff_lat, buff_lon, buff_cos_heading, buff_sin_heading = traj_pool.flatten_trajectory(
            max_num_vehicles=self.max_num_vehicles, time_length=self.history_length+self.pred_length)

        input_matrix, gt_matrix = self.make_training_data_pair(buff_lat, buff_lon, buff_cos_heading, buff_sin_heading)

        input_matrix = torch.tensor(input_matrix, dtype=torch.float32)
        gt_matrix = torch.tensor(gt_matrix, dtype=torch.float32)
        data = {'input': input_matrix, 'gt': gt_matrix}

        return data

    def fill_in_traj_pool(self, t0, datafolder_dirs):
        traj_pool = TrajectoryPool()
        with open(os.path.join(self.path_ignore, self.data_files_ignore[t0]), 'rb') as fb:
            vehicle_list_all = pickle.load(fb)

        for i in range(len(vehicle_list_all)):
            traj_pool.update(vehicle_list_all[i])
            
        return traj_pool


    def make_training_data_pair(self, buff_lat, buff_lon, buff_cos_heading, buff_sin_heading):

        buff_lat_in = buff_lat[:, 0:self.history_length]
        buff_lat_out = buff_lat[:, self.history_length:]
        buff_lon_in = buff_lon[:, 0:self.history_length]
        buff_lon_out = buff_lon[:, self.history_length:]
        buff_cos_heading_in = buff_cos_heading[:, 0:self.history_length]
        buff_cos_heading_out = buff_cos_heading[:, self.history_length:]
        buff_sin_heading_in = buff_sin_heading[:, 0:self.history_length]
        buff_sin_heading_out = buff_sin_heading[:, self.history_length:]

        buff_lat_in = utils.nan_intep_2d(buff_lat_in, axis=1)
        buff_lon_in = utils.nan_intep_2d(buff_lon_in, axis=1)
        buff_cos_heading_in = utils.nan_intep_2d(buff_cos_heading_in, axis=1)
        buff_sin_heading_in = utils.nan_intep_2d(buff_sin_heading_in, axis=1)

        input_matrix = np.concatenate([buff_lat_in, buff_lon_in, buff_cos_heading_in, buff_sin_heading_in], axis=1)
        gt_matrix = np.concatenate([buff_lat_out, buff_lon_out, buff_cos_heading_out, buff_sin_heading_out], axis=1)

        # # mask-out those output traj whose input is nan
        gt_matrix[np.isnan(input_matrix).sum(1) > 0, :] = np.nan

        # shuffle the order of input tokens
        input_matrix, gt_matrix = self._shuffle_tokens(input_matrix, gt_matrix)

        # data augmentation
        input_matrix = self._data_augmentation(input_matrix, pos_scale=0.05, heading_scale=0.001)

        return input_matrix, gt_matrix

    @staticmethod
    def _shuffle_tokens(input_matrix, gt_matrix):

        max_num_vehicles = input_matrix.shape[0]
        shuffle_id = list(range(0, max_num_vehicles))
        random.shuffle(shuffle_id)

        input_matrix = input_matrix[shuffle_id, :]
        gt_matrix = gt_matrix[shuffle_id, :]

        return input_matrix, gt_matrix

    def _data_augmentation(self, input_matrix, pos_scale=1.0, heading_scale=1.0):
        pos_mask, heading_mask = np.ones_like(input_matrix), np.ones_like(input_matrix)
        pos_mask[:, 2*self.history_length:] = 0
        heading_mask[:, :2*self.history_length] = 0

        pos_rand = pos_scale * utils.randn_like(input_matrix) * pos_mask
        heading_rand = heading_scale * utils.randn_like(input_matrix) * heading_mask

        augmented_input = input_matrix + pos_rand + heading_rand

        return augmented_input




def get_loaders(configs):

    if configs["dataset"] == 'AA_rdbt' or configs["dataset"] == 'rounD':
        training_set_origin = MTLTrajectoryPredictionDataset(path_to_traj_data=configs["path_to_traj_data"], history_length=configs["history_length"], pred_length=configs["rollout_num"],
                                                      max_num_vehicles=configs["max_num_vehicles"], is_train=True, dataset=configs["dataset"])

        training_set_ignore = MTLTrajectoryPredictionDataset_Ignore(path_to_traj_data_new_train=configs['path_to_traj_data_new_train'], path_to_traj_data_new_val=configs['path_to_traj_data_new_val'], history_length=configs["history_length"], pred_length=configs["rollout_num"],
                                                 max_num_vehicles=configs["max_num_vehicles"], is_train=True, dataset=configs["dataset"])

        val_set_origin = MTLTrajectoryPredictionDataset(path_to_traj_data=configs["path_to_traj_data"], history_length=configs["history_length"], pred_length=configs["rollout_num"],
                                                 max_num_vehicles=configs["max_num_vehicles"], is_train=False, dataset=configs["dataset"])

        val_set_ignore = MTLTrajectoryPredictionDataset_Ignore(path_to_traj_data_new_train=configs['path_to_traj_data_new_train'], path_to_traj_data_new_val=configs['path_to_traj_data_new_val'], history_length=configs["history_length"], pred_length=configs["rollout_num"],
                                                 max_num_vehicles=configs["max_num_vehicles"], is_train=False, dataset=configs["dataset"])

    else:
        raise NotImplementedError(
            'Wrong dataset name %s (choose one from [AA_rdbt, rounD,...])'
            % configs.dataset)

    concat_data_train = ConcatDataset([training_set_origin, training_set_ignore])
    concat_data_val = ConcatDataset([val_set_origin, val_set_ignore])
    # datasets = {'train': training_set, 'val': val_set}
    # dataloaders = {x: DataLoader(datasets[x], batch_size=configs["batch_size"],
    #                              shuffle=True, num_workers=configs["dataloader_num_workers"])
    #                for x in ['train', 'val']}

    dataloaders = {}
    dataloaders['train'] = DataLoader(concat_data_train, batch_size=configs["batch_size"], shuffle=True, num_workers=configs["dataloader_num_workers"], drop_last=True, pin_memory=True)
    dataloaders['val'] = DataLoader(concat_data_val, batch_size=configs["batch_size"], shuffle=False, num_workers=configs["dataloader_num_workers"], drop_last=False, pin_memory=True)


    return dataloaders

if __name__ == '__main__':
    configs = {}
    configs['batch_size'] = 256
    configs['dataloader_num_workers'] = 8
    configs['max_num_vehicles'] = 32
    configs['history_length'] = 5
    configs['rollout_num'] = 1

    configs['dataset'] = 'rounD'
    configs['path_to_traj_data'] = "/nfs/turbo/coe-mcity/hanhy/LNDE_Data/data/training/behavior_net/rounD/rounD-filtered-VRUs-no-trailer-local-heading-size-36-18"
    configs['path_to_traj_data_new_train'] = "/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/Data_Gen/data_ignore_new_single_0805/train_2/"
    configs['path_to_traj_data_new_val'] = "/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/Data_Gen/data_ignore_new_single_0805/val/"

    dataloader_test = get_loaders(configs)

    test_num = 0
    for batch_id, batch in enumerate(dataloader_test['train'], 0):
        test_num = batch_id
    
    print(test_num)

    test_num = 0

    for batch_id, batch in enumerate(dataloader_test['val'], 0):
        test_num = batch_id
    
    print(test_num)



