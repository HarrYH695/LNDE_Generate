import numpy as np
import os
import random
import torch
import torch.distributions as distributions
import copy
from itertools import combinations
import math

from vehicle import Vehicle, Location, Size3d
from trajectory_pool import TrajectoryPool
from road_matching import RoadMatcher, ROIMatcher

from behavior_net.networks_gmn import define_G
from . import utils

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TrafficSimulator_gmm(object):
    """
    Long-time traffic simulator (inference time).
    """

    def __init__(self, model, history_length, pred_length, m_tokens, n_gaussian, sample_times, checkpoint_dir,
                 safety_mapper_ckpt_dir=None, drivable_map_dir=None, device=device,
                 sim_remove_vehicle_area_map=None,
                 v_modeling_size=Size3d(width=1.8, length=3.6, height=1.5), v_modeling_safe_size=Size3d(width=2.0, length=3.8, height=1.5),
                 map_height=936, map_width=1678):

        self.model = model
        self.history_length =history_length
        self.pred_length = pred_length
        self.m_tokens = m_tokens
        self.n_gaussian = n_gaussian
        self.sample_times = sample_times
        self.checkpoint_dir = checkpoint_dir
        self.safety_mapper_ckpt_dir = safety_mapper_ckpt_dir
        self.device = device

        self.net_G = self.initialize_net_G()
        self.net_G.eval()

        # if safety_mapper_ckpt_dir is not None:
        #     self.net_safety_mapper = self.initialize_net_safety_mapper()
        #     self.net_safety_mapper.eval()

        self.road_matcher = RoadMatcher(map_file_dir=drivable_map_dir, map_height=map_height, map_width=map_width)

        self.ROI_matcher = ROIMatcher(drivable_map_dir=drivable_map_dir, sim_remove_vehicle_area_map_dir=sim_remove_vehicle_area_map, map_height=map_height, map_width=map_width)

        # All vehicles are considered as identical size as below.
        self.v_modeling_size = v_modeling_size
        self.v_modeling_safe_size = v_modeling_safe_size

    def initialize_net_G(self):

        net_G = define_G(
            model=self.model, input_dim=4*self.history_length,
            output_dim=4*self.pred_length, m_tokens=self.m_tokens, n_gaussian=self.n_gaussian).to(self.device)

        if self.checkpoint_dir is None:
            # Initialized traffic_sim during the start of training
            pass
        elif os.path.exists(self.checkpoint_dir):
            # initialize network
            print('initializing networks...')
            checkpoint = torch.load(self.checkpoint_dir, map_location=self.device)
            net_G.load_state_dict(checkpoint['model_G_state_dict'])
            # load pre-trained weights
            print('loading pretrained weights...')
        else:
            print('initializing networks...')
            raise NotImplementedError(
                'pre-trained weights %s does not exist...' % self.checkpoint_dir)

        return net_G

    def initialize_net_safety_mapper(self):

        print('initializing neural safety mapper...')
        net_safety_mapper = define_safety_mapper(self.safety_mapper_ckpt_dir, self.m_tokens, device=self.device).to(self.device)
        net_safety_mapper.eval()

        return net_safety_mapper


    def sampling(self, pred_mean, pred_pi, pred_L):
        eps_sample = np.zeros_like(pred_mean)
        for s_t in range(self.sample_times):
            eps_sample += np.random.randn(*pred_mean.shape)
        eps_sample /= self.sample_times
        
        # print("-------in sampling:-------")
        # print(f"pred_mean:{pred_mean.shape}")
        # print(f"eps_sample:{eps_sample.shape}")
        # print(f"pred_pi:{pred_pi.shape}")
        # print(f"pred_L:{pred_L.shape}")
        
        posi = pred_mean + np.squeeze(pred_L @ np.expand_dims(eps_sample, -1), -1) 
        posi = np.sum(np.expand_dims(pred_pi, -1) * posi, axis=2)

        return posi
    

    def run_forwardpass(self, traj_pool):
        """
        Flatten a trajectory pool and run forward pass...
        """

        buff_lat, buff_lon, buff_cos_heading, buff_sin_heading, buff_vid = traj_pool.flatten_trajectory(
            time_length=self.history_length, max_num_vehicles=self.m_tokens, output_vid=True)

        buff_lat = utils.nan_intep_2d(buff_lat, axis=1)
        buff_lon = utils.nan_intep_2d(buff_lon, axis=1)
        buff_cos_heading = utils.nan_intep_2d(buff_cos_heading, axis=1)
        buff_sin_heading = utils.nan_intep_2d(buff_sin_heading, axis=1)

        input_matrix = np.concatenate([buff_lat, buff_lon, buff_cos_heading, buff_sin_heading], axis=1)
        input_matrix = torch.tensor(input_matrix, dtype=torch.float32)

        # # sample an input state from testing data (e.g. 0th state)
        input_matrix = input_matrix.unsqueeze(dim=0) # make sure the input has a shape of N x D
        input_matrix = input_matrix.to(self.device)

        input_matrix[torch.isnan(input_matrix)] = 0.0

        # run prediction
        # print(f"input_matrix:{input_matrix.shape}")
        mean_pos, std_pos, corr, cos_sin_heading, pi_all, out_L = self.net_G(input_matrix)


        # print(f"mean_pos:{mean_pos.shape}")
        # print(f"std_pos:{std_pos.shape}")
        # print(f"corr:{corr.shape}")
        # print(f"cos_sin_heading:{cos_sin_heading.shape}")
        # print(f"pi_all:{pi_all.shape}")
        # print(f"out_L:{out_L.shape}")

        pred_mean_pos = mean_pos.detach().cpu().numpy()
        pred_cos_sin_heading = cos_sin_heading.detach().cpu().numpy()[0, :, :]
        pred_pi = pi_all.detach().cpu().numpy()
        pred_L = out_L.detach().cpu().numpy()
        pred_posi = self.sampling(pred_mean_pos, pred_pi, pred_L)
        pred_posi = np.squeeze(pred_posi, axis=0)
        # print(f"pred_posi:{pred_posi.shape}")

        pred_lat = pred_posi[:,0:self.pred_length]
        pred_lon = pred_posi[:,self.pred_length:]
        pred_cos_heading = pred_cos_sin_heading[:, 0:self.pred_length].astype(np.float64)
        pred_sin_heading = pred_cos_sin_heading[:, self.pred_length:].astype(np.float64)

        pred_vid = buff_vid

        # print(f"pred_lat:{pred_lat.shape}")
        # print(f"pred_lon:{pred_lon.shape}")
        return pred_lat, pred_lon, pred_cos_heading, pred_sin_heading, pred_vid, buff_vid, buff_lat, buff_lon, 0, 0, 0


    def do_safety_mapping(self, pred_lat, pred_lon, pred_cos_heading, pred_sin_heading, pred_vid, buff_vid, output_delta_position_mask=False):
        # Neural safety mapping
        delta_position_mask_list = []

        for i in range(4):  # Four consecutive pass of safety mapping network to guarantee safety.
            pred_lat, pred_lon, delta_position_mask = self.net_safety_mapper(pred_lat=pred_lat, pred_lon=pred_lon, pred_cos_heading=pred_cos_heading, pred_sin_heading=pred_sin_heading,
                                                                             pred_vid=pred_vid, device=self.device)
            delta_position_mask_list.append(delta_position_mask)

        delta_position_mask = delta_position_mask_list[0] + delta_position_mask_list[1] + delta_position_mask_list[2] + delta_position_mask_list[3]
        # delta_position_mask = np.logical_or(delta_position_mask_list[0], delta_position_mask_list[1])

        if output_delta_position_mask:
            return pred_lat, pred_lon, pred_cos_heading, pred_sin_heading, pred_vid, delta_position_mask

        return pred_lat, pred_lon, pred_cos_heading, pred_sin_heading, pred_vid


    def prediction_to_trajectory_rolling_horizon(self, pred_lat, pred_lon, pred_cos_heading, pred_sin_heading, pred_vid, TIME_BUFF, rolling_step):
        """
        convert predicted tensor to trajectory pool.
        TIME_BUFF can be updated in a rolling horizon fashion (not all pred steps are used).
        """
        assert (rolling_step <= len(pred_lat) and rolling_step <= len(TIME_BUFF))

        TIME_BUFF_NEW_tmp = []

        pred_vid = pred_vid[:, 0]

        for i_steps in range(rolling_step):

            vehicle_list = []
            for vj in range(len(pred_vid)):

                id = pred_vid[vj]
                if np.isnan(id):
                    continue

                lat = pred_lat[vj, i_steps]
                lon = pred_lon[vj, i_steps]
                cos_heading = pred_cos_heading[vj, i_steps]
                sin_heading = pred_sin_heading[vj, i_steps]
                heading = np.arctan2(sin_heading, cos_heading) * 180 / np.pi

                v = Vehicle()
                v.location = Location(x=lat, y=lon)
                v.id = str(int(id))
                v.category = 0
                v.speed_heading = heading
                v.size = self.v_modeling_size
                v.safe_size = self.v_modeling_safe_size
                v.update_poly_box_and_realworld_4_vertices()
                v.update_safe_poly_box()
                v.confidence = True

                vehicle_list.append(v)

            TIME_BUFF_NEW_tmp.append(vehicle_list)

        TIME_BUFF_NEW = TIME_BUFF[rolling_step:] + TIME_BUFF_NEW_tmp


        return TIME_BUFF_NEW

    @staticmethod
    def time_buff_to_traj_pool(TIME_BUFF):
        traj_pool = TrajectoryPool()
        for i in range(len(TIME_BUFF)):
            traj_pool.update(TIME_BUFF[i])
        return traj_pool

    def remove_out_of_bound_vehicles(self, TIME_BUFF, dataset=None):

        kick_out_id_list = []
        for i in range(len(TIME_BUFF)):
            for j in range(len(TIME_BUFF[i])):
                v = TIME_BUFF[i][j]
                if not self.road_matcher._within_map(v.location.x, v.location.y):
                    kick_out_id_list.append(v.id)
                    continue

                if self._within_exit_region(v.location.x, v.location.y, dataset=dataset):
                    kick_out_id_list.append(v.id)

        TIME_BUFF_NEW = []
        for i in range(len(TIME_BUFF)):
            vehicle_list = []
            for j in range(len(TIME_BUFF[i])):
                v = TIME_BUFF[i][j]
                if v.id not in kick_out_id_list:
                    vehicle_list.append(v)
            TIME_BUFF_NEW.append(vehicle_list)

        return TIME_BUFF_NEW
    
    def label_out_of_bound_vehicles(self, TIME_BUFF, dataset=None):

        kick_out_id_list = []
        for i in range(len(TIME_BUFF)):
            for j in range(len(TIME_BUFF[i])):
                v = TIME_BUFF[i][j]
                if not self.road_matcher._within_map(v.location.x, v.location.y):
                    kick_out_id_list.append(v.id)
                    continue

                if self._within_exit_region(v.location.x, v.location.y, dataset=dataset):
                    kick_out_id_list.append(v.id)

        TIME_BUFF_NEW = []
        for i in range(len(TIME_BUFF)):
            vehicle_list = []
            for j in range(len(TIME_BUFF[i])):
                v = TIME_BUFF[i][j]
                if v.id in kick_out_id_list:
                    v.confidence = False

                vehicle_list.append(v)
            TIME_BUFF_NEW.append(vehicle_list)

        return TIME_BUFF_NEW

    def _within_exit_region(self, lat, lon, dataset=None):
        if dataset == 'rounD' or dataset == 'AA_rdbt':
            pxl_pt = self.road_matcher._world2pxl([lat, lon])
            x0, y0 = pxl_pt[0], pxl_pt[1]
            if self.ROI_matcher.sim_remove_vehicle_area_map[y0, x0] > 128.:  # in the exit area.
                return True
            else:
                return False
        else:
            return False

    @staticmethod
    def collision_check(TIME_BUFF, extra_buffer=False):
        """
        Check whether collision happens
        """
        collision_flag = False
        crash_pair = np.zeros(2)
        for vehicle_list in TIME_BUFF:
            for vehicle_pair in combinations(vehicle_list, r=2):
                v1, v2 = vehicle_pair[0], vehicle_pair[1]
                if extra_buffer:
                    v1_poly, v2_poly = v1.safe_poly_box, v2.safe_poly_box
                else:
                    v1_poly, v2_poly = v1.poly_box, v2.poly_box

                if v1_poly.intersects(v2_poly):
                    collision_flag = True
                    crash_pair[0] = int(v1.id)
                    crash_pair[1] = int(v2.id)
                    break
            if collision_flag:
                break

        return collision_flag, crash_pair