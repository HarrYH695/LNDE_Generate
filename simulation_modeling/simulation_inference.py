import os
import glob
import random
import pickle
import numpy as np
import cv2
import torch
import json
import pandas as pd
import time
import copy
from simulation_modeling.traffic_simulator import TrafficSimulator
from simulation_modeling.crashcritic import CrashCritic
from simulation_modeling.trajectory_interpolator import TrajInterpolator
from simulation_modeling.vehicle_generator import AA_rdbt_TrafficGenerator, rounD_TrafficGenerator
from sim_evaluation_metric.realistic_metric import RealisticMetrics
from itertools import combinations
from basemap import Basemap
from tqdm import tqdm

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SimulationInference(object):
    def __init__(self, configs):

        self.dataset = configs["dataset"]
        self.history_length, self.pred_length, self.m_tokens = configs["history_length"], configs["pred_length"], configs["m_tokens"]
        self.rolling_step = configs["rolling_step"]
        self.sim_wall_time = configs["sim_wall_time"]
        self.sim_resol = configs["sim_resol"]
        self.use_neural_safety_mapping = configs["use_neural_safety_mapping"]
        self.background_map = Basemap(map_file_dir=configs["basemap_dir"], map_height=configs["map_height"], map_width=configs["map_width"])
        self.viz_flag = configs["viz_flag"]
        self.save_viz_flag = configs["save_viz_flag"]
        self.save_collision_data_flag = configs["save_collision_data_flag"]
        self.save_fps = configs["save_fps"]
        self.save_viz_folder = configs["save_viz_folder"]
        self.device = configs["device"]

        if configs["dataset"] == "AA_rdbt":
            self.traffic_generator = AA_rdbt_TrafficGenerator(config=configs["traffic_generator_config"])
        elif configs["dataset"] == 'rounD':
            self.traffic_generator = rounD_TrafficGenerator(config=configs["traffic_generator_config"])
        else:
            raise NotImplementedError('{0} does not supported yet...Choose from ["AA_rdbt", "rounD"].'.format(configs["dataset"]))

        self.max_m_steps = int((self.sim_wall_time / self.sim_resol) / self.rolling_step)
        self.traj_dirs, self.subfolder_data_proportion, self.subsubfolder_data_proportion = self._get_traj_dirs(path_to_traj_data=configs["init_traj_clips_dir"])

        self.sim = TrafficSimulator(model=configs["model"], history_length=self.history_length, pred_length=self.pred_length, m_tokens=self.m_tokens,
                                    checkpoint_dir=configs["behavior_model_ckpt_dir"],
                                    safety_mapper_ckpt_dir=configs["safety_mapper_ckpt_dir"], drivable_map_dir=configs["drivable_map_dir"], device=self.device,
                                    sim_remove_vehicle_area_map=configs["sim_remove_vehicle_area_map"], map_height=configs["map_height"], map_width=configs["map_width"])

        # Basic simulation results of all episodes, units: [episodes], [sim steps], [m], [s]
        # Will always be saved regardless of realistic_metric (e.g., vehicle speed, distance, etc.)
        self.colli_num, self.total_sim_steps, self.total_travel_distances, self.total_sim_wall_time = 0, 0, 0, 0
        # Basic simulation stats of one simulation episode (the current simulating episode).
        self.one_sim_TIME_BUFF, self.one_sim_steps_all_vehicles, self.one_sim_travel_distances_all_vehicles = None, None, None
        self.one_sim_wall_time, self.one_sim_colli_flag, self.one_sim_TIME_BUFF_last_moment, self.one_sim_TIME_BUFF_newly_generated, self.one_sim_whole_sim_do_safety_mapping_flag = 0., False, None, None, True

        self.gen_realistic_metric_flag = configs["gen_realistic_metric_flag"]  # Whether to generate metrics.
        self.gen_realistic_metric_dict = configs["gen_realistic_metric_dict"]  # What metrics to generate.
        self.realistic_metric_save_folder = configs["realistic_metric_save_folder"]
        os.makedirs(self.realistic_metric_save_folder, exist_ok=True)

        if self.gen_realistic_metric_flag:

            # ROIs
            circle_map_dir = os.path.join(configs["ROI_map_dir"], 'circle')
            entrance_map_dir = os.path.join(configs["ROI_map_dir"], 'entrance')
            exit_map_dir = os.path.join(configs["ROI_map_dir"], 'exit')
            crosswalk_map_dir = None  # os.path.join(configs["ROI_map_dir"], 'crosswalk')
            yielding_area_map_dir = os.path.join(configs["ROI_map_dir"], 'yielding-area')
            at_circle_lane_map_dir = os.path.join(configs["ROI_map_dir"], 'at-circle-lane')

            # PET (post-encroachment time) analysis configs
            if self.dataset == 'AA_rdbt':
                basemap_img = cv2.imread(configs["basemap_dir"], cv2.IMREAD_COLOR)
                basemap_img = cv2.cvtColor(basemap_img, cv2.COLOR_BGR2RGB)
                basemap_img = cv2.resize(basemap_img, (configs["map_width"], configs["map_height"]))
                basemap_img = (basemap_img.astype(np.float64) * 0.6).astype(np.uint8)

                PET_configs = configs["PET_configs"]
                PET_configs["basemap_img"] = basemap_img

            elif self.dataset == 'rounD':
                basemap_img = cv2.imread(configs["basemap_dir"], cv2.IMREAD_COLOR)
                basemap_img = cv2.cvtColor(basemap_img, cv2.COLOR_BGR2RGB)
                basemap_img = cv2.resize(basemap_img, (configs["map_width"], configs["map_height"]))
                basemap_img = (basemap_img.astype(np.float64) * 0.6).astype(np.uint8)

                PET_configs = configs["PET_configs"]
                PET_configs["basemap_img"] = basemap_img

            self.SimMetricsAnalyzer = RealisticMetrics(drivable_map_dir=configs["drivable_map_dir"], sim_remove_vehicle_area_map=configs["sim_remove_vehicle_area_map"],
                                                       circle_map_dir=circle_map_dir, entrance_map_dir=entrance_map_dir, exit_map_dir=exit_map_dir,
                                                       crosswalk_map_dir=crosswalk_map_dir, yielding_area_map_dir=yielding_area_map_dir, at_circle_lane_map_dir=at_circle_lane_map_dir,
                                                       sim_resol=self.sim_resol,
                                                       map_height=configs["map_height"], map_width=configs["map_width"],
                                                       PET_configs=PET_configs)

            self.output_instant_speed_list = []  # This list is instant speed in the circle
            self.output_yielding_conflict_dist_and_v_dict_list = []
            self.output_distance_all_vehicle_pairs_list_three_circle = []
            self.output_PET_list = []  # Post-encroachment time results.

        self.save_simulated_TIME_BUFF_flag = configs["save_simulated_TIME_BUFF_flag"]
        self.simulated_TIME_BUFF_save_folder = configs["simulated_TIME_BUFF_save_folder"]

        # initialize conflict critic
        self.use_conflict_critic_module = configs["use_conflict_critic_module"]
        if self.use_conflict_critic_module:
            self.conflict_critic_agent = CrashCritic(sim_resol=self.sim_resol)

        self.interpolate_flag = configs["interpolate_flag"]
        if self.interpolate_flag:
            # initialize the trajectory interpolator
            # The number of steps interpolate between predicted steps. For example, if resolution is 0.4s and steps is 3, then new resolution is 0.1s.
            self.intep_steps = configs["intep_steps"]
            self.traj_interpolator = TrajInterpolator()

    @staticmethod
    def _get_traj_dirs(path_to_traj_data):
        subfolders = sorted(os.listdir(os.path.join(path_to_traj_data)))
        subsubfolders = [sorted(os.listdir(os.path.join(path_to_traj_data, subfolders[i]))) for i in
                         range(len(subfolders))]

        traj_dirs = []
        each_subfolder_size = []
        each_subsubfolder_size = []
        for i in range(len(subfolders)):
            one_video = []
            each_subsubfolder_size_tmp = []
            for j in range(len(subsubfolders[i])):
                files_list = sorted(glob.glob(os.path.join(path_to_traj_data, subfolders[i], subsubfolders[i][j], '*.pickle')))
                one_video.append(files_list)
                each_subsubfolder_size_tmp.append(len(files_list))
            traj_dirs.append(one_video)
            each_subfolder_size.append(sum([len(listElem) for listElem in one_video]))
            each_subsubfolder_size.append(each_subsubfolder_size_tmp)

        subfolder_data_proportion = [each_subfolder_size[i] / sum(each_subfolder_size) for i in range(len(each_subfolder_size))]
        subsubfolder_data_proportion = [[each_subsubfolder_size[i][j] / sum(each_subsubfolder_size[i]) for j in range(len(each_subsubfolder_size[i]))] for i in
                                        range(len(each_subsubfolder_size))]

        return traj_dirs, subfolder_data_proportion, subsubfolder_data_proportion

    def _initialize_sim(self):
        subfolder_id = random.choices(range(len(self.subfolder_data_proportion)), weights=self.subfolder_data_proportion)[0]
        subsubfolder_id = random.choices(range(len(self.traj_dirs[subfolder_id])), weights=self.subsubfolder_data_proportion[subfolder_id])[0]
        datafolder_dirs = self.traj_dirs[subfolder_id][subsubfolder_id]

        idx_start = self.history_length + 1
        idx_end = len(datafolder_dirs) - self.pred_length - 1
        idx = random.randint(idx_start, idx_end)

        TIME_BUFF = []
        # print("start idx: {0}".format(str(t0 - history_length)))
        for i in range(idx - self.history_length, idx):
            vehicle_list = pickle.load(open(datafolder_dirs[i], "rb"))
            for j in vehicle_list:
                j.confidence = True

            TIME_BUFF.append(vehicle_list)

        return TIME_BUFF

    def _cal_travel_distance(self, TIME_BUFF_last_moment, TIME_BUFF_newly_generated):
        TIME_BUFF = TIME_BUFF_last_moment + TIME_BUFF_newly_generated
        traj_pool = self.sim.time_buff_to_traj_pool(TIME_BUFF)

        travel_distance = []
        for vid, v_info in traj_pool.pool.items():
            if len(v_info['vehicle']) == 1:  # This vehicle just been generated
                continue
            v_loc = []
            for v_item in v_info['vehicle']:
                wd_x, wd_y = v_item.location.x, v_item.location.y
                v_loc.append([wd_x, wd_y])

            v_loc_array = np.array(v_loc)  # num_of_steps x 2
            v_loc_diff = np.diff(v_loc_array, axis=0)
            travel_distance_each_step = np.linalg.norm(v_loc_diff, axis=1)
            dist = np.sum(travel_distance_each_step)
            travel_distance.append(dist)

        total_travel_distance = sum(travel_distance)
        return total_travel_distance

    def generate_simulation_metric(self, one_sim_TIME_BUFF):
        """
        The input is the simulation episode index and the simulated trajectories of this episode.
        This function will calculate the simulation metrics (e.g., vehicle speed, distance, etc.)
        """
        if self.interpolate_flag:
            # interpolate the trajectory to a finer resolution first for analysis
            evaluate_metric_TIME_BUFF, new_resol = self.traj_interpolator.interpolate_traj(one_sim_TIME_BUFF, intep_steps=self.intep_steps, original_resolution=self.sim_resol)
            self.SimMetricsAnalyzer.sim_resol = new_resol
        else:
            evaluate_metric_TIME_BUFF = one_sim_TIME_BUFF
        self._gen_sim_metric(one_sim_TIME_BUFF=evaluate_metric_TIME_BUFF)

    def update_basic_sim_metric(self, sim_idx):
        if self.one_sim_colli_flag:
            self.colli_num += 1
        self.total_sim_steps += self.one_sim_steps_all_vehicles
        self.total_travel_distances += self.one_sim_travel_distances_all_vehicles
        self.total_sim_wall_time += self.one_sim_wall_time

        # if sim_idx % 100 == 1:
        if True:  # when sim_wall_time is very long (e.g., 30 mins)
            print(f'Sim number:{sim_idx + 1}, colli num: {self.colli_num}, total sim wall time: '
                  f'{self.total_sim_wall_time} (s), total travel dist: {self.total_travel_distances} (m) \n')
            res_df = pd.DataFrame(np.array([sim_idx + 1, self.colli_num, self.total_sim_wall_time, self.total_travel_distances]).reshape(1, -1),
                                  columns=['Sim number', 'colli num', 'total sim wall time', 'total travel dist'])

            res_df_save_path = os.path.join(self.realistic_metric_save_folder, "res_df.csv")
            res_df.to_csv(res_df_save_path, index=False)

    def run_simulations(self, sim_num, initial_TIME_BUFF=None):
        """
        Run simulations
        """

        for sim_idx in range(sim_num):

            # Run one episode of simulation
            self.run_a_sim(sim_idx=sim_idx, initial_TIME_BUFF=initial_TIME_BUFF)

            # Generate and save simulation metrics
            if self.gen_realistic_metric_flag:
                self.generate_simulation_metric(self.one_sim_TIME_BUFF)
                self.save_sim_metric()
                print("Saving sim metrics!")

            # Save simulation video
            if self.save_viz_flag:
                self.save_time_buff_video(self.one_sim_TIME_BUFF, background_map=self.background_map, file_name=sim_idx, save_path=self.save_viz_folder)

            # Save simulated trajectories
            if self.save_simulated_TIME_BUFF_flag:
                save_viz_path = os.path.join(self.simulated_TIME_BUFF_save_folder, str(sim_idx))
                self.save_trajectory(self.one_sim_TIME_BUFF, save_path=save_viz_path, sim_id=sim_idx, step_id=0)

            # Update, print, and save basic simulation results
            self.update_basic_sim_metric(sim_idx)

    def _gen_sim_metric(self, one_sim_TIME_BUFF):
        # Construct traj dataframe
        self.SimMetricsAnalyzer.construct_traj_data(one_sim_TIME_BUFF)

        # PET analysis
        if self.gen_realistic_metric_dict["PET"]:
            PET_list = self.SimMetricsAnalyzer.PET_analysis()
            self.output_PET_list.append(PET_list)

        # In circle instant speed analysis
        if self.gen_realistic_metric_dict["instant_speed"]:
            instant_speed_list = self.SimMetricsAnalyzer.in_circle_instant_speed_analysis()
            self.output_instant_speed_list.append(instant_speed_list)

        # yielding distance and speed analysis
        if self.gen_realistic_metric_dict["yielding_speed_and_distance"]:
            yielding_conflict_dist_and_v_dict = self.SimMetricsAnalyzer.yilding_distance_and_speed_analysis()
            self.output_yielding_conflict_dist_and_v_dict_list.append(yielding_conflict_dist_and_v_dict)

        # all positions distance distribution analysis
        if self.gen_realistic_metric_dict["distance"]:
            distance_all_vehicle_pairs_list_three_circle = self.SimMetricsAnalyzer.distance_analysis(mode='three_circle', only_in_roundabout_circle=False)
            self.output_distance_all_vehicle_pairs_list_three_circle.append(distance_all_vehicle_pairs_list_three_circle)

    def save_sim_metric(self):
        if self.gen_realistic_metric_dict["PET"]:
            with open(os.path.join(self.realistic_metric_save_folder, "output_PET_list.json"), 'w') as f:
                json.dump(self.output_PET_list, f, indent=4)

        if self.gen_realistic_metric_dict["instant_speed"]:
            with open(os.path.join(self.realistic_metric_save_folder, "output_instant_speed_list.json"), 'w') as f:
                json.dump(self.output_instant_speed_list, f, indent=4)

        if self.gen_realistic_metric_dict["yielding_speed_and_distance"]:
            with open(os.path.join(self.realistic_metric_save_folder, "output_yielding_conflict_dist_and_v_dict_list.json"), 'w') as f:
                json.dump(self.output_yielding_conflict_dist_and_v_dict_list, f, indent=4)

        if self.gen_realistic_metric_dict["distance"]:
            with open(os.path.join(self.realistic_metric_save_folder, "output_distance_list_three_circle.json"), 'w') as f:
                json.dump(self.output_distance_all_vehicle_pairs_list_three_circle, f, indent=4)

    def initialize_sim(self, initial_TIME_BUFF=None):
        """
        Initialize a simulation episode by sampling from trajectory clips or given TIME_BUFF.
        """
        # read frames and initialize trajectory pool. 
        initial_has_collision = True
        while initial_has_collision:
            TIME_BUFF = self._initialize_sim() if initial_TIME_BUFF is None else initial_TIME_BUFF
            TIME_BUFF = self.sim.remove_out_of_bound_vehicles(TIME_BUFF, dataset=self.dataset)  # remove initial vehicles in exit area.    
            traj_pool = self.sim.time_buff_to_traj_pool(TIME_BUFF)
            initial_has_collision,_ = self.sim.collision_check(TIME_BUFF)
            
            if initial_has_collision and initial_TIME_BUFF is not None:
                raise ValueError("The given initial TIME BUFF is not safe, collision in it!")

        # initial_no_collision = True
        # while initial_no_collision:
        # TIME_BUFF = self._initialize_sim() if initial_TIME_BUFF is None else initial_TIME_BUFF
        # TIME_BUFF = self.sim.remove_out_of_bound_vehicles(TIME_BUFF, dataset=self.dataset)  # remove initial vehicles in exit area.    
        # traj_pool = self.sim.time_buff_to_traj_pool(TIME_BUFF)
            # initial_no_collision = not self.sim.collision_check(TIME_BUFF)
            # print(initial_no_collision)
            # if initial_has_collision and initial_TIME_BUFF is not None:
            #     raise ValueError("The given initial TIME BUFF is not safe, collision in it!")

        # Initialize some basic stats for this simulation episode
        self.one_sim_TIME_BUFF = TIME_BUFF
        self.one_sim_steps_all_vehicles = 0  # sum of number of all steps of all vehicles
        self.one_sim_travel_distances_all_vehicles = 0  # sum of all vehicles travel distances
        self.one_sim_wall_time = 0.  # [s] simulation wall time of this simulation episode
        self.one_sim_colli_flag = False
        self.one_sim_TIME_BUFF_last_moment = None
        self.one_sim_TIME_BUFF_newly_generated = None
        self.one_sim_whole_sim_do_safety_mapping_flag = True

        return TIME_BUFF, traj_pool

    def visualize_time_buff(self, TIME_BUFF, tt=0):
        # visualization and write result video
        if self.viz_flag:
            if self.interpolate_flag:
                visualize_TIME_BUFF, _ = self.traj_interpolator.interpolate_traj(TIME_BUFF, intep_steps=self.intep_steps, original_resolution=self.sim_resol)
                freq = self.intep_steps + 1
            else:
                visualize_TIME_BUFF = TIME_BUFF
                freq = 1

            if tt == 0:
                self._visualize_time_buff(visualize_TIME_BUFF, self.background_map)
            else:
                self._visualize_time_buff(visualize_TIME_BUFF[-self.rolling_step * freq:], self.background_map)

    def run_one_sim_step(self, traj_pool, TIME_BUFF):
        """
        Run one simulation step
        """

        # run self-simulation
        pred_lat, pred_lon, pred_cos_heading, pred_sin_heading, pred_vid, buff_vid, current_lat, current_lon, pred_std_x, pred_std_y, pred_corr = self.sim.run_forwardpass(traj_pool)
        output_delta_position_mask = np.zeros(buff_vid.shape, dtype=bool)
        # print(pred_std_x.shape, pred_std_y.shape)
        # determine whether to do safety mapping
        if self.use_neural_safety_mapping:
            if self.use_conflict_critic_module:
                tmp_TIME_BUFF = self.sim.prediction_to_trajectory_rolling_horizon(pred_lat, pred_lon, pred_cos_heading, pred_sin_heading, pred_vid, TIME_BUFF, rolling_step=self.rolling_step)
                do_safety_mapping_flag = self.conflict_critic_agent.main_func(TIME_BUFF=tmp_TIME_BUFF)
            else:
                do_safety_mapping_flag = True  # Always use safety mapping
            self.one_sim_whole_sim_do_safety_mapping_flag = self.one_sim_whole_sim_do_safety_mapping_flag and do_safety_mapping_flag

            if do_safety_mapping_flag:
                pred_lat, pred_lon, pred_cos_heading, pred_sin_heading, pred_vid, output_delta_position_mask = self.sim.do_safety_mapping(pred_lat, pred_lon, pred_cos_heading, pred_sin_heading, pred_vid, buff_vid,
                                                                                                                                          output_delta_position_mask=True)
            else:
                print('Generate a collision!')

        TIME_BUFF_new = self.sim.prediction_to_trajectory_rolling_horizon(pred_lat, pred_lon, pred_cos_heading, pred_sin_heading, pred_std_x, pred_std_y, pred_vid, TIME_BUFF, rolling_step=self.rolling_step)

        return TIME_BUFF_new, pred_vid, output_delta_position_mask #, pred_std_x, pred_std_y

    def update_basic_stats_of_the_current_sim_episode(self, tt, TIME_BUFF, pred_vid):
        if tt == 0:
            self.one_sim_TIME_BUFF_newly_generated = TIME_BUFF
            self.one_sim_TIME_BUFF_last_moment = TIME_BUFF[:1]
        else:
            self.one_sim_TIME_BUFF_newly_generated = TIME_BUFF[-self.rolling_step:]

        # Total steps
        num_vehicle_in_the_sim = (pred_vid[~np.isnan(pred_vid)]).shape[0] / self.pred_length
        self.one_sim_steps_all_vehicles += (num_vehicle_in_the_sim * self.rolling_step)
        # Travel distances
        travel_distance = self._cal_travel_distance(self.one_sim_TIME_BUFF_last_moment, self.one_sim_TIME_BUFF_newly_generated)
        self.one_sim_travel_distances_all_vehicles += travel_distance
        self.one_sim_TIME_BUFF_last_moment = TIME_BUFF[-1:]

    def run_a_sim(self, sim_idx, initial_TIME_BUFF=None):
        """
        Run one simulation
        """

        # Initialize the simulation
        TIME_BUFF, traj_pool = self.initialize_sim(initial_TIME_BUFF=initial_TIME_BUFF)

        # Run this simulation episode
        for tt in range(self.max_m_steps):

            # Visualize frames
            self.visualize_time_buff(TIME_BUFF, tt=tt)

            # Run one simulation step
            TIME_BUFF, pred_vid, output_delta_position_mask = self.run_one_sim_step(traj_pool=traj_pool, TIME_BUFF=TIME_BUFF)

            # Remove out of bound vehicles
            TIME_BUFF = self.sim.remove_out_of_bound_vehicles(TIME_BUFF, dataset=self.dataset)

            # Generate entering vehicles at source points.
            TIME_BUFF = self.traffic_generator.generate_veh_at_source_pos(TIME_BUFF)  # Generate entering vehicles at source points.
            traj_pool = self.sim.time_buff_to_traj_pool(TIME_BUFF)
            self.one_sim_TIME_BUFF += TIME_BUFF[-self.rolling_step:]

            # Update basic stats of this current simulation episode.
            self.update_basic_stats_of_the_current_sim_episode(tt, TIME_BUFF, pred_vid)

            # Collision check. If collision, save out crash video and trajectory data if the flag is True.
            self.one_sim_colli_flag = self.sim.collision_check(self.one_sim_TIME_BUFF_newly_generated)
            if self.one_sim_colli_flag:
                # Visualize frames
                self.visualize_time_buff(TIME_BUFF, tt=tt)

                if self.save_collision_data_flag:
                    save_steps = 2
                    save_TIME_BUFF = self.one_sim_TIME_BUFF[-save_steps:]
                    collision_data_path = os.path.join(self.realistic_metric_save_folder, 'crash_frame')
                    self.save_trajectory(save_TIME_BUFF, save_path=collision_data_path, sim_id=sim_idx, step_id=0)

                break

        self.one_sim_wall_time = (tt + 1) * self.sim_resol * self.rolling_step

    def run_sim_steps_for_certain_TIME_BUFF(self, time_buff, sim_num, result_dir, num_idx, poc_dir, T=5): 
        #First record the states of TIME_BUFF, then run multiple one_step simulation to get result matrix and PoC
        # car_num_ineq = True
        # while car_num_ineq: 
        #     TIME_BUFF, traj_pool = self.initialize_sim(initial_TIME_BUFF=initial_TIME_BUFF) #TIME_BUFF: [history_length, N]
        #     tao = len(TIME_BUFF)
        #     car_nums_per_t = np.array([len(TIME_BUFF[i]) for i in range(tao)])
        #     if np.all(car_nums_per_t == car_nums_per_t[0]):
        #         if car_nums_per_t[0] >= 3:
        #             car_num_ineq = False

        tao = len(time_buff) #mention that tao = 5
        vid_all = []
        for i in range(tao):
            for j in range(len(time_buff[i])):
                vid = int(time_buff[i][j].id)
                if vid not in vid_all:
                    vid_all.append(int(time_buff[i][j].id))
        vid_all = sorted(vid_all)
        N = len(vid_all)

        # for t in range(tao):
        #     print(f"Time_step:{t}")
        #     car_ids = np.array([TIME_BUFF[t][i].id for i in range(N)])
        #     print(car_ids)

        Trajectory_info = {}
        Trajectory_info["scenario_id"] = "42735108"
        Trajectory_info["city"] = 1
        Trajectory_info["map_id"] = "rounD"
        
        #record the info of TIME_BUFF 
        agent = {}
        agent["num_nodes"] = N
        agent["av_index"] = 0
        agent["predict_mask"] = np.ones((N, tao))
        agent["predict_mask"] = torch.tensor(agent["predict_mask"], dtype=torch.bool)
        agent["id"] = np.arange(N)
        agent["type"] = torch.zeros(N, dtype=torch.uint8)
        agent["category"] = torch.zeros(N, dtype=torch.uint8)
        agent["shape"] = torch.tensor([3.6, 1.8, 0], dtype=torch.float32)

        agent["position"] = -np.ones((N, tao, 3))
        agent["heading"] = -np.ones((N, tao))
        agent["velocity"] = -np.ones((N, tao, 3))
        agent["valid_mask"] = np.zeros((N, tao))

        #wrong code
        #for t in range(tao):
        #    for i in range(N):
        #        v = TIME_BUFF[t][i]
        #        agent["position"][i, t, :] = np.array([v.location.x, v.location.y, v.location.z])
        #        agent["heading"][i, t] = v.speed_heading
        #        if t > 0:
        #            agent["velocity"][i, t] = (np.sqrt((TIME_BUFF[t][i].location.x - TIME_BUFF[t-1][i].location.x)**2 + (TIME_BUFF[t][i].location.y - TIME_BUFF[t-1][i].location.y)**2))/0.4
        #        else:
        #            agent["velocity"][i, t] = (np.sqrt((TIME_BUFF[t][i].location.x - TIME_BUFF[t+1][i].location.x)**2 + (TIME_BUFF[t][i].location.y - TIME_BUFF[t+1][i].location.y)**2))/0.4


        for t in range(tao):
            cars_t = time_buff[t]
            for car in cars_t:
                car_id = int(car.id)
                j = vid_all.index(car_id)
                agent["position"][j, t, :] = np.array([car.location.x, car.location.y, 0])
                agent["valid_mask"][j, t] = 1
                if car.speed_heading <= 180:
                    agent["heading"][j, t] = car.speed_heading * np.pi / 180
                else:
                    agent["heading"][j, t] = (car.speed_heading - 360) * np.pi / 180

        for t in range(tao):
            cars_t = time_buff[t]
            for car in cars_t:
                car_id = int(car.id)
                j = vid_all.index(car_id)
                if t == 0:
                    if agent["valid_mask"][j, t + 1] == 1:
                        vel = np.sqrt((agent["position"][j, t, 0] - agent["position"][j, t + 1, 0])**2 + (agent["position"][j, t, 1] - agent["position"][j, t + 1, 1])**2)/0.4
                        agent["velocity"][j, t, 0] = np.abs(np.cos(agent["heading"][j, t] * np.pi / 180) * vel)
                        agent["velocity"][j, t, 1] = np.abs(np.sin(agent["heading"][j, t] * np.pi / 180) * vel)
                        agent["velocity"][j, t, 2] = 0
                else:
                    if agent["valid_mask"][j, t - 1] == 1:
                        vel = np.sqrt((agent["position"][j, t, 0] - agent["position"][j, t - 1, 0])**2 + (agent["position"][j, t, 1] - agent["position"][j, t - 1, 1])**2)/0.4
                        agent["velocity"][j, t, 0] = np.abs(np.cos(agent["heading"][j, t] * np.pi / 180) * vel)
                        agent["velocity"][j, t, 1] = np.abs(np.sin(agent["heading"][j, t] * np.pi / 180) * vel)
                        agent["velocity"][j, t, 2] = 0

        agent["valid_mask"] = torch.tensor(agent["valid_mask"], dtype=torch.bool)
        agent["position"] = torch.tensor(agent["position"], dtype=torch.float32)
        agent["heading"] = torch.tensor(agent["heading"], dtype=torch.float32)
        agent["velocity"] = torch.tensor(agent["velocity"], dtype=torch.float32)
        Trajectory_info["agent"] = agent
        
        #run "sim_num" number of i.d. simulations of the TIME_BUFF 
        each_time_num = int(sim_num / T)
        future_states = -np.ones((sim_num, N, 3))
        PoC_T = np.zeros((N, N))

        
        #Attention!!! Delete the cars that did not show up in the last time step!!!
        time_buff_copy = copy.deepcopy(time_buff)
        car_id_in_last_step = [int(time_buff_copy[-1][ci].id) for ci in range(len(time_buff_copy[-1]))]
        
        time_buff_input = []
        for timeb_t in time_buff_copy:
            timb_single = []
            for car in timeb_t:
                if int(car.id) in car_id_in_last_step:
                    timb_single.append(car)

            time_buff_input.append(timb_single)

        traj_pool_input = self.sim.time_buff_to_traj_pool(time_buff_input)


        #simulate
        for i in range(sim_num):
            
            PoC_T_tmp = np.zeros((N, N))
            if_coll_in_first_step = False
            TIME_BUFF_new, _, _, _, _, _ = self.run_one_sim_step(traj_pool=traj_pool_input, TIME_BUFF=time_buff_input)
            TIME_BUFF_new = self.sim.remove_out_of_bound_vehicles(TIME_BUFF_new, dataset=self.dataset)

            for car in TIME_BUFF_new[-1]:
                j = vid_all.index(int(car.id))
                future_states[i, j, 0] = car.location.x
                future_states[i, j, 1] = car.location.y
                future_states[i, j, 2] = 0

            #detect if crash for every pair
            crash_num = 0
            for car_pairs in combinations(TIME_BUFF_new[-1], r=2):
                car_1, car_2 = car_pairs[0], car_pairs[1]
                idx = vid_all.index(int(car_1.id))
                jdx = vid_all.index(int(car_2.id))
                if idx != jdx:
                    v1_poly = car_1.poly_box
                    v2_poly = car_2.poly_box
                    if v1_poly.intersects(v2_poly):
                        PoC_T_tmp[idx, jdx] = 1
                        PoC_T_tmp[jdx, idx] = 1
                        crash_num += 1
                        if_coll_in_first_step = True
                        

            # if crash_num > 0:
            #     print(0)
            #     with open(result_dir + f"{int(num_idx[0])}_{int(num_idx[1])}_{i}.pkl", "wb") as f:
            #         pickle.dump(TIME_BUFF_new, f)
                


            if not if_coll_in_first_step:
                traj_pool_new = self.sim.time_buff_to_traj_pool(TIME_BUFF_new)
                crash_event_num2 = 0
                for roll_i in range(4):
                    TIME_BUFF_new, _, _, _, _, _ = self.run_one_sim_step(traj_pool=traj_pool_new, TIME_BUFF=TIME_BUFF_new)
                    TIME_BUFF_new = self.sim.remove_out_of_bound_vehicles(TIME_BUFF_new, dataset=self.dataset)
                    traj_pool_new = self.sim.time_buff_to_traj_pool(TIME_BUFF_new)
                    for car_pairs in combinations(TIME_BUFF_new[-1], r=2):
                        car_1, car_2 = car_pairs[0], car_pairs[1]
                        idx = vid_all.index(int(car_1.id))
                        jdx = vid_all.index(int(car_2.id))
                        if idx != jdx:
                            v1_poly = car_1.poly_box
                            v2_poly = car_2.poly_box
                            if v1_poly.intersects(v2_poly):
                                PoC_T_tmp[idx, jdx] = 1
                                PoC_T_tmp[jdx, idx] = 1
                                crash_event_num2 += 1

                    if crash_event_num2 > 0:
                        # print(roll_i)
                        # with open(result_dir + f"{int(num_idx[0])}_{int(num_idx[1])}_{i}.pkl", "wb") as f:
                        #     pickle.dump(TIME_BUFF_new, f)
                        break 

            
            PoC_T = PoC_T + PoC_T_tmp

        # print(f"max:{np.max(PoC_T)}, min:{np.min(PoC_T)}")
        # print(f"car_num:{N}, N**2={N**2}")
        # print(f"no_conflict_num:{np.sum(PoC_T == 0)}")
        # print(f"no_conflict_rate:{(np.sum(PoC_T == 0) - N)/(N**2-N)}")]

        safety_flag = np.zeros((agent["num_nodes"], agent["num_nodes"]))
        for i in range(N):
            for j in range(N):
                if i == j or PoC_T[i, j] == 0:
                    safety_flag[i, j] = 0
                    safety_flag[j, i] = 0
                elif PoC_T[i, j] >= sim_num:
                    safety_flag[i, j] = 1
                    safety_flag[j, i] = 1
                else:
                    safety_flag[i, j] = 2
                    safety_flag[j, i] = 2
                
        Trajectory_info["safety_flag"] = [[torch.tensor(flag, dtype=torch.float32) for flag in line]for line in safety_flag]

        PoC_T = PoC_T / sim_num
        #save poc
        df_poc = pd.DataFrame(PoC_T)
        df_poc.to_csv(poc_dir + f"{int(num_idx[0])}_{int(num_idx[1])}.csv", index=False)

        Trajectory_info["future_states"] = torch.tensor(future_states, dtype=torch.float32)
        Trajectory_info["PoC_T"] = torch.tensor(PoC_T, dtype=torch.float32)

        #record all the info above
        with open(result_dir + f"{int(num_idx[0])}_{int(num_idx[1])}.pkl", "wb") as f:
            pickle.dump(Trajectory_info, f)
    
    def check_crash_samples(self, max_time, result_dir, num_idx, if_all, initial_TIME_BUFF=None, save_not_coll=False):
        #sample from history traj, then simulate for a long time until get a crash.
        #allow new cars in, allow out of bound cars. Remember the time_idx for new cars and valid_mask.
        #if find proper samples, save the whole traj.
        TIME_BUFF, traj_pool = self.initialize_sim(initial_TIME_BUFF=initial_TIME_BUFF)
        TIME_BUFF_new = copy.deepcopy(TIME_BUFF)
        traj_pool_new = copy.deepcopy(traj_pool)
        # std_x_all = []
        # std_y_all = []
        #corr_all = []
        for i in range(max_time):
            TIME_BUFF_new, pred_vid, output_delta_position_mask = self.run_one_sim_step(traj_pool=traj_pool_new, TIME_BUFF=TIME_BUFF_new)

            TIME_BUFF_new = self.sim.remove_out_of_bound_vehicles(TIME_BUFF_new, self.dataset)
            TIME_BUFF_new = self.traffic_generator.generate_veh_at_source_pos(TIME_BUFF_new)  # Generate entering vehicles at source points.
            traj_pool_new = self.sim.time_buff_to_traj_pool(TIME_BUFF_new)
            self.one_sim_TIME_BUFF += TIME_BUFF_new[-self.rolling_step:]
            self.update_basic_stats_of_the_current_sim_episode(i, TIME_BUFF_new, pred_vid)
            self.one_sim_colli_flag, crash_pair = self.sim.collision_check(self.one_sim_TIME_BUFF_newly_generated)

            # std_x_all.append(std_x)
            # std_y_all.append(std_y)
            #corr_all.append(pred_corr)

            if self.one_sim_colli_flag:
                #infos["inference_step"] = i + 1
                #infos["inital_state"] = TIME_BUFF
                #infos["whole_inference_states"] = self.one_sim_TIME_BUFF
                #save the time_buffs which contain crash pairs
                #print(crash_pair)
                # time_buff_t = self.one_sim_TIME_BUFF[-1]
                # print([int(time_buff_t[p].id) for p in range(len(time_buff_t))])
                # print([int(self.one_sim_TIME_BUFF_newly_generated[-1][p].id) for p in range(len(time_buff_t))])
                states_to_be_considered = []
                final_idx = 0
                for k in range(len(self.one_sim_TIME_BUFF)):
                    time_buff_t = self.one_sim_TIME_BUFF[-1-k]
                    vids_t = [int(time_buff_t[p].id) for p in range(len(time_buff_t))]
                    # print(vids_t)
                    if crash_pair[0] in vids_t and crash_pair[1] in vids_t:
                        states_to_be_considered.append(time_buff_t)
                        final_idx = -1-k
                #print(len(states_to_be_considered))

                if len(states_to_be_considered) >= 20:

                    infos = {}
                    if if_all:
                        infos["states_all"] = self.one_sim_TIME_BUFF
                    else:

                        #infos["whole_inference_states"] = self.one_sim_TIME_BUFF
                        infos["states_considered"] = states_to_be_considered[::-1]
                        idx_before_start = np.max(len(self.one_sim_TIME_BUFF) + final_idx - 5, 0)
                        infos["states_before"] = self.one_sim_TIME_BUFF[idx_before_start:final_idx] # to check all the states considered for 3-sigma
                        infos['crash_step'] = i + 1
                        # infos["std_x"] = std_x_all
                        # infos["std_y"] = std_y_all
                        #infos["corr"] = corr_all

                    with open(result_dir + f"{num_idx}.pkl", "wb") as f:
                        pickle.dump(infos, f)

                    return 1
                else:
                    if save_not_coll:
                        infos = {}
                        infos["states_considered"] = self.one_sim_TIME_BUFF

                        with open(result_dir + f"{num_idx}_short.pkl", "wb") as f:
                            pickle.dump(infos, f)
                    return 0        

        if save_not_coll:
            infos = {}
            infos["states_considered"] = self.one_sim_TIME_BUFF

            with open(result_dir + f"{num_idx}_no_coll.pkl", "wb") as f:
                pickle.dump(infos, f)

        return 0

    def check_crash_distribute(self, max_time, result_dir, num_idx, num_try, num_steps, if_all, initial_TIME_BUFF=None, save_not_coll=False):
        #sample from history traj, then simulate for a long time until get a crash.
        #allow new cars in, allow out of bound cars. Remember the time_idx for new cars and valid_mask.
        #if find proper samples, save the whole traj.
        TIME_BUFF, traj_pool = self.initialize_sim(initial_TIME_BUFF=initial_TIME_BUFF[-num_steps-4:-num_steps+1])
        TIME_BUFF_new = copy.deepcopy(TIME_BUFF)
        traj_pool_new = copy.deepcopy(traj_pool)
        _, crash_pair_1 = self.sim.collision_check(initial_TIME_BUFF[-1:])

        # print(crash_pair_1)


        for i in range(max_time):
            TIME_BUFF_new, pred_vid, output_delta_position_mask = self.run_one_sim_step(traj_pool=traj_pool_new, TIME_BUFF=TIME_BUFF_new)

            TIME_BUFF_new = self.sim.remove_out_of_bound_vehicles(TIME_BUFF_new, self.dataset)
            # TIME_BUFF_new = self.traffic_generator.generate_veh_at_source_pos(TIME_BUFF_new)  # Generate entering vehicles at source points.
            traj_pool_new = self.sim.time_buff_to_traj_pool(TIME_BUFF_new)
            self.one_sim_TIME_BUFF += TIME_BUFF_new[-self.rolling_step:]
            self.update_basic_stats_of_the_current_sim_episode(i, TIME_BUFF_new, pred_vid)
            self.one_sim_colli_flag, crash_pair_2 = self.sim.collision_check(self.one_sim_TIME_BUFF_newly_generated)


            if self.one_sim_colli_flag:
                #print('find crash:',crash_pair_2)
                infos = {}
                if if_all:
                    infos["states_all"] = self.one_sim_TIME_BUFF
                    infos['orignal crash'] = crash_pair_1
                    infos['simulate crash'] = crash_pair_2
                with open(result_dir + f"{num_idx}_{num_try}.pkl", "wb") as f:
                    pickle.dump(infos, f)

                return 1
        

        infos = {}
        infos["states_all"] = self.one_sim_TIME_BUFF
        with open(result_dir + f"{num_idx}_{num_try}.pkl", "wb") as f:
            pickle.dump(infos, f)

        
        return 0
                    

    def check_if_wrong_traj(self, scene_data):
        scene_tb_length = len(scene_data)

        vid_all = []
        for i in range(scene_tb_length):
            for j in range(len(scene_data[i])):
                car_id = int(scene_data[i][j].id)
                if car_id not in vid_all:
                    vid_all.append(car_id)
        vid_all = sorted(vid_all)

        all_cars_posi = -np.ones((len(vid_all), scene_tb_length, 2))
        for i in range(scene_tb_length):
            for car in scene_data[i]:
                car_id = int(car.id)
                j = vid_all.index(car_id)
                all_cars_posi[j, i, :] = np.array([car.location.x, car.location.y])

        for i in range(len(vid_all)):
            position = all_cars_posi[i]
            heading = []
            for j in range(1, len(position)):
                if position[j][0] == -1 or position[j-1][0] == -1:
                    continue
                #heading.append([np.arctan2(position[j][1] - position[j - 1][1], position[j][0] - position[j - 1][0]), j])
                dis_car = np.sqrt((position[j][1] - position[j - 1][1])**2 + (position[j][0] - position[j - 1][0])**2)
                heading.append([np.arctan2(position[j][1] - position[j - 1][1], position[j][0] - position[j - 1][0]), j, dis_car])
                if dis_car >= 12:
                    return True
            
            if len(heading) < 2:
                continue
            for j in range(len(heading) - 1):
                if abs(heading[j][0] - heading[j + 1][0]) < np.pi / 4 or abs(2 * np.pi - abs(heading[j][0] - heading[j + 1][0])) < np.pi / 4:
                    continue
                if heading[j][2] < 0.4 or heading[j+1][2] < 0.4:
                    continue
                
                return True

        return False

    
    def truncated_exp_sample(self, lam=2, low=1, high=4):
        while True:
            x = np.random.exponential(scale=1/lam)
            if x > low and x < high:
                return round(x)

    def generate_prob_ignore_results(self, save_path_all, save_path_single, save_idx, max_time=400, ignore_rate=0.9, lambda_exp = 2, initial_TIME_BUFF=None):
        TIME_BUFF, traj_pool = self.initialize_sim(initial_TIME_BUFF=initial_TIME_BUFF)
        
        car_id_by_time = []
        for cars in TIME_BUFF:
            car_id_t = []
            for car in cars:
                car_id_t.append(car.id)
            car_id_by_time.append(car_id_t)

        while True:
            cars_to_ignore = []
            for car_id in car_id_by_time[-1]:
                p_ignore = np.random.binomial(n=1, p=ignore_rate)
                if p_ignore:
                    cars_to_ignore.append(car_id)

            if len(cars_to_ignore) > 0.7 * len(car_id_by_time[-1]):
                break

        # 对于每一个被选中的车，它们都看不见任何其他车
        # 没有被选中的车，可以看到其他所有车

        # 对每辆选中的车生成盲区时间，范围是[1,5]

        cars_ignore_dict = {}

        for car_id in cars_to_ignore:
            cars_ignore_dict[car_id] = self.truncated_exp_sample(lam=lambda_exp)

        # time_to_ignore = []
        # for i in range(len(cars_to_ignore)):
        #     car_id = cars_to_ignore[i]
        #     t_ignore = self.truncated_exp_sample(lam=lambda_exp)
        #     time_to_ignore.append(t_ignore)


        TIME_BUFF_all = copy.deepcopy(TIME_BUFF)

        for t in range(max_time):
            # 生成正常的下一步
            TIME_BUFF_new, _, _ = self.run_one_sim_step(traj_pool=traj_pool, TIME_BUFF=TIME_BUFF)

            # 为每辆被选中的车生成下一步，并替换正常下一步中对应车的位置
            # for car in TIME_BUFF_new[-1]:
            #     print(car.id, car.location.x, car.location.y)
            # print('--------')
            # print(cars_to_ignore)
            # print('--------')

            ignored_next_step = []

            for car_id in cars_ignore_dict:
                if car_id not in car_id_by_time[-1]:
                    continue

                t_ignore = cars_ignore_dict[car_id]

                TIME_BUFF_changed = []
                for i in range(len(TIME_BUFF)):
                    if (i + t) < t_ignore:
                        TIME_BUFF_changed.append(TIME_BUFF[i])
                    else:
                        for car in TIME_BUFF[i]:
                            if car.id == car_id:
                                TIME_BUFF_changed.append([car])
                                break

                traj_pool_changed = self.sim.time_buff_to_traj_pool(TIME_BUFF_changed)
                TIME_BUFF_changed_new, _, _, = self.run_one_sim_step(traj_pool=traj_pool_changed, TIME_BUFF=TIME_BUFF_changed)

                for car in TIME_BUFF_changed_new[-1]:
                    if car.id == car_id:
                        ignored_next_step.append(car)
                        break

            TIME_BUFF_last = []

            ignore_id_list = [key for key in cars_ignore_dict]
            for car in TIME_BUFF_new[-1]:
                if car.id not in ignore_id_list:
                    TIME_BUFF_last.append(car)

            TIME_BUFF_last += ignored_next_step
            
            TIME_BUFF = TIME_BUFF[1:]
            TIME_BUFF.append(TIME_BUFF_last)

            # for car in TIME_BUFF[-1]:
            #     print(car.id, car.location.x, car.location.y)
            # print('-----------')
            TIME_BUFF = self.sim.remove_out_of_bound_vehicles(TIME_BUFF, dataset=self.dataset)
            TIME_BUFF = self.traffic_generator.generate_veh_at_source_pos(TIME_BUFF)
            traj_pool = self.sim.time_buff_to_traj_pool(TIME_BUFF)


            TIME_BUFF_all.append(TIME_BUFF[-1])

            car_id_last = [car.id for car in TIME_BUFF_all[-1]]
            car_id_by_time.append(car_id_last)

            cars_new = []
            for car_id in car_id_by_time[-1]:
                if car_id not in car_id_by_time[-2]:
                    cars_new.append(car_id)

            if len(cars_new) > 0:
                while True:
                    cars_new_to_ignore = []
                    for car_id in cars_new:
                        p_ignore = np.random.binomial(n=1, p=ignore_rate)
                        if p_ignore:
                            cars_new_to_ignore.append(car_id)

                    if len(cars_new_to_ignore) > 0.5 * len(cars_new):
                        break
                    
                for car_id in cars_new_to_ignore:
                    cars_ignore_dict[car_id] = self.truncated_exp_sample(lam=lambda_exp) + (t + 4)
            

            coll_flag, _ = self.sim.collision_check(TIME_BUFF_all[-1:])

            if coll_flag:
                break
            # for car in TIME_BUFF[-1]:
            #     print(car.id, car.location.x, car.location.y)

        if_wrong_traj = self.check_if_wrong_traj(TIME_BUFF_all)

        if not if_wrong_traj:
            with open(os.path.join(save_path_all, f'{save_idx}.pkl'), 'wb') as fb:
                pickle.dump(TIME_BUFF_all, fb)

            for i in range(len(TIME_BUFF_all) - 6):
                with open(os.path.join(save_path_single, f'{save_idx}_{i}.pkl'), 'wb') as fb2:
                    pickle.dump(TIME_BUFF_all[i:i+6], fb2)

            return 1

        else:
            return 0

    def truncated_exp_sample_new(self, lam=1, low=5, high=20):
        while True:
            x = np.random.exponential(scale=1/lam)
            if x > low and x < high:
                return round(x)


    def generate_prob_ignore_results_new(self, save_path_all, save_path_single, save_idx, max_time=500, ignore_rate=0.8, lambda_exp = 0.1, initial_TIME_BUFF=None):
        TIME_BUFF, traj_pool = self.initialize_sim(initial_TIME_BUFF=initial_TIME_BUFF)
        
        cars_ignore_dict = {}

        for car in TIME_BUFF[-1]:
            car_id = car.id
            p_ignore = np.random.binomial(n=1, p=ignore_rate)
            if p_ignore:
                cars_ignore_dict[car_id] = [self.truncated_exp_sample_new(lam=lambda_exp), 0]


        TIME_BUFF_all = copy.deepcopy(TIME_BUFF)

        for t in range(max_time):
            # 生成正常的下一步
            TIME_BUFF_new, _, _ = self.run_one_sim_step(traj_pool=traj_pool, TIME_BUFF=TIME_BUFF)

            # 为每辆被选中的车生成下一步，并替换正常下一步中对应车的位置
            # for car in TIME_BUFF_new[-1]:
            #     print(car.id, car.location.x, car.location.y)
            # print('--------')
            # print(cars_to_ignore)
            # print('--------')

            ignored_next_step = []
            car_id_by_time = [car.id for car in TIME_BUFF[-1]]

            for car_id in cars_ignore_dict:
                if car_id not in car_id_by_time:
                    continue

                t_ignore = cars_ignore_dict[car_id][0]
                t_ignore_start = cars_ignore_dict[car_id][1]

                TIME_BUFF_changed = []
                for i in range(len(TIME_BUFF)):
                    if (i + t) < t_ignore:
                        TIME_BUFF_changed.append(TIME_BUFF[i])
                    else:
                        for car in TIME_BUFF[i]:
                            if car.id == car_id:
                                TIME_BUFF_changed.append([car])
                                break

                traj_pool_changed = self.sim.time_buff_to_traj_pool(TIME_BUFF_changed)
                TIME_BUFF_changed_new, _, _, = self.run_one_sim_step(traj_pool=traj_pool_changed, TIME_BUFF=TIME_BUFF_changed)

                for car in TIME_BUFF_changed_new[-1]:
                    if car.id == car_id:
                        ignored_next_step.append(car)
                        break

            TIME_BUFF_last = []

            ignore_id_list = [key for key in cars_ignore_dict]
            for car in TIME_BUFF_new[-1]:
                if car.id not in ignore_id_list:
                    TIME_BUFF_last.append(car)

            TIME_BUFF_last += ignored_next_step
            
            TIME_BUFF = TIME_BUFF[1:]
            TIME_BUFF.append(TIME_BUFF_last)

            # for car in TIME_BUFF[-1]:
            #     print(car.id, car.location.x, car.location.y)
            # print('-----------')
            TIME_BUFF = self.sim.remove_out_of_bound_vehicles(TIME_BUFF, dataset=self.dataset)
            TIME_BUFF = self.traffic_generator.generate_veh_at_source_pos(TIME_BUFF)
            traj_pool = self.sim.time_buff_to_traj_pool(TIME_BUFF)


            TIME_BUFF_all.append(TIME_BUFF[-1])

            car_id_last = [car.id for car in TIME_BUFF_all[-1]]
            car_id_by_time.append(car_id_last)

            for car_id in car_id_by_time[-1]:
                if car_id not in car_id_by_time[-2]:
                    p_ignore = np.random.binomial(n=1, p=ignore_rate)
                    if p_ignore:
                        cars_ignore_dict[car_id] = self.truncated_exp_sample(lam=lambda_exp) + (t + 5)
            

            coll_flag, _ = self.sim.collision_check(TIME_BUFF_all[-1:])

            if coll_flag:
                break
            # for car in TIME_BUFF[-1]:
            #     print(car.id, car.location.x, car.location.y)

        if_wrong_traj = self.check_if_wrong_traj(TIME_BUFF_all)

        if not if_wrong_traj:
            with open(os.path.join(save_path_all, f'{save_idx}.pkl'), 'wb') as fb:
                pickle.dump(TIME_BUFF_all, fb)

            for i in range(len(TIME_BUFF_all) - 6):
                with open(os.path.join(save_path_single, f'{save_idx}_{i}.pkl'), 'wb') as fb2:
                    pickle.dump(TIME_BUFF_all[i:i+6], fb2)

            return 1

        else:
            return 0



    def check_one_sample(self, TIME_BUFF):
        TIME_BUFF = self.sim.remove_out_of_bound_vehicles(TIME_BUFF, dataset=self.dataset)   
        #Attention!!! Delete the cars that did not show up in the last time step!!!
        time_buff_copy = copy.deepcopy(TIME_BUFF)
        car_id_in_last_step = [int(time_buff_copy[-1][ci].id) for ci in range(len(time_buff_copy[-1]))]
        
        time_buff_input = []
        for timeb_t in time_buff_copy:
            timb_single = []
            for car in timeb_t:
                if int(car.id) in car_id_in_last_step:
                    timb_single.append(car)

            time_buff_input.append(timb_single)

        traj_pool_input = self.sim.time_buff_to_traj_pool(time_buff_input)

        _, _, pred_cos_heading, pred_sin_heading, pred_vid, buff_vid, current_lat, current_lon, pred_lat_mean, pred_lon_mean, pred_lat_std, pred_lon_std = self.sim.run_forwardpass(traj_pool_input)
        output_delta_position_mask = np.zeros(buff_vid.shape, dtype=bool)
        # print(pred_lat.shape) (32,1)
        # print(pred_lat_mean.shape)
        # print(pred_lat_std.shape)
        pred_lat = pred_lat_mean - 3 * pred_lat_std
        pred_lon = pred_lon_mean - 3 * pred_lon_std
        TIME_BUFF_zs = self.sim.prediction_to_trajectory_rolling_horizon(pred_lat, pred_lon, pred_cos_heading, pred_sin_heading, pred_vid, TIME_BUFF, rolling_step=self.rolling_step)

        # pred_lat = pred_lat_mean - 3 * pred_lat_std
        # pred_lon = pred_lon_mean + 3 * pred_lon_std
        # TIME_BUFF_zx = self.sim.prediction_to_trajectory_rolling_horizon(pred_lat, pred_lon, pred_cos_heading, pred_sin_heading, pred_vid, TIME_BUFF, rolling_step=self.rolling_step)

        # pred_lat = pred_lat_mean + 3 * pred_lat_std
        # pred_lon = pred_lon_mean - 3 * pred_lon_std
        # TIME_BUFF_ys = self.sim.prediction_to_trajectory_rolling_horizon(pred_lat, pred_lon, pred_cos_heading, pred_sin_heading, pred_vid, TIME_BUFF, rolling_step=self.rolling_step)

        pred_lat = pred_lat_mean + 3 * pred_lat_std
        pred_lon = pred_lon_mean + 3 * pred_lon_std
        TIME_BUFF_yx = self.sim.prediction_to_trajectory_rolling_horizon(pred_lat, pred_lon, pred_cos_heading, pred_sin_heading, pred_vid, TIME_BUFF, rolling_step=self.rolling_step)

        return TIME_BUFF_zs[-1], TIME_BUFF_yx[-1]



    def save_check_sample_result(self, time_buff, idx, save_path, with_traj):
        self.save_time_buff_video(TIME_BUFF=time_buff, background_map=self.background_map, file_name=idx, save_path=save_path, with_traj=with_traj)
        
    
    def _visualize_time_buff(self, TIME_BUFF, background_map):
        for i in range(len(TIME_BUFF)):
            vehicle_list = TIME_BUFF[i]
            vis = background_map.render(vehicle_list, with_traj=True, linewidth=6)
            img = vis[:, :, ::-1]
            # img = cv2.resize(img, (768, int(768 * background_map.h / background_map.w)))  # resize when needed
            cv2.imshow('vis', img)  # rgb-->bgr
            cv2.waitKey(1)

    def _save_vis_time_buff(self, TIME_BUFF, background_map, save_path):
        for i in range(len(TIME_BUFF)):
            vehicle_list = TIME_BUFF[i]
            vis = background_map.render(vehicle_list, with_traj=True, linewidth=6)
            img = vis[:, :, ::-1]
            # img = cv2.resize(img, (768, int(768 * background_map.h / background_map.w)))  # resize when needed
            cv2.imwrite(save_path, img)  # rgb-->bgr


    def save_time_buff_video(self, TIME_BUFF, background_map, file_name, save_path, color_vid_list=None, with_traj=True):
        if self.interpolate_flag:
            visualize_TIME_BUFF, _ = self.traj_interpolator.interpolate_traj(TIME_BUFF,
                                                                             intep_steps=self.intep_steps,
                                                                             original_resolution=self.sim_resol)
        else:
            visualize_TIME_BUFF = TIME_BUFF

        vid_all = []
        for i in range(len(visualize_TIME_BUFF)):
            for j in range(len(visualize_TIME_BUFF[i])):
                vid = int(visualize_TIME_BUFF[i][j].id)
                if vid not in vid_all:
                    vid_all.append(int(visualize_TIME_BUFF[i][j].id))
        vid_all = sorted(vid_all)


        os.makedirs(save_path, exist_ok=True)
        collision_video_writer = cv2.VideoWriter(save_path + r'/{0}.mp4'.format(file_name), cv2.VideoWriter_fourcc(*'mp4v'), self.save_fps, (background_map.w, background_map.h))
        for i in range(len(visualize_TIME_BUFF)):
            vehicle_list = visualize_TIME_BUFF[i]
            vis = background_map.render(vehicle_list=vehicle_list, id_list=vid_all, with_traj=with_traj, linewidth=5, color_vid_list=color_vid_list)
            img = vis[:, :, ::-1]
            # img = cv2.resize(img, (768, int(768 * background_map.h / background_map.w)))  # resize when needed
            collision_video_writer.write(img)

    def save_trajectory(self, TIME_BUFF, save_path, sim_id, step_id):
        os.makedirs(save_path, exist_ok=True)
        for i in range(len(TIME_BUFF)):
            vehicle_list = TIME_BUFF[i]
            frame_id = step_id + i
            output_file_path = os.path.join(save_path, str(sim_id) + '-' + str(frame_id).zfill(5) + '.pickle')
            pickle.dump(vehicle_list, open(output_file_path, "wb"))
