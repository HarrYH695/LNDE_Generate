import os
import numpy as np
import pickle

# file_vis = "/nfs/turbo/coe-mcity/hanhy/LNDE_Results/AA_Trial_1/2_vis/"
# files = os.listdir(file_vis)
# print(len(files))

# file_save = "/nfs/turbo/coe-mcity/hanhy/LNDE_Results/AA_Trial_1/4_check_hascarinfo/"
# processed_files = os.listdir(file_save)

# for file in processed_files:
#     file = "27.pkl"
#     data = pickle.load(open(file_save+file, "rb"))
#     len_tb = data["tb_len"]
#     distance_info = data["distance_info"] #(car_num, time, 4)

#     car_dis_simu = distance_info[:, :, :2]

#     car_num = 6
#     disall = []
#     for i in range(6,11):
#         dis = np.linalg.norm(car_dis_simu[car_num, i + 1, :] - car_dis_simu[car_num, i, :])
#         disall.append(dis)

#     print(disall)
#     break


def check_crash_samples(self, max_time, save_dir, save_name, initial_TIME_BUFF=None):
        TIME_BUFF, traj_pool = self.initialize_sim(initial_TIME_BUFF=initial_TIME_BUFF)
        TIME_BUFF_new = copy.deepcopy(TIME_BUFF)
        traj_pool_new = copy.deepcopy(traj_pool)
        for i in range(max_time):
            TIME_BUFF_new, pred_vid, output_delta_position_mask = self.run_one_sim_step(traj_pool=traj_pool_new, TIME_BUFF=TIME_BUFF_new)
            TIME_BUFF_new = self.sim.remove_out_of_bound_vehicles(TIME_BUFF_new, self.dataset)
            #TIME_BUFF_new = self.traffic_generator.generate_veh_at_source_pos(TIME_BUFF_new)  #gen new cars
            traj_pool_new = self.sim.time_buff_to_traj_pool(TIME_BUFF_new)
            self.one_sim_TIME_BUFF += TIME_BUFF_new[-self.rolling_step:]
            self.update_basic_stats_of_the_current_sim_episode(i, TIME_BUFF_new, pred_vid)
            self.one_sim_colli_flag = self.sim.collision_check(self.one_sim_TIME_BUFF_newly_generated)

            if self.one_sim_colli_flag:
                break
                
            with open(save_dir+save_name, "wb") as fr:
                pickle.dump(self.one_sim_TIME_BUFF, fr)