import torch
import torch, torch.nn.functional as F
import argparse
import os
import yaml
import warnings
import shutil
from tqdm import tqdm
from simulation_modeling.simulation_inference import SimulationInference
import pickle
import numpy as np
from geo_engine import GeoEngine
import cv2


# settings
parser = argparse.ArgumentParser()
parser.add_argument('--folder-idx', type=int, default='1', metavar='N',
                    help='Worker id of the running experiment')
parser.add_argument('--experiment-name', type=str, required=True,
                    help='The name of the experiment folder where the data will be stored')
parser.add_argument('--save-result-path', type=str, default=r'./results/inference/',
                    help='The path to save the simulation results, a folder with experiment_name will be created in the path')
parser.add_argument('--config', type=str, required=True,
                    help='The path to the simulation config file. E.g., ./configs/AA_rdbt_inference.yml')
parser.add_argument('--viz-flag', action='store_true', help='Default is False, adding this argument will overwrite the same flag in config file')

args = parser.parse_args()

class GeoMapping(GeoEngine):
    def __init__(self, map_file_dir, map_height=1024, map_width=1024):
        super(GeoMapping, self).__init__(map_file_dir, map_height=map_height, map_width=map_width)

        basemap = cv2.imread(map_file_dir, cv2.IMREAD_COLOR)
        self.basemap = cv2.cvtColor(basemap, cv2.COLOR_BGR2RGB)
        self.basemap = cv2.resize(self.basemap, (map_width, map_height))
        self.basemap = (self.basemap.astype(np.float64) * 0.6).astype(np.uint8)

        self.traj_alpha = np.zeros([self.h, self.w, 3], dtype=np.float32)

        self.get_pixel_resolution()

    def posi_to_pixel(self, posi, output_int=True):
        pixel = self._world2pxl(posi, output_int=output_int)

        return pixel


    def get_pixel_resolution(self):
        dx_meter, dy_meter = self.f.tl[0] - self.f.tr[0], self.f.tl[1] - self.f.tr[1]
        d = np.linalg.norm([dx_meter, dy_meter])

        self.x_pixel_per_meter = self.w / d

        dx_meter, dy_meter = self.f.tl[0] - self.f.bl[0], self.f.tl[1] - self.f.bl[1]
        d = np.linalg.norm([dx_meter, dy_meter])

        self.y_pixel_per_meter = self.h / d

def check_if_wrong_traj(scene_data):
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
            if dis_car >= 10:
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

def cal_gmm_var(mu, std, corr, pi):
    sx, sy = std[:, 0], std[:, 1]

    Sigma_k              = np.zeros((3, 2, 2))
    Sigma_k[:, 0, 0]     = sx**2
    Sigma_k[:, 1, 1]     = sy**2
    Sigma_k[:, 0, 1]     = corr * sx * sy
    Sigma_k[:, 1, 0]     = corr * sx * sy

    outer_mu             = mu[:, :, None] * mu[:, None, :]
    first_term           = (pi[:, None, None] * (Sigma_k + outer_mu)).sum(axis=0)

    mu_mix               = (pi[:, None] * mu).sum(axis=0)       
    Sigma_mix            = first_term - np.outer(mu_mix, mu_mix)

    return Sigma_mix


if __name__ == '__main__':
    # Load config file
    with open(args.config) as file:
        try:
            configs = yaml.safe_load(file)
            print(f"Loading config file: {args.config}")
        except yaml.YAMLError as exception:
            print(exception)

    # Settings
    configs["device"] = torch.device("cuda:0" if configs["use_gpu"] else "cpu")
    print(f"Using device: {configs['device']}...")
    print(f"Simulating {configs['dataset']} using {configs['model']} model...")
    print('Using conflict critic module!') if configs["use_conflict_critic_module"] else print('Not using conflict critic module!') #True
    print(f'Using neural safety mapper!' if configs["use_neural_safety_mapping"] else 'Not using neural safety mapper!')
    assert (configs["rolling_step"] <= configs["pred_length"])
    configs["viz_flag"] = configs["viz_flag"] or args.viz_flag  # The visualization flag can be easily modified through input argument.

    # Saving results paths
    folder_idx = args.folder_idx  # The worker index, the simulation can be run in multiple cores (e.g., using job array on the Great Lakes HPC)
    experiment_name = args.experiment_name
    save_result_path = args.save_result_path
    configs["realistic_metric_save_folder"] = os.path.join(save_result_path, f'{configs["dataset"]}_inference/{experiment_name}/{str(configs["sim_wall_time"])}s/raw_data/{str(folder_idx)}')
    configs["simulated_TIME_BUFF_save_folder"] = os.path.join(save_result_path, f'{configs["dataset"]}_inference/{experiment_name}/{str(configs["sim_wall_time"])}s/TIME_BUFF/{str(folder_idx)}')
    configs["save_viz_folder"] = os.path.join(save_result_path, f'{configs["dataset"]}_inference/{experiment_name}/{str(configs["sim_wall_time"])}s/demo_video/{str(folder_idx)}')

    # Save the config file of this experiment
    os.makedirs(os.path.join(save_result_path, f'{configs["dataset"]}_inference/{experiment_name}/{str(configs["sim_wall_time"])}s'), exist_ok=True)
    save_path = os.path.join(save_result_path, f'{configs["dataset"]}_inference/{experiment_name}/{str(configs["sim_wall_time"])}s', "config.yml")
    shutil.copyfile(args.config, save_path)

    # Initialize the simulation inference model.
    #simulation_inference_model = SimulationInference(configs=configs)

    geomap = GeoMapping(map_file_dir=configs["basemap_dir"], map_height=configs["map_height"], map_width=configs["map_width"])


    dir_name = "rD_trial_2_4"
    file_ori = "/nfs/turbo/coe-mcity/hanhy/LNDE_inference_data/LNDE_ignore_0726_2/" + dir_name + "/1_4/" 

    #file_ori = '/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/LNDE_Results/rD_baseline/1/'


    scenes_all = os.listdir(file_ori)
    print(len(scenes_all))
    num_load = 0

    print(geomap.basemap.shape) # 963, 1678, 3
    h_map = geomap.basemap.shape[0]
    w_map = geomap.basemap.shape[1]

    heat_matrix = np.zeros((h_map, w_map))
    heat_matrix_2 = np.zeros((h_map, w_map))
    heat_matrix_3 = np.zeros((h_map, w_map))

    for scene in tqdm(scenes_all):
        scene_data_file = pickle.load(open(file_ori+scene, "rb"))
        if 'states_all' in scene_data_file:
            scene_data = scene_data_file["states_all"]
        else:
            scene_data = scene_data_file["states_considered"]

        if check_if_wrong_traj(scene_data):
            continue
        
        num_load += 1

        len_scene = len(scene_data)
        # print(len_scene)

        num_all = 0
        for i in range(len_scene):
            time_step = scene_data[i]
            for car in time_step:

                if not hasattr(car, 'std'):
                    continue
                
                car_x = car.location.x
                car_y = car.location.y

                if car_x is None:
                    continue

                num_all += 1

                std_car = car.std
                corr_car = car.corr
                pi_car = torch.tensor(car.pi)
                pi_car = F.softmax(pi_car, dim=-1).cpu().numpy()
                mean_car = car.mean_posi

                posi_at_map = geomap._world2pxl([car_x, car_y])
                # print(posi_at_map)

                sigma_matrix = cal_gmm_var(mean_car, std_car, corr_car, pi_car)

                var_x = sigma_matrix[0,0]
                var_y = sigma_matrix[1,1]

                heat_matrix[posi_at_map[1], posi_at_map[0]] += (var_x + var_y)
                # heat_matrix_2[posi_at_map[1], posi_at_map[0]] += np.sqrt(var_x + var_y)、
                heat_matrix_3[posi_at_map[1], posi_at_map[0]] += 1



    heat_matrix_all = {}
    heat_matrix_all['heat_matrix'] = heat_matrix
    heat_matrix_all['heat_matrix_2'] = heat_matrix_2
    heat_matrix_all['heat_matrix_3'] = heat_matrix_3
    heat_matrix_all['num_load'] = num_load
    heat_matrix_all['num_all'] = num_all

    with open('/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/heat_matrices_2.pkl', 'wb') as fb:
        pickle.dump(heat_matrix_all, fb)
    
    print(num_load)


    
# python draw_std_distribution.py --experiment-name vis_1 --folder-idx 4 --config ./configs/rounD_inference.yml