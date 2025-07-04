import torch
import argparse
import os
import yaml
import warnings
import shutil
from tqdm import tqdm
from simulation_modeling.simulation_inference import SimulationInference
import pickle
import numpy as np

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

    dir_name = "rD_baseline"
    file_ori = "/nfs/turbo/coe-mcity/hanhy/LNDE_Results/" + dir_name + "/1/" 
    file_t = "/nfs/turbo/coe-mcity/hanhy/LNDE_Results/" + dir_name + "/2/"
    file_save = "/nfs/turbo/coe-mcity/hanhy/LNDE_Results/" + dir_name + "/check/"

    if not os.path.exists(file_save):
        os.makedirs(file_save)

    scenes_all = os.listdir(file_ori)
    print(len(scenes_all)) #4427
    num_load = 0
    for scene in scenes_all:
        scene = "1599.pkl"
        test_id = 1
        scene_data_file = pickle.load(open(file_ori+scene, "rb"))
        scene_data = scene_data_file["states_considered"] #attention: the last buff is the crash buff and should not be considered
        #check if the time buff is out of 3 sigma or 2 sigma. 
        #then check if the time buff is "wrong". we want to know if there is out of 3 sigma or 2 sigma before the wrong data.abs
        #save the out of x-sigma and wrong time buff in terms of every scene.
        scene_info = {}
        scene_tb_length = len(scene_data) - 1
        scene_info["tb_len"] = scene_tb_length
        
        #Check out the wrong data
        #first get all car info
        vid_all = []
        for i in range(scene_tb_length):
            for j in range(len(scene_data[i])):
                car_id = int(scene_data[i][j].id)
                if car_id not in vid_all:
                    vid_all.append(car_id)
        vid_all = sorted(vid_all)

        scene_info["wrong_and_sigma"] = np.zeros((len(vid_all), scene_tb_length, 7))
        scene_info["distance_info"] = -np.ones((len(vid_all), scene_tb_length, 6))

        all_cars_posi = -np.ones((len(vid_all), scene_tb_length, 2))
        for i in range(scene_tb_length):
            for car in scene_data[i]:
                car_id = int(car.id)
                j = vid_all.index(car_id)
                all_cars_posi[j, i, :] = np.array([car.location.x, car.location.y])

        scene_info["distance_info"][:, :, :2] = all_cars_posi

        #then check if angle or dis is too big
        for i in range(len(vid_all)):
            position = all_cars_posi[i]
            if i == test_id:
                print(position)
            heading = []
            for j in range(1, len(position)):
                if position[j][0] == -1 or position[j-1][0] == -1:
                    continue
                #heading.append([np.arctan2(position[j][1] - position[j - 1][1], position[j][0] - position[j - 1][0]), j])
                dis_car = np.sqrt((position[j][1] - position[j - 1][1])**2 + (position[j][0] - position[j - 1][0])**2)
                heading.append([np.arctan2(position[j][1] - position[j - 1][1], position[j][0] - position[j - 1][0]), j, dis_car])
                if dis_car >= 10:
                    scene_info["wrong_and_sigma"][i, j, 0] = 1
            
            if len(heading) < 2:
                continue
            for j in range(len(heading) - 1):
                if i == test_id:
                    print(j)
                    print(heading[j][1], heading[j + 1][1])

                    print(abs(heading[j][0] - heading[j + 1][0])*180/np.pi)
                    print(abs(2 * np.pi - abs(heading[j][0] - heading[j + 1][0]))*180/np.pi)
                    print(heading[j][2], heading[j+1][2])

                if abs(heading[j][0] - heading[j + 1][0]) < np.pi / 4 or abs(2 * np.pi - abs(heading[j][0] - heading[j + 1][0])) < np.pi / 4:
                    continue
                if heading[j][2] < 0.5 or heading[j+1][2] < 0.5:
                    continue
                scene_info["wrong_and_sigma"][i, heading[j + 1][1], 1] = 1

        break
#python z_try2.py --experiment-name vis_1 --folder-idx 4 --config ./configs/rounD_inference.yml