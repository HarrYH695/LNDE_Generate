import torch
import argparse
import os
import yaml
import warnings
import shutil
from tqdm import tqdm
from simulation_modeling.simulation_inference_gmm import SimulationInference_gmm
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
    simulation_inference_model = SimulationInference_gmm(configs=configs)

    dir_name = "rD_trial_2"
    file_ori = "/nfs/turbo/coe-mcity/hanhy/LNDE_inference_data/LNDE_ignore_0730_2/" + dir_name + "/1_1/"
    save_dir = "/nfs/turbo/coe-mcity/hanhy/LNDE_inference_data/LNDE_ignore_0730_2/" + dir_name + "/scene_videos_1_1/"

    # file_ori = '/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/LNDE_inference_data/test/baseline_1_t2/1/'
    # save_dir = '/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/LNDE_inference_data/test/baseline_1_t2/video_1/'

    file_ori = '/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/LNDE_inference_data/test/gmm_0726_t2/cases/54/6steps/'
    save_dir = '/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/LNDE_inference_data/test/gmm_0726_t2/cases/54/6steps_vis/'

    # file_ori = '/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/LNDE_inference_data/test/baseline_1_t2/cases/51/5steps/'
    # save_dir = '/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/LNDE_inference_data/test/baseline_1_t2/cases/51/5steps_vis/'

    # file_ori = '/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/LNDE_inference_data/LNDE_ignore_0726_2/rD_trial_2_4/4_1/'
    # save_dir = '/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/LNDE_inference_data/LNDE_ignore_0726_2/rD_trial_2_4/scene_video_4_1/'

    number = 691
    file_ori = '/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/LNDE_inference_data/LNDE_ignore_0726_2/rD_trial_2_4/bigangle/'
    save_dir = file_ori + f'{number}_vis/'
    file_ori += f'{number}/'


    file_ori = '/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/LNDE_inference_data/test/gmm_0726_t2/cases/gmm/51_2/10steps/'
    save_dir = '/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/LNDE_inference_data/test/gmm_0726_t2/cases/gmm/51_2/10steps_vis/'

    file_ori = '/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/Data_Gen/data_ignore_new_all_0805/train/'
    save_dir = '/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/Data_Gen/data_ignore_new_all_0805/train_vis/'

    dir_name = "rD_baseline"
    # dir_name = "rD_Trial_2"
    dir_wrong_data_info = "/nfs/turbo/coe-mcity/hanhy/LNDE_Results/" + dir_name + "/1/"

    file_ori = '/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/LNDE_inference_data/LNDE_ignore_0807/t4/2/'
    save_dir = '/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/LNDE_inference_data/LNDE_ignore_0807/t4/2_vis/'
    scenes_all = os.listdir(file_ori)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    num = 0
    for scene_i in tqdm(scenes_all):
        
        # scene_i = '24_1.pkl'
        # if not os.path.exists(file_ori+scene_i):
        #     continue
        # scene_i = '462.pkl'
        scene_data_file = pickle.load(open(file_ori+scene_i, "rb"))

        # for cars in scene_data_file:
        #     print([car.id for car in cars])
        # break
        # scene_data = scene_data_file["states_all"]

        # scene_data_file = pickle.load(open(dir_wrong_data_info+'9242.pkl', 'rb'))
        # scene_data = scene_data_file['states_considered'][13:]
        
        # if_wrong_traj = check_if_wrong_traj(scene_data_file)

        if True: #not if_wrong_traj:
            simulation_inference_model.save_check_sample_result(time_buff=scene_data_file, idx=scene_i[:-4], save_path=save_dir, with_traj=True)
            num += 1
            # coll_all.append(collision_time)
        # break

        # if num >= 100:
        #     break
    print(f"video_num:{num}")
    #print(coll_all)
    

#python save_video_res.py --experiment-name vis_1 --folder-idx 4 --config ./configs/rounD_inference.yml