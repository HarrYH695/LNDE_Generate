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
    simulation_inference_model = SimulationInference(configs=configs)

    file_t = "/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/LNDE_Results/AA_Trial_1/1/"
    file_o = "/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/LNDE_Results/AA_Trial_1/2/"

    file_to_check = "/home/hanhy/ondemand/data/sys/myjobs/Conflict_Identifier_Network/AA_rdbt_data_after_check_txt/wrong_poc_small.txt"
    num_ca = 0
    num_std = 0
    with open(file_to_check, "r") as f1:
        for line in tqdm(f1):
            line = line.strip()[:-4]
            num, idx = line.split("_")
            if int(idx) > 0:
                flag = True

                file_input = num + ".pkl"
                #print(file_input)
                data = pickle.load(open(file_t + file_input, "rb"))
                info = data["states_considered"]
                info_input = info[int(idx) - 1 : int(idx) + 4]
                zs, zx, ys, yx = simulation_inference_model.check_one_sample(info_input)

                pred_range = {}
                for car in zs:
                    pred_range[int(car.id)] = []
                    pred_range[int(car.id)].append([car.location.x, car.location.y])

                for car in zx:
                    pred_range[int(car.id)].append([car.location.x, car.location.y])

                for car in ys:
                    pred_range[int(car.id)].append([car.location.x, car.location.y])

                for car in yx:
                    pred_range[int(car.id)].append([car.location.x, car.location.y])

                #原simu结果
                info_output = info[int(idx) + 4]

                #check if out 3_sigma
                for car in info_output:
                    car_x = car.location.x
                    car_y = car.location.y

                    if int(car.id) in pred_range:
                        car_zs = pred_range[int(car.id)][0]
                        car_yx = pred_range[int(car.id)][-1]

                        # print(car_zs)
                        # print(car_yx)
                        # print(car_x, car_y)
                        std_x = (car_yx[0] - car_zs[0]) / 6
                        std_y = (car_yx[1] - car_zs[1]) / 6
                        mean_x = (car_yx[0] + car_zs[0]) / 2
                        mean_y = (car_yx[1] + car_zs[1]) / 2
                        # if car_x < car_zs[0] or car_x > car_yx[0]:
                        #     flag = False

                        # if car_y < car_zs[1] or car_y > car_yx[1]:
                        #     flag = False
                        if np.abs(car_x - mean_x) > 3 * std_x:
                            flag = False

                        if np.abs(car_y - mean_y) > 3 * std_y:
                            flag = False

                if flag == False:
                    num_ca += 1
                    # with open("/home/hanhy/ondemand/data/sys/myjobs/Conflict_Identifier_Network/AA_rdbt_data_after_check_txt/poc_out_of_3sigma.txt", "a+") as ff:
                    #     ff.write(line+".pkl\n")

                if std_x >= 0.5 or std_y >= 0.5:
                    num_std += 1
print(num_ca)
print(num_std)
#python run_inference_check.py --experiment-name vis_1 --folder-idx 4 --config ./configs/AA_rdbt_inference.yml