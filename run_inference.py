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
    
    #test!
    file_t = "/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/LNDE_Results/Trial_6/2/"
    # with open("/home/hanhy/ondemand/data/sys/myjobs/Conflict_Identifier_Network/wrong_case_2.txt", "r") as f:
    #     for line in f:
    #         file_info = line.strip()
    #         file_info = line[:-5]
    #         file_name, num_s = file_info.split('_')
    #         data = pickle.load(open(file_t+file_name+'.pkl', "rb"))
    #         timeb = data["states_considered"]
    #         timeb_considered = timeb[int(num_s):int(num_s)+6]
    #         num_idx=np.zeros(2)
    #         num_idx[0] = int(file_name)
    #         num_idx[1] = int(num_s)
    #         simulation_inference_model.run_sim_steps_for_certain_TIME_BUFF(time_buff=timeb_considered, sim_num=100, result_dir="/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/z_res/", num_idx=num_idx, poc_dir="/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/z_poc_changed/")


    # file_idx = 4043
    # start_i = 9
    # file_t = "/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/LNDE_Results/Trial_7/1/"
    # data = pickle.load(open(file_t+f"{file_idx}.pkl", "rb"))
    # timeb = data["states_considered"]
    # timeb_considered = timeb[start_i:start_i+5]
    # print(len(timeb_considered))
    # simulation_inference_model.run_sim_steps_for_certain_TIME_BUFF(time_buff=timeb_considered, sim_num=100, result_dir="/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/z_res/", num_idx=np.zeros(2), poc_dir="/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/")

    # for i in tqdm(range(10)):
    #     if os.path.exists("/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/z_res/"+f"0_0_{i}.pkl"):
    #         data = pickle.load(open("/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/z_res/"+f"0_0_{i}.pkl", "rb"))
    #         simulation_inference_model.save_check_sample_result(time_buff=data, idx=f"{i}", save_path="/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/z_res2/")

    # Run simulations.
    #simulation_inference_model.run_simulations(sim_num=configs["sim_num"])
    dir_name = "rD_Trial_2r_woD"
    save_sim_path = "/nfs/turbo/coe-mcity/hanhy/LNDE_Results/" + dir_name + "/1/"
    if not os.path.exists(save_sim_path):
        os.makedirs(save_sim_path)
    coll_num = 0
    for idx in tqdm(range(10000)):
        #print(f"----------------{idx}----------------")
        coll = simulation_inference_model.check_crash_samples(max_time=1000, result_dir=save_sim_path, num_idx=idx)
        coll_num += coll
    print(f"Find collision num: {coll_num}")

    #Get the visual of 1000 results
    # save_path_1 = "/nfs/turbo/coe-mcity/hanhy/LNDE_Results/" + dir_name + "/2/"
    # if not os.path.exists(save_path_1):
    #     os.makedirs(save_path_1)
    
    # save_path_2 = "/nfs/turbo/coe-mcity/hanhy/LNDE_Results/" + dir_name + "/3/"
    # if not os.path.exists(save_path_2):
    #     os.makedirs(save_path_2)
    
    # for i in tqdm(range(10000)):
    #     if os.path.exists(save_sim_path + f"{i}.pkl"):
    #         with open(save_sim_path + f"{i}.pkl", "rb") as f:
    #             infos = pickle.load(f)
    #             time_buff_all = infos["states_considered"]
    #             if len(time_buff_all) > 6:
    #                 for win_start in range(len(time_buff_all) - 6,len(time_buff_all) - 5):
    #                     num_idx = np.zeros(2)
    #                     num_idx[0] = i
    #                     num_idx[1] = win_start

    #                     #simulate + vis_res: Image and PoC
    #                     dir_path = save_path_2 + f"{i}_{win_start}/"
    #                     if not os.path.exists(dir_path):
    #                         os.makedirs(dir_path)
    #                     simulation_inference_model.run_sim_steps_for_certain_TIME_BUFF(time_buff=time_buff_all[win_start:(win_start+5)], sim_num=100, result_dir=save_path_1, num_idx=num_idx, poc_dir=dir_path)
    #                     simulation_inference_model.save_check_sample_result(time_buff=time_buff_all[win_start:(win_start+5)], idx="vis", save_path=dir_path, with_traj=True)
                        
                        
                        #simulation_inference_model._save_vis_time_buff(TIME_BUFF=time_buff_all[win_start:(win_start+5)], background_map=simulation_inference_model.background_map, save_path=dir_path+"vis.png")

#python run_inference.py --experiment-name vis_1 --folder-idx 4 --config ./configs/rounD_inference.yml --viz-flag
# store results in : /nfs/turbo/coe-mcity/hanhy/LNDE_Results
# 2c: 99
# 2r2: 279
# baseline3: 279
# baseline2: 279
# 2r_woD: 99
# 2c_woD: 209
# 2s_woD: 199
# 0 0 0 0 0 0 0 