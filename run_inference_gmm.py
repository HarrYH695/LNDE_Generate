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
import random

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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    # Load config file
    with open(args.config) as file:
        try:
            configs = yaml.safe_load(file)
            print(f"Loading config file: {args.config}")
        except yaml.YAMLError as exception:
            print(exception)

    #set and save the random seed
    seed = 0
    set_seed(seed)

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
    
    #test and gen ignore data
    # save_dir = '/nfs/turbo/coe-mcity/hanhy/LNDE_Data/data_ignore_gmm_b3_val/'
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

    # for i in tqdm(range(2500)):
    #     save_path_i = os.path.join(save_dir, f'{i}.pkl')
    #     simulation_inference_model.generate_prob_ignore_results(save_path=save_path_i)

    # dir_name = "gmm_0726_t2"
    # num_steps = 6
    # num_idx_list = [11, 24, 43, 54, 65, 87]
    # for num_idx in range(100):
    #     #num_idx = 26
    #     if num_idx not in num_idx_list:
    #         continue

    #     save_sim_path = "/nfs/turbo/coe-mcity/hanhy/LNDE_inference_data/test/" + dir_name + f"/cases/{num_idx}/{num_steps}steps/"
    #     if not os.path.exists(save_sim_path):
    #         os.makedirs(save_sim_path)
    #     coll_num = 0

    #     file_path = f"/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/LNDE_inference_data/test/baseline_1_t2/1/{num_idx}.pkl"
    #     if not os.path.exists(file_path):
    #         continue

    #     data = pickle.load(open(file_path, 'rb'))
    #     tb = data['states_all']

    #     for idx in range(10):
    #         #print(f"----------------{idx}----------------")
    #         coll = simulation_inference_model.check_crash_distribute(max_time=100, result_dir=save_sim_path, num_idx=num_idx, num_try=idx, num_steps=num_steps,if_all=True, initial_TIME_BUFF=tb)
    #         coll_num += coll
        
    #         # if coll_num >= 9:
    #     print(num_idx, coll_num)
    #         # print(coll_num)

    # print(f"Find collision num: {coll_num}")


    # Run simulations.
    #simulation_inference_model.run_simulations(sim_num=configs["sim_num"])

    # dir_name = "rD_trial_2_4"
    # number = 4790
    # save_sim_path = "/nfs/turbo/coe-mcity/hanhy/LNDE_inference_data/LNDE_ignore_0726_2/" + dir_name + f"/bigangle/{number}/"
    # if not os.path.exists(save_sim_path):
    #     os.makedirs(save_sim_path)
    # coll_num = 0

    # original_dir = f'/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/LNDE_Results/rD_baseline/1/{number}.pkl'
    # data = pickle.load(open(original_dir, 'rb'))
    # print(len(data['states_considered']))

    # tb = data['states_considered'][8:13]
    # for cars in data['states_considered']:
    #     print([car.id for car in cars])
    # for idx in tqdm(range(1)):
    #     #print(f"----------------{idx}----------------")
    #     coll = simulation_inference_model.check_crash_samples(max_time=100, result_dir=save_sim_path, num_idx=idx, if_all=True, initial_TIME_BUFF=tb, save_not_coll=True)
    #     coll_num += coll
    # print(f"Find collision num: {coll_num}")

# python run_inference_gmm.py --experiment-name vis_1 --folder-idx 4 --config ./configs/rounD_inference.yml --viz-flag

    # dir_name = "rD_trial_2_4"
    # save_sim_path = "/nfs/turbo/coe-mcity/hanhy/LNDE_inference_data/LNDE_ignore_0726_2/" + dir_name + "/4_1/"
    # if not os.path.exists(save_sim_path):
    #     os.makedirs(save_sim_path)
    # coll_num = 0

    # all_samples = []

    # for idx in range(2, 100):
    #     if not os.path.exists(save_sim_path + f'{idx}.pkl'):
    #         continue

    #     data = pickle.load(open(save_sim_path + f'{idx}.pkl', 'rb'))

    #     crash_step = data['crash_step']
    #     all_steps = data['states_considered']

    #     crash_rate_all = []
    #     print(idx, len(all_steps))

    #     steps_to_go = len(all_steps) - 5
    #     for i in tqdm(range(steps_to_go)):
    #         time_buff = all_steps[i:i+5]
    #         num_crash = 0

    #         for j in range(50):
    #             coll = simulation_inference_model.check_crash_distribute(max_time=len(all_steps)+50, result_dir='', num_idx=0, num_try=0, num_steps=0, if_all=False, initial_TIME_BUFF=time_buff)
    #             num_crash += coll

    #         crash_rate_all.append(num_crash/50)


    #     all_samples.append(crash_rate_all)

    #     print(crash_rate_all)

    
    # with open('z_gmm_crash_change.pkl', 'wb') as f1:
    #     pickle.dump(all_samples, f1)


    # file_path = '/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/LNDE_inference_data/test/gmm_0726_t2/cases/gmm/51/10steps/'
    # file_path = '/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/LNDE_inference_data/test/gmm_0726_t2/cases/gmm/71/10steps/'



    # data = pickle.load(open(file_path + '71_0.pkl', 'rb'))

    # all_steps = data['states_all']

    # i = 0
    # time_buff = all_steps[i:i+5]
    # num_crash = 0

    # coll = simulation_inference_model.check_crash_distribute(max_time=100, result_dir=file_path, num_idx=71, num_try=-1, num_steps=0, if_all=False, initial_TIME_BUFF=time_buff)

    # print(coll)


    # dir_name = "rD_trial_2"
    # save_sim_path = "/nfs/turbo/coe-mcity/hanhy/LNDE_inference_data/LNDE_ignore_0730_2/" + dir_name + "/1_1/"
    # if not os.path.exists(save_sim_path):
    #     os.makedirs(save_sim_path)
    # coll_num = 0
    # for idx in tqdm(range(40)):
    #     #print(f"----------------{idx}----------------")
    #     coll = simulation_inference_model.check_samples_multi_all_in_one(save_dir=save_sim_path, save_idx=idx, repeat_num=20, predict_time=20, initial_TIME_BUFF=None)
    #     coll_num += coll
    # print(f"Find collision num: {coll_num}")


    # Get the visual of 1000 results
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
                        
                        
    #                     simulation_inference_model._save_vis_time_buff(TIME_BUFF=time_buff_all[win_start:(win_start+5)], background_map=simulation_inference_model.background_map, save_path=dir_path+"vis.png")

    dir_name = "gmm_0726_t2"
    # num_idx = 26
    # num_steps = 4
    num_idx_list = [24, 43, 59, 64, 65, 72, 86, 87] #[11, 51, 54, 71] # [24, 43, 59, 64, 65, 72, 86, 87] 

    coll_rate_dir = {}
    for num_idx in num_idx_list:
        coll_rate_dir[f'{num_idx}'] = []
    
    # for num_idx in num_idx_list:
    #     file_path = f"/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/LNDE_inference_data/test/baseline_1_t2/1/{num_idx}.pkl"

    #     data = pickle.load(open(file_path, 'rb'))
    #     tb = data['states_all']
    #     print(len(tb))

    for num_steps in tqdm(range(3, 14)):
    
        for num_idx in num_idx_list:
            save_sim_path = "/nfs/turbo/coe-mcity/hanhy/LNDE_inference_data/test/" + dir_name + f"/cases/gmm/{num_idx}_2/{num_steps}steps/"
            if not os.path.exists(save_sim_path):
                os.makedirs(save_sim_path)

            file_path = f"/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/LNDE_inference_data/test/baseline_1_t2/1/{num_idx}.pkl"

            data = pickle.load(open(file_path, 'rb'))
            tb = data['states_all']
            # print(len(tb))
            num_counted = 0
            coll_num = 0

            for idx in range(1000):
                #print(f"----------------{idx}----------------")
                coll = simulation_inference_model.check_crash_distribute(max_time=300, result_dir=save_sim_path, num_idx=num_idx, num_try=idx, num_steps=num_steps,if_all=True, initial_TIME_BUFF=tb)
                
                if coll == 1:
                    num_counted += 1
                    coll_num += 1
                elif coll == 0:
                    num_counted += 1

                if num_counted >= 100:
                    coll_rate_dir[f'{num_idx}'].append([coll_num, num_counted])
                    print(num_steps, num_idx, coll_num, num_counted)
                    break

                if idx > 998:
                    coll_rate_dir[f'{num_idx}'].append([coll_num, num_counted])
                    print(num_steps, num_idx, coll_num, num_counted)
                    break

    with open('zzz_coll_curve_2_1.pkl', 'wb') as f1:
        pickle.dump(coll_rate_dir, f1)


            # if coll_num <= 6:
            


        # print(f"Find collision num: {coll_num}")

# python run_inference_gmm.py --experiment-name vis_1 --folder-idx 4 --config ./configs/rounD_inference.yml --viz-flag

# store results in : /nfs/turbo/coe-mcity/hanhy/LNDE_Results
# baseline: 299
# 2: 69
# 2c: 99
# 2r2: 279
# baseline3: 279
# baseline2: 279
# 2r_woD: 59
# 2c_woD: 109
# 2s_woD: 199
# 0 0 0 0 0 0 0 

#----------------------------------
#new try
#baseline:139
#b_seed_r4:99
#t1:139 t1_r:the best trial before, trial_2
#t2_loss_1:29(non nan) 129(nan)(have experienced: can not inference)
#t2_loss_2:129

#t1_r1:59
#t1_r2:89
#t1_r3:109
#t1_r4:84