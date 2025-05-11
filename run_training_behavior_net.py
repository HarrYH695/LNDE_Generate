import argparse
import os
import yaml
import random
import shutil
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle

from behavior_net import datasets
from behavior_net import Trainer

# settings
parser = argparse.ArgumentParser()
parser.add_argument('--experiment-name', type=str, required=True,
                    help='The name of the experiment folder where the data will be stored')
parser.add_argument('--save-result-path', type=str, default=r'./results_latest/training/behavior_net',
                    help='The path to save the training results, a folder with experiment_name will be created in the path')
parser.add_argument('--config', type=str, required=True,
                    help='The path to the training config file. E.g., ./configs/AA_rdbt_behavior_net_training.yml')
args = parser.parse_args()


def check_data_loading():
    # to check if the data is loading correctly?
    for i in range(100):
        data = next(iter(dataloaders['train']))
        x_in = data['input'][0]
        x_out = data['gt'][0]
        plt.subplot(1, 2, 1), plt.imshow(x_in), plt.title('input')
        plt.subplot(1, 2, 2), plt.imshow(x_out), plt.title('gt')
        plt.show()
        print(data['input'].shape)
        print(data['gt'].shape)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def dump_rng_states(path):
    state = {
        'python_random_state': random.getstate(),
        'numpy_random_state': np.random.get_state(),
        'torch_cpu_rng_state': torch.get_rng_state(),
        'torch_cuda_rng_states': (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() else None
        ),
        'torch_initial_seed': torch.initial_seed(),
    }

    with open(os.path.join(path, "seeds.pkl"), 'wb') as f:
        pickle.dump(state, f)
    
def load_rng_states(path):
    with open(path, 'rb') as f:
        state = pickle.load(f)

    random.setstate(state['python_random_state'])
    np.random.set_state(state['numpy_random_state'])
    torch.set_rng_state(state['torch_cpu_rng_state'])
    if state['torch_cuda_rng_states'] is not None:
        torch.cuda.set_rng_state_all(state['torch_cuda_rng_states'])

    print(f"Restored RNG states.  original torch initial seed = "
          f"{state['torch_initial_seed']}")


if __name__ == '__main__':
    # Load config file
    with open(args.config) as file:
        try:
            configs = yaml.safe_load(file)
            print("Loading config file: {0}".format(args.config))
        except yaml.YAMLError as exception:
            print(exception)

    # Checkpoints and training process visualizations save paths
    experiment_name = args.experiment_name
    save_result_path = args.save_result_path
    configs["checkpoint_dir"] = os.path.join(save_result_path, experiment_name, "checkpoints")  # The path to save trained checkpoints
    configs["vis_dir"] = os.path.join(save_result_path, experiment_name, "vis_training")  # The path to save training visualizations

    # Save the config file of this experiment
    os.makedirs(os.path.join(save_result_path, experiment_name), exist_ok=True)
    save_path = os.path.join(save_result_path, experiment_name, "config.yml")
    shutil.copyfile(args.config, save_path)

    # set or save or load the random seed
    # seed = 2025
    # set_seed(seed)

    #dump_rng_states(os.path.join(save_result_path, experiment_name))

    seed_file = "/home/hanhy/ondemand/data/sys/myjobs/LNDE_Generate/results_latest/training/behavior_net/rounD_t1_r4/seeds.pkl"
    load_rng_states(seed_file)


    # Initialize the DataLoader
    dataloaders = datasets.get_loaders(configs)

    # Check data loading
    # check_data_loading()

    # Initialize the training process
    m = Trainer(configs=configs, dataloaders=dataloaders)
    m.train_models()

# python run_training_behavior_net.py --config ./configs/rounD_behavior_net_training.yml --experiment-name rounD_b_seed_r4
# 2: corr和std分离 
# 3:mae->mse   
# 2_r and 2r2: repeat 2 , 2r2 record std and corr
# 4: all mse_c in loss
# 2_c : use square rather than elu()+1
# 2_s : softplus() # std**2.
# 2r_woD: 2r + no adv loss
# 2c_woD: 2c + no adv loss

# baseline_2 : only add vae
# baseline_3 : mse

# --------------------------------------
# New: set seed = 0
# baseline
# Trial_1: joint gaussian + vae_sample
# Trial_1_r1: change seed to 42
# Trial_1_r2: random seed and read them
# Trial_1_r3: change seed to 2025
# Trial_1_r4: random seed and read them
# Trial_2: just joint gaussian
# Trial_3: joint gaussian + gaussian mixture: 3 mixture