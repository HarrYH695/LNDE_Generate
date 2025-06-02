import numpy as np
import matplotlib.pyplot as plt
import os
import json
import random
import math

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import torch.distributions as distributions
import torch.nn.functional as F

from .networks_gmn import define_G, define_D, set_requires_grad, Network_G
from .loss import UncertaintyRegressionLoss, GANLoss
from .metric import RegressionAccuracy
from . import utils

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Trainer_gmn(object):
    """
    Model trainer. This class provides fundamental methods for model training and checkpoint management.
    """

    def __init__(self, configs, dataloaders):

        self.dataloaders = dataloaders

        # input and output dimension
        self.input_dim = 4 * configs["history_length"]  # x, y, cos_heading, sin_heading
        self.output_dim = 4 * configs["pred_length"]  # x, y, cos_heading, sin_heading
        self.pred_length = configs["pred_length"]
        self.m_tokens = configs["max_num_vehicles"]

        self.n_gaussian = configs["n_gaussian"]
        self.sample_times = configs["sample_times"]
        self.epoch_thres = configs["epoch_thres"]

        # initialize networks
        # self.net_G = define_G(
        #     configs["model"], input_dim=self.input_dim, output_dim=self.output_dim,
        #     m_tokens=self.m_tokens, n_gaussian=self.n_gaussian).to(device)
        self.net_G = Network_G(configs["model"], input_dim=self.input_dim, output_dim=self.output_dim,
            m_tokens=self.m_tokens, n_gaussian=self.n_gaussian).to(device)
        self.net_D = define_D(input_dim=self.output_dim).to(device)

        # Learning rate
        self.lr = configs["lr"]
        self.warmup_steps = configs["warmup_steps"]
        self.total_steps = configs["max_num_epochs"]
        
        self.lambda_H = configs["lambda_H"]
        self.lambda_std = configs["lambda_std"]
        self.lambda_corr = configs["lambda_corr"]
        
        self.lambda_reg_posi = configs["lambda_reg_posi"]
        self.lambda_reg_angle = configs["lambda_reg_angle"]

        # define optimizers
        self.optimizer_G = optim.AdamW(self.net_G.parameters(), lr=self.lr, betas=(0.9,0.999), weight_decay=1e-4,  eps=1e-8)
        self.optimizer_D = optim.RMSprop(self.net_D.parameters(), lr=self.lr)

        # define lr schedulers
        # self.exp_lr_scheduler_G = lr_scheduler.StepLR(
        #     self.optimizer_G, step_size=configs["lr_decay_step_size"], gamma=configs["lr_decay_gamma"])
        self.exp_lr_scheduler_D = lr_scheduler.StepLR(
            self.optimizer_D, step_size=configs["lr_decay_step_size"], gamma=configs["lr_decay_gamma"])

        warmup_steps = self.warmup_steps
        total_steps = self.total_steps
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            lr_cos = 0.001 + 0.999 * 0.5 * (1 + math.cos(torch.pi * progress))
            return lr_cos  # eta_min=1e-3*lr
        
        self.exp_lr_scheduler_G =  torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda)

        # define loss function and error metric
        self.regression_loss_func_pos = UncertaintyRegressionLoss(choice='mae_c')
        self.regression_loss_func_mu = UncertaintyRegressionLoss(choice='mae')
        self.regression_loss_func_heading = UncertaintyRegressionLoss(choice='cos_sin_heading_mae')

        self.gan_loss = GANLoss(gan_mode='vanilla').to(device)

        self.accuracy_metric_pos = RegressionAccuracy(choice='neg_mae')
        self.accuracy_metric_heading = RegressionAccuracy(choice='neg_mae')

        # define some other vars to record the training states
        self.running_acc_pos = []
        self.running_acc_sample = []
        self.running_acc_heading = []
        self.running_loss_G, self.running_loss_D = [], []
        self.running_loss_G_reg_pos, self.running_loss_G_reg_heading, self.running_loss_G_adv = [], [], []
        self.running_std_x, self.running_std_y, self.running_corr, self.running_pi = [], [], [], []
        self.train_epoch_acc_pos, self.train_epoch_acc_sample, self.train_epoch_acc_heading = -1e9,-1e9,-1e9
        self.train_epoch_loss = 0.0
        self.val_epoch_acc_pos, self.val_epoch_acc_sample, self.val_epoch_acc_heading = -1e9,-1e9,-1e9
        self.val_epoch_loss = 0.0
        self.epoch_to_start = 0
        self.max_num_epochs = configs["max_num_epochs"]

        self.is_training = True
        self.batch = None
        self.batch_loss = 0.0
        self.batch_acc_pos = -1e9
        self.batch_acc_heading = -1e9
        self.batch_acc_sample = -1e9
        self.batch_mean_std_x = []
        self.batch_mean_std_y = []
        self.batch_mean_corr = []
        self.batch_mean_pi = []
        self.batch_id = 0
        self.epoch_id = 0

        self.checkpoint_dir = configs["checkpoint_dir"]
        self.vis_dir = configs["vis_dir"]

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        if os.path.exists(self.vis_dir) is False:
            os.makedirs(self.vis_dir, exist_ok=True)

        # buffers to logfile
        self.logfile = {'val_acc_pos': [], 'train_acc_pos': [],
                        'val_acc_heading': [], 'train_acc_heading': [],
                        'val_loss_G': [], 'train_loss_G': [],
                        'val_loss_D': [], 'train_loss_D': [],
                        'epochs': []}

        self.logfile_batch = {'train_loss_G_pos': [], 'train_loss_G_ang': [],
                            'train_loss_G_adv': [], 'train_loss_D':[], 
                            'iter': []}

        # tensorboard writer
        self.writer = SummaryWriter(self.vis_dir)

    def _load_checkpoint(self, start_id=0):

        if os.path.exists(os.path.join(self.checkpoint_dir, f'ckpt_{start_id}.pt')):
            print('loading last checkpoint...')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, f'ckpt_{start_id}.pt'))

            # update net_G states
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            self.exp_lr_scheduler_G.load_state_dict(
                checkpoint['exp_lr_scheduler_G_state_dict'])
            self.net_G.to(device)

            # update net_D states
            self.net_D.load_state_dict(checkpoint['model_D_state_dict'])
            self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
            self.exp_lr_scheduler_D.load_state_dict(
                checkpoint['exp_lr_scheduler_D_state_dict'])
            self.net_D.to(device)

            # update some other states
            self.epoch_to_start = checkpoint['epoch_id'] + 1

            print('Epoch_to_start = %d' % self.epoch_to_start)
            print()

        else:
            print('training from scratch...')

    def _save_checkpoint(self, ckpt_name):
        torch.save({
            'epoch_id': self.epoch_id,
            'model_G_state_dict': self.net_G.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'exp_lr_scheduler_G_state_dict': self.exp_lr_scheduler_G.state_dict(),
            'model_D_state_dict': self.net_D.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'exp_lr_scheduler_D_state_dict': self.exp_lr_scheduler_D.state_dict()
        }, os.path.join(self.checkpoint_dir, ckpt_name))

    def _update_lr_schedulers(self):
        self.exp_lr_scheduler_G.step()
        #self.exp_lr_scheduler_D.step()

    def _collect_running_batch_states(self):

        self.running_acc_sample.append(self.batch_acc_sample.item())
        self.running_acc_heading.append(self.batch_acc_heading.item())
        self.running_loss_G.append(self.batch_loss_G.item())
        # self.running_loss_D.append(self.batch_loss_D.item())
        # self.running_loss_G_reg_pos.append(self.reg_loss_position.item())
        # self.running_loss_G_reg_heading.append(self.reg_loss_heading.item())
        # self.running_loss_G_adv.append(self.G_adv_loss.item())
        
        self.running_std_x.append(self.batch_mean_std_x)
        self.running_std_y.append(self.batch_mean_std_y)
        self.running_corr.append(self.batch_mean_corr)
        self.running_pi.append(self.batch_mean_pi)


        # if self.is_training:
        #     m_batches = len(self.dataloaders['train'])
        # else:
        #     m_batches = len(self.dataloaders['val'])

        # if True:#np.mod(self.batch_id, 500) == 1:
        #     print('Is_training: %s. epoch [%d,%d], batch [%d,%d], reg_loss_pos: %.5f, reg_loss_heading: %.5f, '
        #           'G_adv_loss: %.5f, D_adv_loss: %.5f, running_acc_pos: %.5f, running_acc_heading: %.5f'
        #           % (self.is_training, self.epoch_id, self.max_num_epochs-1, self.batch_id, m_batches,
        #              self.reg_loss_position.item(), self.reg_loss_heading.item(),
        #              self.G_adv_loss.item(), self.D_adv_loss.item(), np.mean(self.running_acc_pos),
        #              np.mean(self.running_acc_heading)))
            
        #     print(f"std_x:{np.mean(self.running_std_x)}, std_y:{np.mean(self.running_std_y)}, corr:{np.mean(self.running_corr)}")
        #     print(len(self.running_std_x), len(self.running_std_y), len(self.running_corr))


    def _collect_epoch_states(self):
        if self.is_training:
            self.train_epoch_acc_sample = np.mean(self.running_acc_sample).item()
            self.train_epoch_acc_heading = np.mean(self.running_acc_heading).item()
            self.train_epoch_loss_G = np.mean(self.running_loss_G).item()
            # self.train_epoch_loss_D = np.mean(self.running_loss_D).item()
            # self.train_epoch_loss_G_reg_pos = np.mean(self.running_loss_G_reg_pos).item()
            # self.train_epoch_loss_G_reg_heading = np.mean(self.running_loss_G_reg_heading).item()
            # self.train_epoch_loss_G_adv = np.mean(self.running_loss_G_adv).item()

            std_x_all = np.array(self.running_std_x)
            std_y_all = np.array(self.running_std_y)
            corr_all = np.array(self.running_corr)
            pi_all_epoch = np.array(self.running_pi)

            self.train_epoch_std_x = np.mean(std_x_all, axis=0)
            self.train_epoch_std_y = np.mean(std_y_all, axis=0)
            self.train_epoch_corr = np.mean(corr_all, axis=0)
            self.train_epoch_pi = np.mean(pi_all_epoch, axis=0)

            print('Training, Epoch %d / %d, epoch_loss_G= %.5f, epoch_acc_sample= %.5f' %
                  (self.epoch_id, self.max_num_epochs-1,
                   self.train_epoch_loss_G, self.train_epoch_acc_sample))

            # print(f"train_epoch_std_x:{self.train_epoch_std_x}, train_epoch_std_y:{self.train_epoch_std_y}, train_epoch_corr:{self.train_epoch_corr}")


            self.writer.add_scalar("Accuracy/train_sample", self.train_epoch_acc_sample, self.epoch_id)
            self.writer.add_scalar("Accuracy/train_heading", self.train_epoch_acc_heading, self.epoch_id)
            self.writer.add_scalar("Loss/train_G", self.train_epoch_loss_G, self.epoch_id)
            # self.writer.add_scalar("Loss/train_D", self.train_epoch_loss_D, self.epoch_id)
            # self.writer.add_scalar("Loss/train_G_reg_position", self.train_epoch_loss_G_reg_pos, self.epoch_id)
            # self.writer.add_scalar("Loss/train_G_reg_heading", self.train_epoch_loss_G_reg_heading, self.epoch_id)
            # self.writer.add_scalar("Loss/train_G_adv", self.train_epoch_loss_G_adv, self.epoch_id)

            for i in range(self.n_gaussian):
                self.writer.add_scalar(f"Std_x_train/std_x_{i}", self.train_epoch_std_x[i], self.epoch_id)
                self.writer.add_scalar(f"Std_y_train/std_y_{i}", self.train_epoch_std_y[i], self.epoch_id)
                self.writer.add_scalar(f"Corr_train/corr_{i}", self.train_epoch_corr[i], self.epoch_id)
                self.writer.add_scalar(f"Pi_train/pi_{i}", self.train_epoch_pi[i], self.epoch_id)

        else:
            self.val_epoch_acc_sample = np.mean(self.running_acc_sample).item()
            self.val_epoch_acc_heading = np.mean(self.running_acc_heading).item()
            self.val_epoch_loss_G = np.mean(self.running_loss_G).item()
            # self.val_epoch_loss_D = np.mean(self.running_loss_D).item()
            # self.val_epoch_loss_G_reg_pos = np.mean(self.running_loss_G_reg_pos).item()
            # self.val_epoch_loss_G_reg_heading = np.mean(self.running_loss_G_reg_heading).item()
            # self.val_epoch_loss_G_adv = np.mean(self.running_loss_G_adv).item()
            
            std_x_all = np.array(self.running_std_x)
            std_y_all = np.array(self.running_std_y)
            corr_all = np.array(self.running_corr)
            pi_all_epoch = np.array(self.running_pi)

            self.val_epoch_std_x = np.mean(std_x_all, axis=0)
            self.val_epoch_std_y = np.mean(std_y_all, axis=0)
            self.val_epoch_corr = np.mean(corr_all, axis=0)
            self.val_epoch_pi = np.mean(pi_all_epoch, axis=0)

            print('Validation, Epoch %d / %d, epoch_loss_G= %.5f, epoch_acc_sample= %.5f' %
                  (self.epoch_id, self.max_num_epochs - 1,
                   self.val_epoch_loss_G, self.val_epoch_acc_sample))

            # print(f"val_epoch_std_x:{self.val_epoch_std_x}, val_epoch_std_y:{self.val_epoch_std_y}, val_epoch_corr:{self.val_epoch_corr}")

            self.writer.add_scalar("Accuracy/val_sample", self.val_epoch_acc_sample, self.epoch_id)
            self.writer.add_scalar("Accuracy/val_heading", self.val_epoch_acc_heading, self.epoch_id)
            self.writer.add_scalar("Loss/val_G", self.val_epoch_loss_G, self.epoch_id)
            # self.writer.add_scalar("Loss/val_D", self.val_epoch_loss_D, self.epoch_id)
            # self.writer.add_scalar("Loss/val_G_reg_position", self.val_epoch_loss_G_reg_pos, self.epoch_id)
            # self.writer.add_scalar("Loss/val_G_reg_heading", self.val_epoch_loss_G_reg_heading, self.epoch_id)
            # self.writer.add_scalar("Loss/val_G_adv", self.val_epoch_loss_G_adv, self.epoch_id)
            
            for i in range(self.n_gaussian):
                self.writer.add_scalar(f"Std_x_val/std_x_{i}", self.val_epoch_std_x[i], self.epoch_id)
                self.writer.add_scalar(f"Std_y_val/std_y_{i}", self.val_epoch_std_y[i], self.epoch_id)
                self.writer.add_scalar(f"Corr_val/corr_{i}", self.val_epoch_corr[i], self.epoch_id)
                self.writer.add_scalar(f"Pi_val/pi_{i}", self.val_epoch_pi[i], self.epoch_id)

        print()


    def _update_checkpoints(self, epoch_id):

        # save current model
        self._save_checkpoint(ckpt_name=f'ckpt_{epoch_id}.pt')
        sum_val_epoch_acc = self.val_epoch_acc_sample + self.val_epoch_acc_heading
        print('Latest model updated. Epoch_acc_sample=%.4f, Epoch_acc_heading=%.4f,'
              'Sum Epoch_acc=%.4f'
              % (self.val_epoch_acc_sample, self.val_epoch_acc_heading, sum_val_epoch_acc))
        print()

    def _update_logfile(self):

        logfile_path = os.path.join(self.checkpoint_dir, 'logfile.json')


        # read historical logfile and update
        if os.path.exists(logfile_path):
            with open(logfile_path) as json_file:
                self.logfile = json.load(json_file)
        else:
            self.logfile = {'train_acc_sample': [], 
                            'train_acc_heading': [],
                            'val_acc_sample': [],
                            'val_acc_heading': [],
                            'train_loss_G': [],
                            'val_loss_G': [],
                            # 'train_loss_D': [],
                            # 'val_loss_D': [],
                            'epochs': []}

        self.logfile['train_acc_sample'].append(self.train_epoch_acc_sample)
        self.logfile['train_acc_heading'].append(self.train_epoch_acc_heading)
        self.logfile['val_acc_sample'].append(self.val_epoch_acc_sample)
        self.logfile['val_acc_heading'].append(self.val_epoch_acc_heading)
        self.logfile['train_loss_G'].append(self.train_epoch_loss_G)
        self.logfile['val_loss_G'].append(self.val_epoch_loss_G)
        # self.logfile['train_loss_D'].append(self.train_epoch_loss_D)
        # self.logfile['val_loss_D'].append(self.val_epoch_loss_D)
        self.logfile['epochs'].append(self.epoch_id)

        # save new logfile to disk
        with open(logfile_path, "w") as fp:
            json.dump(self.logfile, fp)

    def _update_logfile_batch_train(self):

        logfile_path = os.path.join(self.checkpoint_dir, 'logfile_batch.json')

        # read historical logfile and update
        if os.path.exists(logfile_path):
            with open(logfile_path) as json_file:
                self.logfile_batch = json.load(json_file)
        else:
            self.logfile_batch = {'train_loss_G': [], 'iter': []}

        # self.logfile_batch['train_loss_G_pos'].append(self.reg_loss_position.item())
        # self.logfile_batch['train_loss_G_ang'].append(self.reg_loss_heading.item())
        # self.logfile_batch['train_loss_G_adv'].append(self.G_adv_loss.item())
        # self.logfile_batch['train_loss_D'].append(self.D_adv_loss.item())

        self.logfile_batch['train_loss_G'].append(self.batch_loss_G.item())
        self.logfile_batch['iter'].append(self.batch_id + self.epoch_id)

        # save new logfile to disk
        with open(logfile_path, "w") as fp:
            json.dump(self.logfile_batch, fp)

    def _visualize_logfile(self):

        plt.subplot(1, 2, 1)
        plt.plot(self.logfile['epochs'], self.logfile['train_loss_G'])
        plt.plot(self.logfile['epochs'], self.logfile['val_loss_G'])
        # plt.plot(self.logfile['epochs'], self.logfile['train_loss_D'])
        # plt.plot(self.logfile['epochs'], self.logfile['val_loss_D'])
        plt.xlabel('m epochs')
        plt.ylabel('loss')
        plt.legend(['train loss G', 'val loss G'])#, 'train loss D', 'val loss D'])

        plt.subplot(1, 2, 2)
        plt.plot(self.logfile['epochs'], self.logfile['train_acc_sample'])
        plt.plot(self.logfile['epochs'], self.logfile['train_acc_heading'])
        plt.plot(self.logfile['epochs'], self.logfile['val_acc_sample'])
        plt.plot(self.logfile['epochs'], self.logfile['val_acc_heading'])
        plt.xlabel('m epochs')
        plt.ylabel('acc')
        plt.legend(['train acc', 'train heading', 'val acc', 'val heading'])

        logfile_path = os.path.join(self.vis_dir, 'logfile.png')
        plt.savefig(logfile_path)
        plt.close('all')

    def _visualize_prediction(self):

        print('visualizing prediction...')
        print()

        batch = next(iter(self.dataloaders['val']))
        with torch.no_grad():
            self._forward_pass(batch)

        vis_path = os.path.join(
            self.vis_dir, 'epoch_'+str(self.epoch_id).zfill(5)+'_pred.png')

        x_pos, gt = self.x[:, :, :int(self.output_dim / 2)], self.gt[:, :, :int(self.output_dim / 2)]
        pred_pos_mean, pred_pos_std = self.G_pred_mean[0][:, :, :int(self.output_dim / 2)], self.G_pred_std[0][:, :, :int(self.output_dim / 2)]

        utils.visualize_training(
            x=x_pos, pred_true=gt,
            pred_mean=pred_pos_mean, pred_std=pred_pos_std, vis_path=vis_path)

    def _sampling_from_mu_and_std(self, mu, std, corr, epsilons_1, epsilons_2):

        #[bs, 32, 1]
        lat_mean = mu[:, :, 0:self.pred_length]
        lon_mean = mu[:, :, self.pred_length:]
        lat_std = std[:, :, 0:self.pred_length]
        lon_std = std[:, :, self.pred_length:]
        # lat_std = torch.sqrt(std[:, :, 0:self.pred_length])
        # lon_std = torch.sqrt(std[:, :, self.pred_length:])

        lat = lat_mean + lat_std * epsilons_1
        lon = lon_mean + lon_std * epsilons_1 * corr + lon_std * torch.sqrt(1 - corr ** 2) * epsilons_2
        # print(f"lat:{lat.shape}")
        # print(f"lon:{lon.shape}")

        return torch.cat([lat, lon], dim=-1)

    def _sampling_from_standard_gaussian(self, mu):

        bs, _, _ = mu.shape #[bs, 32, 2]
        epsilons_lat = torch.tensor([random.gauss(0, 1) for _ in range(bs * self.m_tokens)]).to(device).detach()
        epsilons_lon = torch.tensor([random.gauss(0, 1) for _ in range(bs * self.m_tokens)]).to(device).detach()
        epsilons_1 = torch.reshape(epsilons_lat, [bs, -1, 1])
        epsilons_2 = torch.reshape(epsilons_lon, [bs, -1, 1])

        return epsilons_1, epsilons_2
    
    def _forward_pass(self, batch, rollout=5):

        self.batch = batch
        self.x = batch['input'].to(device)
        self.gt = self.batch['gt'].to(device)

        self.mask = torch.ones_like(self.gt)
        self.mask[torch.isnan(self.gt)] = 0.0
        self.rollout_mask = torch.ones_like(self.gt)
        self.rollout_mask[torch.isnan(self.gt)] = 0.0
        self.rollout_mask[torch.sum(self.mask, dim=-1) > 0, :] = 1.0

        self.x[torch.isnan(self.x)] = 0.0
        self.gt[torch.isnan(self.gt)] = 0.0

        x_input = self.x
        self.G_pred_mean, self.G_pred_std, self.G_pred_corr, self.G_pred_pi = [], [], [], []
        self.rollout_pos = []

        for _ in range(rollout):
            # mu, std, corr, cos_sin_heading, pi_all, L = self.net_G(x_input)

            # #construct the gaussian mixture model and calculate NLL loss
            # dis_mu_L = distributions.MultivariateNormal(loc=mu, scale_tril=L)
            # dis_mix = distributions.Categorical(probs=pi_all)
            # gaussian_mixture = distributions.MixtureSameFamily(dis_mix, dis_mu_L)

            # self.loss_nll = -gaussian_mixture.log_prob(self.gt[:,:,:2]).mean()
            
            # lambda_H = self.lambda_H  
            # entropy = -(pi_all * (pi_all+1e-8).log()).sum(-1).mean()
            # self.loss_nll += lambda_H * entropy

            # lambda_std = self.lambda_std
            # penalty_std = F.relu(1e-2 - std).pow(2).mean()
            # self.loss_nll += lambda_std * penalty_std

            # lambda_corr = self.lambda_corr
            # penalty_corr = F.relu(torch.abs(corr) - 0.97).pow(2).mean()
            # self.loss_nll += lambda_corr * penalty_corr

            # #sample the output, multi-times
            # eps_sample = torch.zeros_like(mu)
            # for s_t in range(self.sample_times):
            #     eps_sample += torch.randn_like(mu)
            # eps_sample /= self.sample_times
            
            # posi = mu + (L @ eps_sample.unsqueeze(-1)).squeeze(-1)    
            # posi = (pi_all.unsqueeze(-1) * posi).sum(dim=2)

            # x_input = torch.cat([posi, cos_sin_heading], dim=-1)
            # x_input = x_input * self.rollout_mask  # For future rollouts

            # self.rollout_pos.append(x_input)
            # self.G_pred_std.append(std)
            # self.G_pred_corr.append(corr)
            # self.G_pred_pi.append(pi_all)

            # # ----------------method 2----------------
            # mu, std, corr, cos_sin_heading, pi_all, L = self.net_G(x_input)

            # # construct n_gaussian gaussian, compute -log separately, then get the min to be the nll loss
            # gaussian_1 = distributions.MultivariateNormal(loc=mu[:,:,0,:], scale_tril=L[:,:,0,:,:])
            # gaussian_2 = distributions.MultivariateNormal(loc=mu[:,:,1,:], scale_tril=L[:,:,1,:,:])
            # gaussian_3 = distributions.MultivariateNormal(loc=mu[:,:,2,:], scale_tril=L[:,:,2,:,:])

            # nll_1 = - (torch.log(pi_all[:,:,0]) + gaussian_1.log_prob(self.gt[:,:,:2])).mean()
            # nll_2 = - (torch.log(pi_all[:,:,1]) + gaussian_1.log_prob(self.gt[:,:,:2])).mean()
            # nll_3 = - (torch.log(pi_all[:,:,2]) + gaussian_1.log_prob(self.gt[:,:,:2])).mean()

            # nll_all = torch.stack([nll_1, nll_2, nll_3])
            # nll_min, _ = nll_all.min(dim=0)

            # self.loss_nll = nll_min

            # #sample the output
            # eps = torch.randn_like(mu)
            # posi = mu + (L @ eps.unsqueeze(-1)).squeeze(-1)    
            # posi = (pi_all.unsqueeze(-1) * posi).sum(dim=2)

            # x_input = torch.cat([posi, cos_sin_heading], dim=-1)
            # x_input = x_input * self.rollout_mask  # For future rollouts

            # self.rollout_pos.append(x_input)
            # self.G_pred_std.append(std)
            # self.G_pred_corr.append(corr)
            # self.G_pred_pi.append(pi_all)


            # ----------------method 3: L1_Loss----------------
            if self.epoch_id <= 50:
                mu, std, corr, cos_sin_heading, pi_all, L = self.net_G(x_input,if_mean_grad=True, if_std_grad=False, if_corr_grad=False, if_pi_grad=False)

                # construct n_gaussian gaussian, compute -log separately, then get the min to be the nll loss
                mu_1 = mu[:,:,0,:]
                std_1 = std[:,:,0,:]
                corr_1 = corr[:,:,0].unsqueeze(-1)

                mu_2 = mu[:,:,1,:]
                std_2 = std[:,:,1,:]
                corr_2 = corr[:,:,1].unsqueeze(-1)

                mu_3 = mu[:,:,2,:]
                std_3 = std[:,:,2,:]
                corr_3 = corr[:,:,2].unsqueeze(-1)

                gt_pos, mask_pos = self.gt[:, :, :int(self.output_dim / 2)], self.mask[:, :, :int(self.output_dim / 2)]
                gt_cos_sin_heading, mask_cos_sin_heading = self.gt[:, :, int(self.output_dim / 2):], self.mask[:, :, int(self.output_dim / 2):]

                cos_sin_heading = cos_sin_heading * mask_cos_sin_heading
                self.reg_loss_heading = 20 * self.regression_loss_func_heading(y_pred_mean=cos_sin_heading, y_pred_std=None, y_true=gt_cos_sin_heading, weight=mask_cos_sin_heading)

                # posi_1 = (mu_1 + x_input[-1,:,:int(self.output_dim / 2)]) * self.rollout_mask[:,:,:2]
                # posi_2 = (mu_2 + x_input[-1,:,:int(self.output_dim / 2)]) * self.rollout_mask[:,:,:2]
                # posi_3 = (mu_3 + x_input[-1,:,:int(self.output_dim / 2)]) * self.rollout_mask[:,:,:2]
                posi_1 = mu_1 * self.rollout_mask[:,:,:2]
                posi_2 = mu_2 * self.rollout_mask[:,:,:2]
                posi_3 = mu_3 * self.rollout_mask[:,:,:2]

                L1_mu_1 = self.regression_loss_func_pos(y_pred_mean=posi_1, y_pred_std=None, y_true=gt_pos, weight=mask_pos)
                L1_mu_2 = self.regression_loss_func_pos(y_pred_mean=posi_2, y_pred_std=None, y_true=gt_pos, weight=mask_pos)
                L1_mu_3 = self.regression_loss_func_pos(y_pred_mean=posi_3, y_pred_std=None, y_true=gt_pos, weight=mask_pos)

                L1_mu_all = torch.stack([L1_mu_1, L1_mu_2, L1_mu_3])
                L1_mu_min, idx_mu = L1_mu_all.min(0)

                self.loss_nll = L1_mu_min + self.reg_loss_heading
                #print(self.loss_nll.item())
                if idx_mu == 0:
                    posi = posi_1
                elif idx_mu == 1:
                    posi = posi_2
                else:
                    posi = posi_3

                x_pred = torch.cat([posi, cos_sin_heading], dim=-1)
                x_pred = x_pred * self.rollout_mask

                self.rollout_pos.append(x_pred)
                self.G_pred_std.append(std)
                self.G_pred_corr.append(corr)
                self.G_pred_pi.append(pi_all)

            elif self.epoch_id <= 150:
                mu, std, corr, cos_sin_heading, pi_all, L = self.net_G(x_input,if_mean_grad=False, if_std_grad=True, if_corr_grad=True, if_pi_grad=False)

                # construct n_gaussian gaussian, compute -log separately, then get the min to be the nll loss
                mu_1 = mu[:,:,0,:]
                std_1 = std[:,:,0,:]
                corr_1 = corr[:,:,0].unsqueeze(-1)
                L_1 = L[:,:,0,:,:]
    
                mu_2 = mu[:,:,1,:]
                std_2 = std[:,:,1,:]
                corr_2 = corr[:,:,1].unsqueeze(-1)
                L_2 = L[:,:,1,:,:]
    
                mu_3 = mu[:,:,2,:]
                std_3 = std[:,:,2,:]
                corr_3 = corr[:,:,2].unsqueeze(-1)
                L_3 = L[:,:,2,:,:]
    
                gt_pos, mask_pos = self.gt[:, :, :int(self.output_dim / 2)], self.mask[:, :, :int(self.output_dim / 2)]
                
                #gt_cos_sin_heading, mask_cos_sin_heading = self.gt[:, :, int(self.output_dim / 2):], self.mask[:, :, int(self.output_dim / 2):]
                #self.reg_loss_heading = 20 * self.regression_loss_func_heading(y_pred_mean=cos_sin_heading, y_pred_std=None, y_true=gt_cos_sin_heading, weight=mask_cos_sin_heading)
                
                eps1, eps2 = self._sampling_from_standard_gaussian(mu_1)

                x_input_1 = self._sampling_from_mu_and_std(mu_1, std_1, corr_1, eps1, eps2)
                x_input_2 = self._sampling_from_mu_and_std(mu_2, std_2, corr_2, eps1, eps2)
                x_input_3 = self._sampling_from_mu_and_std(mu_3, std_3, corr_3, eps1, eps2)

                x_input_1 = x_input_1 * self.rollout_mask[:,:,:2]
                x_input_2 = x_input_2 * self.rollout_mask[:,:,:2]
                x_input_3 = x_input_3 * self.rollout_mask[:,:,:2]

                L1_loss_1 = self.regression_loss_func_pos(y_pred_mean=x_input_1, y_pred_std=None, y_true=gt_pos, weight=mask_pos)
                L1_loss_2 = self.regression_loss_func_pos(y_pred_mean=x_input_2, y_pred_std=None, y_true=gt_pos, weight=mask_pos)
                L1_loss_3 = self.regression_loss_func_pos(y_pred_mean=x_input_3, y_pred_std=None, y_true=gt_pos, weight=mask_pos)

                # add NLL Loss for every branch
                mvn_1 = distributions.MultivariateNormal(loc=mu_1, scale_tril=L_1)
                mvn_2 = distributions.MultivariateNormal(loc=mu_2, scale_tril=L_2)
                mvn_3 = distributions.MultivariateNormal(loc=mu_3, scale_tril=L_3)

                weight_nll = 0.01
                nll_loss_1 = -weight_nll * mvn_1.log_prob(self.gt[:,:,:2]).mean()
                nll_loss_2 = -weight_nll * mvn_2.log_prob(self.gt[:,:,:2]).mean()
                nll_loss_3 = -weight_nll * mvn_3.log_prob(self.gt[:,:,:2]).mean()

                # print(f"dis1:{L1_loss_1}, {nll_loss_1}")
                # print(f"dis1:{L1_loss_2}, {nll_loss_2}")
                # print(f"dis1:{L1_loss_3}, {nll_loss_3}")

                loss_all_1 = nll_loss_1 + L1_loss_1
                loss_all_2 = nll_loss_2 + L1_loss_2
                loss_all_3 = nll_loss_3 + L1_loss_3


                L1_loss_all = torch.stack([loss_all_1, loss_all_2, loss_all_3])
                L1_loss_min, idx = L1_loss_all.min(dim=0)

                self.loss_nll = L1_loss_min # + self.reg_loss_heading

                if idx == 0:
                    posi = x_input_1
                elif idx == 1:
                    posi = x_input_2
                else:
                    posi = x_input_3

                x_pred = torch.cat([posi, cos_sin_heading], dim=-1)
                x_pred = x_pred * self.rollout_mask  # For future rollouts

                self.rollout_pos.append(x_pred)
                self.G_pred_std.append(std)
                self.G_pred_corr.append(corr)
                self.G_pred_pi.append(pi_all)

            else:
                mu, std, corr, cos_sin_heading, pi_all, L = self.net_G(x_input, if_mean_grad=False, if_std_grad=False, if_corr_grad=False, if_pi_grad=True)

                dis_mu_L = distributions.MultivariateNormal(loc=mu, scale_tril=L)
                dis_mix = distributions.Categorical(probs=pi_all)
                gaussian_mixture = distributions.MixtureSameFamily(dis_mix, dis_mu_L)

                weight_nll_whole = 0.5 * 1e-4
                self.loss_nll = -weight_nll_whole * gaussian_mixture.log_prob(self.gt[:,:,:2]).mean()

                lambda_H = self.lambda_H  
                entropy = -(pi_all * (pi_all+1e-8).log()).sum(-1).mean()
                self.loss_nll += lambda_H * entropy

                #sample the output, multi-times
                eps_sample = torch.zeros_like(mu)
                # for s_t in range(self.sample_times):
                #     eps_sample += torch.randn_like(mu)
                # eps_sample /= self.sample_times

                posi = mu + (L @ eps_sample.unsqueeze(-1)).squeeze(-1)    
                posi = (pi_all.unsqueeze(-1) * posi).sum(dim=2)

                x_pred = torch.cat([posi, cos_sin_heading], dim=-1)
                x_pred = x_pred * self.rollout_mask  # For future rollouts

                self.rollout_pos.append(x_pred)
                self.G_pred_std.append(std)
                self.G_pred_corr.append(corr)
                self.G_pred_pi.append(pi_all)


    def _compute_acc(self):
        sample_at_stpe0 = self.rollout_pos[0][:, :, :int(self.output_dim / 2)]
        G_pred_cos_sin_heading_at_step0 = self.rollout_pos[0][:, :, int(self.output_dim / 2):]
        std_pred = self.G_pred_std[0] #[batch, 32, 3, 2]
        corr_pred = self.G_pred_corr[0] #[batch, 32, 3]
        pi_pred = self.G_pred_pi[0] #[batch, 32, 3]

        gt_pos, mask_pos = self.gt[:, :, :int(self.output_dim / 2)], self.mask[:, :, :int(self.output_dim / 2)]
        gt_cos_sin_heading, mask_heading = self.gt[:, :, int(self.output_dim / 2):], self.mask[:, :, int(self.output_dim / 2):]

        self.batch_acc_sample = self.accuracy_metric_pos(sample_at_stpe0.detach(), gt_pos.detach(), mask=mask_pos > 0)
        self.batch_acc_heading = self.accuracy_metric_heading(G_pred_cos_sin_heading_at_step0.detach(), gt_cos_sin_heading.detach(), mask=mask_heading > 0)

        self.batch_mean_std_x = []
        self.batch_mean_std_y = []
        self.batch_mean_corr = []
        self.batch_mean_pi = []

        for i in range(self.n_gaussian):
            std_pred_i = std_pred[:,:,i,:]
            std_pred_i = (std_pred_i * mask_pos).detach().cpu().numpy()
            corr_pred_i = ((corr_pred[:,:,i].unsqueeze(-1)) * (self.mask[:, :, 0].unsqueeze(-1))).detach().cpu().numpy()
            pi_pred_i = ((pi_pred[:,:,i].unsqueeze(-1)) * (self.mask[:, :, 0].unsqueeze(-1))).detach().cpu().numpy()

            self.batch_mean_std_x.append(np.mean(std_pred_i[:,:,0]))
            self.batch_mean_std_y.append(np.mean(std_pred_i[:,:,1]))
            self.batch_mean_corr.append(np.mean(corr_pred_i))
            self.batch_mean_pi.append(np.mean(pi_pred_i))

    
    def _compute_loss_G(self, epoch_threshold=50):
        if (self.epoch_id + 1) < epoch_threshold:
            self.batch_loss_G = self.loss_nll
        else: 
            sample_at_step0 = self.rollout_pos[0]

            G_pred_pos_at_step0, G_pred_cos_sin_heading_at_step0 = sample_at_step0[:, :, :int(self.output_dim / 2)], sample_at_step0[:, :, int(self.output_dim / 2):]

            gt_pos, mask_pos = self.gt[:, :, :int(self.output_dim / 2)], self.mask[:, :, :int(self.output_dim / 2)]
            gt_cos_sin_heading, mask_cos_sin_heading = self.gt[:, :, int(self.output_dim / 2):], self.mask[:, :, int(self.output_dim / 2):]

            self.reg_loss_position = self.regression_loss_func_pos(y_pred_mean=sample_at_step0[:, :, :int(self.output_dim / 2)], y_pred_std=None, y_true=gt_pos, weight=mask_pos)
            self.reg_loss_heading = 20 * self.regression_loss_func_heading(y_pred_mean=G_pred_cos_sin_heading_at_step0, y_pred_std=None, y_true=gt_cos_sin_heading, weight=mask_cos_sin_heading)

            # D_pred_fake = self.net_D(sample_at_step0)

            # # Filter out ghost vehicles
            # # Reformat the size into num x n, where num = bs * m_token - ghost vehicles (also for those have missing values in gt)
            # ghost_vehicles_mask = (torch.sum(self.mask, dim=-1) == 4).flatten()  # bs * m_token.
            # D_pred_fake_filtered = (D_pred_fake.flatten())[ghost_vehicles_mask]

            # self.G_adv_loss = 0.1*self.gan_loss(D_pred_fake_filtered, True)

            self.batch_loss_G = self.loss_nll + self.lambda_reg_posi*self.reg_loss_position + self.lambda_reg_angle*self.reg_loss_heading # + 0.5*self.G_adv_loss




    def _compute_loss_D(self):
        # print(f"self.G_pred_mean[0]:{self.G_pred_mean[0].shape}")
        # print(f"self.gt:{self.gt.shape}")

        D_pred_fake = self.net_D(self.rollout_pos[0].detach())
        D_pred_real = self.net_D(self.gt.detach())

        # print(f"D_pred_fake:{D_pred_fake.shape}")
        # print(f"D_pred_real:{D_pred_real.shape}")

        # Filter out ghost vehicles
        pred_fake_ghost_vehicles_mask = (torch.sum(self.mask, dim=-1) == 4).flatten()  # bs * m_token.
        D_pred_fake_filtered = (D_pred_fake.flatten())[pred_fake_ghost_vehicles_mask]

        pred_real_ghost_vehicles_mask = (torch.sum(self.mask, dim=-1) == 4).flatten()  # bs * m_token.
        D_pred_real_filtered = (D_pred_real.flatten())[pred_real_ghost_vehicles_mask]

        # print(f"D_pred_fake_filtered:{D_pred_fake_filtered.shape}")
        # print(f"D_pred_real_filtered:{D_pred_real_filtered.shape}")

        D_adv_loss_fake = self.gan_loss(D_pred_fake_filtered, False)
        D_adv_loss_real = self.gan_loss(D_pred_real_filtered, True)
        self.D_adv_loss = 0.5 * (D_adv_loss_fake + D_adv_loss_real)

        self.batch_loss_D = self.D_adv_loss
        # print(f"D_adv_loss_fake:{D_adv_loss_fake}")
        # print(f"D_adv_loss_real:{D_adv_loss_real}")

    def _backward_G(self):
        self.batch_loss_G.backward()
        # print(f'------------{self.epoch_id, self.batch_id}------------')
        # for n, p in self.net_G.named_parameters():
        #     if p.grad is None:
        #         print(f"{n}:grad None")
        #     else:
        #         print(f"{n}:{p.grad.norm():.4f}")

    def _backward_D(self):
        self.batch_loss_D.backward()

    def _clear_cache(self):
        self.running_acc_pos, self.running_acc_heading, self.running_acc_sample = [], [], []
        self.running_loss_G, self.running_loss_D = [], []
        self.running_loss_G_reg_pos, self.running_loss_G_reg_heading, self.running_loss_G_adv = [], [], []
        self.running_std_x, self.running_std_y, self.running_corr, self.running_pi = [],[],[],[]

    def train_models(self):
        """
        Main training loop. We loop over the dataset multiple times.
        In each minibatch, we perform gradient descent on the network parameters.
        """

        self._load_checkpoint(start_id=-1)
        # loop over the dataset multiple times
        for self.epoch_id in range(self.epoch_to_start, self.max_num_epochs):
            # ################## train #################
            # ##########################################
            self._clear_cache()
            self.is_training = True
            self.net_G.train()  # Set model to training mode
            # Iterate over data.
            for self.batch_id, batch in enumerate(self.dataloaders['train'], 0):
                self._forward_pass(batch, rollout=1)

                # # update D
                # set_requires_grad(self.net_D, True)
                # self.optimizer_D.zero_grad()
                # self._compute_loss_D()
                # self._backward_D()
                # self.optimizer_D.step()
                
                # update G
                set_requires_grad(self.net_D, False)
                self.optimizer_G.zero_grad()
                self._compute_loss_G(epoch_threshold=self.epoch_thres)
                self._backward_G()
                self.optimizer_G.step()

                # evaluate acc
                self._compute_acc()

                self._collect_running_batch_states()
                self._update_logfile_batch_train()
                # if self.batch_id == 6:
                #     break
            self._collect_epoch_states()

            # ################## Eval ##################
            ##########################################
            print('Begin evaluation...')
            self._clear_cache()
            self.is_training = False
            self.net_G.eval()  # Set model to eval mode

            # Iterate over data.
            for self.batch_id, batch in enumerate(self.dataloaders['val'], 0):
                with torch.no_grad():
                    self._forward_pass(batch, rollout=1)
                    self._compute_loss_G()
                    #self._compute_loss_D()
                    self._compute_acc()
                self._collect_running_batch_states()
            self._collect_epoch_states()

            ########### Update_Checkpoints ###########
            ##########################################
            if (self.epoch_id + 1) <= 200:
                if (self.epoch_id + 1) % 5 == 0:
                    self._update_checkpoints(epoch_id=self.epoch_id)
            elif (self.epoch_id + 1) <= 500:
                if (self.epoch_id + 1) % 10 == 0:
                    self._update_checkpoints(epoch_id=self.epoch_id)
            else:
                if (self.epoch_id + 1) % 50 == 0:
                    self._update_checkpoints(epoch_id=self.epoch_id)

            ########### Update_LR Scheduler ##########
            ##########################################
            self._update_lr_schedulers() #注意加上D之后这里要改！！！

            ############## Update logfile ############
            ##########################################
            self._update_logfile()
            self._visualize_logfile()

            ########### Visualize Prediction #########
            ##########################################
            # self._visualize_prediction()

        self.writer.close()