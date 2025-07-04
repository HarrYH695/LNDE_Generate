import numpy as np
import matplotlib.pyplot as plt
import os
import json
import random

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from .networks import define_G, define_D, set_requires_grad, define_safety_mapper, Network_G
from .loss import UncertaintyRegressionLoss, GANLoss
from .metric import RegressionAccuracy
from . import utils

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Trainer(object):
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

        self.rollout_num = configs["rollout_num"]

        # initialize networks
        # self.net_G = define_G(
        #     configs["model"], input_dim=self.input_dim, output_dim=self.output_dim,
        #     m_tokens=self.m_tokens).to(device)
        self.net_G = Network_G(configs["model"], input_dim=self.input_dim, output_dim=self.output_dim,
            m_tokens=self.m_tokens).to(device)
        self.net_D = define_D(input_dim=self.rollout_num).to(device)

        # initialize safety mapping networks to involve it in the training loop
        self.net_safety_mapper = define_safety_mapper(configs["safety_mapper_ckpt_dir"], self.m_tokens).to(device)
        self.net_safety_mapper.eval()
        set_requires_grad(self.net_safety_mapper, requires_grad=False)

        # Learning rate
        self.lr = configs["lr"]

        # define optimizers
        self.optimizer_G = optim.RMSprop(self.net_G.parameters(), lr=self.lr)
        self.optimizer_D = optim.RMSprop(self.net_D.parameters(), lr=self.lr)

        # define lr schedulers
        self.exp_lr_scheduler_G = lr_scheduler.StepLR(
            self.optimizer_G, step_size=configs["lr_decay_step_size"], gamma=configs["lr_decay_gamma"])
        self.exp_lr_scheduler_D = lr_scheduler.StepLR(
            self.optimizer_D, step_size=configs["lr_decay_step_size"], gamma=configs["lr_decay_gamma"])

        # define loss function and error metric
        self.regression_loss_func_pos = UncertaintyRegressionLoss(choice='mae_c')
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
        self.running_std_x, self.running_std_y, self.running_corr = [], [], []
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
        self.batch_mean_std_x = 0.0
        self.batch_mean_std_y = 0.0
        self.batch_mean_corr = 0.0
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
        self.exp_lr_scheduler_D.step()

    def _collect_running_batch_states(self):

        # self.running_acc_pos.append(self.batch_acc_pos.item())
        # self.running_acc_sample.append(self.batch_acc_sample.item())
        # self.running_acc_heading.append(self.batch_acc_heading.item())
        # self.running_loss_G.append(self.batch_loss_G.item())
        # self.running_loss_D.append(self.batch_loss_D.item())
        # self.running_loss_G_reg_pos.append(self.reg_loss_position.item())
        # self.running_loss_G_reg_heading.append(self.reg_loss_heading.item())
        # self.running_loss_G_adv.append(self.G_adv_loss.item())
        # self.running_std_x.append(self.batch_mean_std_x.item())
        # self.running_std_y.append(self.batch_mean_std_y.item())
        # self.running_corr.append(self.batch_mean_corr.item())


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



        self.running_acc_pos.append(self.batch_acc_pos.item())
        #self.running_acc_sample.append(self.batch_acc_sample.item())
        self.running_acc_heading.append(self.batch_acc_heading.item())
        self.running_loss_G.append(self.batch_loss_G.item())
        self.running_loss_D.append(self.batch_loss_D.item())
        self.running_loss_G_reg_pos.append(self.reg_loss_position.item())
        self.running_loss_G_reg_heading.append(self.reg_loss_heading.item())
        self.running_loss_G_adv.append(self.G_adv_loss.item())

        # self.running_std_x.append(self.batch_mean_std_x.item())
        # self.running_std_y.append(self.batch_mean_std_y.item())

        # if self.is_training:
        #     m_batches = len(self.dataloaders['train'])
        # else:
        #     m_batches = len(self.dataloaders['val'])

        # if np.mod(self.batch_id, 100) == 1:
        #     print('Is_training: %s. epoch [%d,%d], batch [%d,%d], reg_loss_pos: %.5f, reg_loss_heading: %.5f, '
        #           'G_adv_loss: %.5f, D_adv_loss: %.5f, running_acc_pos: %.5f, running_acc_heading: %.5f'
        #           % (self.is_training, self.epoch_id, self.max_num_epochs-1, self.batch_id, m_batches,
        #              self.reg_loss_position.item(), self.reg_loss_heading.item(),
        #              self.G_adv_loss.item(), self.D_adv_loss.item(), np.mean(self.running_acc_pos),
        #              np.mean(self.running_acc_heading)))


    def _collect_epoch_states(self):
        # if self.is_training:
        #     self.train_epoch_acc_pos = np.mean(self.running_acc_pos).item()
        #     self.train_epoch_acc_sample = np.mean(self.running_acc_sample).item()
        #     self.train_epoch_acc_heading = np.mean(self.running_acc_heading).item()
        #     self.train_epoch_loss_G = np.mean(self.running_loss_G).item()
        #     self.train_epoch_loss_D = np.mean(self.running_loss_D).item()
        #     self.train_epoch_loss_G_reg_pos = np.mean(self.running_loss_G_reg_pos).item()
        #     self.train_epoch_loss_G_reg_heading = np.mean(self.running_loss_G_reg_heading).item()
        #     self.train_epoch_loss_G_adv = np.mean(self.running_loss_G_adv).item()
        #     self.train_epoch_std_x = np.mean(self.running_std_x).item()
        #     self.train_epoch_std_y = np.mean(self.running_std_y).item()
        #     self.train_epoch_corr = np.mean(self.running_corr).item()

        #     # print('Training, Epoch %d / %d, epoch_loss_G= %.5f, epoch_loss_D= %.5f, epoch_acc_pos= %.5f, epoch_acc_sample= %.5f, epoch_acc_heading= %.5f' %
        #     #       (self.epoch_id, self.max_num_epochs-1,
        #     #        self.train_epoch_loss_G, self.train_epoch_loss_D, self.train_epoch_acc_pos, self.train_epoch_acc_sample, self.train_epoch_acc_heading))

        #     print(f"train_epoch_std_x:{self.train_epoch_std_x}, train_epoch_std_y:{self.train_epoch_std_y}, train_epoch_corr:{self.train_epoch_corr}")

        #     self.writer.add_scalar("Accuracy/train_pos", self.train_epoch_acc_pos, self.epoch_id)
        #     self.writer.add_scalar("Accuracy/train_sample", self.train_epoch_acc_sample, self.epoch_id)
        #     self.writer.add_scalar("Accuracy/train_heading", self.train_epoch_acc_heading, self.epoch_id)
        #     self.writer.add_scalar("Loss/train_G", self.train_epoch_loss_G, self.epoch_id)
        #     self.writer.add_scalar("Loss/train_D", self.train_epoch_loss_D, self.epoch_id)
        #     self.writer.add_scalar("Loss/train_G_reg_position", self.train_epoch_loss_G_reg_pos, self.epoch_id)
        #     self.writer.add_scalar("Loss/train_G_reg_heading", self.train_epoch_loss_G_reg_heading, self.epoch_id)
        #     self.writer.add_scalar("Loss/train_G_adv", self.train_epoch_loss_G_adv, self.epoch_id)
        #     self.writer.add_scalar("Std/std_x_train", self.train_epoch_std_x, self.epoch_id)
        #     self.writer.add_scalar("Std/std_y_train", self.train_epoch_std_y, self.epoch_id)
        #     self.writer.add_scalar("Std/corr_train", self.train_epoch_corr, self.epoch_id)

        # else:
        #     self.val_epoch_acc_pos = np.mean(self.running_acc_pos).item()
        #     self.val_epoch_acc_sample = np.mean(self.running_acc_sample).item()
        #     self.val_epoch_acc_heading = np.mean(self.running_acc_heading).item()
        #     self.val_epoch_loss_G = np.mean(self.running_loss_G).item()
        #     self.val_epoch_loss_D = np.mean(self.running_loss_D).item()
        #     self.val_epoch_loss_G_reg_pos = np.mean(self.running_loss_G_reg_pos).item()
        #     self.val_epoch_loss_G_reg_heading = np.mean(self.running_loss_G_reg_heading).item()
        #     self.val_epoch_loss_G_adv = np.mean(self.running_loss_G_adv).item()
        #     self.val_epoch_std_x = np.mean(self.running_std_x).item()
        #     self.val_epoch_std_y = np.mean(self.running_std_y).item()
        #     self.val_epoch_corr = np.mean(self.running_corr).item()

        #     # print('Validation, Epoch %d / %d, epoch_loss_G= %.5f, epoch_loss_D= %.5f, epoch_acc_pos= %.5f, epoch_acc_sample= %.5f, epoch_acc_heading= %.5f' %
        #     #       (self.epoch_id, self.max_num_epochs - 1,
        #     #        self.val_epoch_loss_G, self.val_epoch_loss_D, self.val_epoch_acc_pos, self.val_epoch_acc_sample, self.val_epoch_acc_heading))

        #     print(f"val_epoch_std_x:{self.val_epoch_std_x}, val_epoch_std_y:{self.val_epoch_std_y}, val_epoch_corr:{self.val_epoch_corr}")

        #     self.writer.add_scalar("Accuracy/val_pos", self.val_epoch_acc_pos, self.epoch_id)
        #     self.writer.add_scalar("Accuracy/val_sample", self.val_epoch_acc_sample, self.epoch_id)
        #     self.writer.add_scalar("Accuracy/val_heading", self.val_epoch_acc_heading, self.epoch_id)
        #     self.writer.add_scalar("Loss/val_G", self.val_epoch_loss_G, self.epoch_id)
        #     self.writer.add_scalar("Loss/val_D", self.val_epoch_loss_D, self.epoch_id)
        #     self.writer.add_scalar("Loss/val_G_reg_position", self.val_epoch_loss_G_reg_pos, self.epoch_id)
        #     self.writer.add_scalar("Loss/val_G_reg_heading", self.val_epoch_loss_G_reg_heading, self.epoch_id)
        #     self.writer.add_scalar("Loss/val_G_adv", self.val_epoch_loss_G_adv, self.epoch_id)
        #     self.writer.add_scalar("Std/std_x_val", self.val_epoch_std_x, self.epoch_id)
        #     self.writer.add_scalar("Std/std_y_val", self.val_epoch_std_y, self.epoch_id)
        #     self.writer.add_scalar("Std/corr_val", self.val_epoch_corr, self.epoch_id)
        # print()


        if self.is_training:
            self.train_epoch_acc_pos = np.mean(self.running_acc_pos).item()
            self.train_epoch_acc_heading = np.mean(self.running_acc_heading).item()
            self.train_epoch_loss_G = np.mean(self.running_loss_G).item()
            self.train_epoch_loss_D = np.mean(self.running_loss_D).item()
            self.train_epoch_loss_G_reg_pos = np.mean(self.running_loss_G_reg_pos).item()
            self.train_epoch_loss_G_reg_heading = np.mean(self.running_loss_G_reg_heading).item()
            self.train_epoch_loss_G_adv = np.mean(self.running_loss_G_adv).item()
            self.train_epoch_std_x = np.mean(self.running_std_x).item()
            self.train_epoch_std_y = np.mean(self.running_std_y).item()
            #self.train_epoch_acc_sample = np.mean(self.running_acc_sample).item()

            print('Training, Epoch %d / %d, epoch_loss_G= %.5f, epoch_loss_D= %.5f, epoch_acc_pos= %.5f, epoch_acc_heading= %.5f' %
                  (self.epoch_id, self.max_num_epochs-1,
                   self.train_epoch_loss_G, self.train_epoch_loss_D, self.train_epoch_acc_pos, self.train_epoch_acc_heading))


            #self.writer.add_scalar("Accuracy/train_sample", self.train_epoch_acc_sample, self.epoch_id)
            self.writer.add_scalar("Accuracy/train_pos", self.train_epoch_acc_pos, self.epoch_id)
            self.writer.add_scalar("Accuracy/train_heading", self.train_epoch_acc_heading, self.epoch_id)
            self.writer.add_scalar("Loss/train_G", self.train_epoch_loss_G, self.epoch_id)
            self.writer.add_scalar("Loss/train_D", self.train_epoch_loss_D, self.epoch_id)
            self.writer.add_scalar("Loss/train_G_reg_position", self.train_epoch_loss_G_reg_pos, self.epoch_id)
            self.writer.add_scalar("Loss/train_G_reg_heading", self.train_epoch_loss_G_reg_heading, self.epoch_id)
            self.writer.add_scalar("Loss/train_G_adv", self.train_epoch_loss_G_adv, self.epoch_id)
            self.writer.add_scalar("Std/std_x_train", self.train_epoch_std_x, self.epoch_id)
            self.writer.add_scalar("Std/std_y_train", self.train_epoch_std_y, self.epoch_id)

        else:
            self.val_epoch_acc_pos = np.mean(self.running_acc_pos).item()
            self.val_epoch_acc_heading = np.mean(self.running_acc_heading).item()
            self.val_epoch_loss_G = np.mean(self.running_loss_G).item()
            self.val_epoch_loss_D = np.mean(self.running_loss_D).item()
            self.val_epoch_loss_G_reg_pos = np.mean(self.running_loss_G_reg_pos).item()
            self.val_epoch_loss_G_reg_heading = np.mean(self.running_loss_G_reg_heading).item()
            self.val_epoch_loss_G_adv = np.mean(self.running_loss_G_adv).item()
            self.val_epoch_std_x = np.mean(self.running_std_x).item()
            self.val_epoch_std_y = np.mean(self.running_std_y).item()
            #self.val_epoch_acc_sample = np.mean(self.running_acc_sample).item()

            print('Validation, Epoch %d / %d, epoch_loss_G= %.5f, epoch_loss_D= %.5f, epoch_acc_pos= %.5f, epoch_acc_heading= %.5f' %
                  (self.epoch_id, self.max_num_epochs - 1,
                   self.val_epoch_loss_G, self.val_epoch_loss_D, self.val_epoch_acc_pos, self.val_epoch_acc_heading))

            #self.writer.add_scalar("Accuracy/val_sample", self.val_epoch_acc_sample, self.epoch_id)
            self.writer.add_scalar("Accuracy/val_pos", self.val_epoch_acc_pos, self.epoch_id)
            self.writer.add_scalar("Accuracy/val_heading", self.val_epoch_acc_heading, self.epoch_id)
            self.writer.add_scalar("Loss/val_G", self.val_epoch_loss_G, self.epoch_id)
            self.writer.add_scalar("Loss/val_D", self.val_epoch_loss_D, self.epoch_id)
            self.writer.add_scalar("Loss/val_G_reg_position", self.val_epoch_loss_G_reg_pos, self.epoch_id)
            self.writer.add_scalar("Loss/val_G_reg_heading", self.val_epoch_loss_G_reg_heading, self.epoch_id)
            self.writer.add_scalar("Loss/val_G_adv", self.val_epoch_loss_G_adv, self.epoch_id)
            self.writer.add_scalar("Std/std_x_val", self.val_epoch_std_x, self.epoch_id)
            self.writer.add_scalar("Std/std_y_val", self.val_epoch_std_y, self.epoch_id)
        print()


    def _update_checkpoints(self, epoch_id):

        # save current model
        self._save_checkpoint(ckpt_name=f'ckpt_{epoch_id}.pt')
        sum_val_epoch_acc = self.val_epoch_acc_pos + self.val_epoch_acc_heading
        print('Latest model updated. Epoch_acc_pos=%.4f, Epoch_acc_heading=%.4f,'
              'Sum Epoch_acc=%.4f'
              % (self.val_epoch_acc_pos, self.val_epoch_acc_heading, sum_val_epoch_acc))
        print()

    def _update_logfile(self):

        logfile_path = os.path.join(self.checkpoint_dir, 'logfile.json')

        # read historical logfile and update
        if os.path.exists(logfile_path):
            with open(logfile_path) as json_file:
                self.logfile = json.load(json_file)

        self.logfile['train_acc_pos'].append(self.train_epoch_acc_pos)
        self.logfile['train_acc_heading'].append(self.train_epoch_acc_heading)
        self.logfile['val_acc_pos'].append(self.val_epoch_acc_pos)
        self.logfile['val_acc_heading'].append(self.val_epoch_acc_heading)
        self.logfile['train_loss_G'].append(self.train_epoch_loss_G)
        self.logfile['val_loss_G'].append(self.val_epoch_loss_G)
        self.logfile['train_loss_D'].append(self.train_epoch_loss_D)
        self.logfile['val_loss_D'].append(self.val_epoch_loss_D)
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

        self.logfile_batch['train_loss_G_pos'].append(self.reg_loss_position.item())
        self.logfile_batch['train_loss_G_ang'].append(self.reg_loss_heading.item())
        self.logfile_batch['train_loss_G_adv'].append(self.G_adv_loss.item())
        self.logfile_batch['train_loss_D'].append(self.D_adv_loss.item())
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
        plt.plot(self.logfile['epochs'], self.logfile['train_acc_pos'])
        plt.plot(self.logfile['epochs'], self.logfile['train_acc_heading'])
        plt.plot(self.logfile['epochs'], self.logfile['val_acc_pos'])
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

    def _sampling_from_mu_and_std(self, mu, std, heading_mu, corr):
        bs, _, _ = mu.shape #[bs, 32, 2]
        epsilons_lat = torch.tensor([random.gauss(0, 1) for _ in range(bs * self.m_tokens)]).to(device).detach()
        epsilons_lon = torch.tensor([random.gauss(0, 1) for _ in range(bs * self.m_tokens)]).to(device).detach()
        epsilons_1 = torch.reshape(epsilons_lat, [bs, -1, 1])
        epsilons_2 = torch.reshape(epsilons_lon, [bs, -1, 1])

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

        return lat, lon, heading_mu[:,:,0:self.pred_length], heading_mu[:,:,self.pred_length:]


        # bs, _, _ = mu.shape #[bs, 32, 2]
        # epsilons_lat = torch.tensor([random.gauss(0, 1) for _ in range(bs * self.m_tokens)]).to(device).detach()
        # epsilons_lon = torch.tensor([random.gauss(0, 1) for _ in range(bs * self.m_tokens)]).to(device).detach()
        # epsilons_1 = torch.reshape(epsilons_lat, [bs, -1, 1])
        # epsilons_2 = torch.reshape(epsilons_lon, [bs, -1, 1])

        # #[bs, 32, 1]
        # lat_mean = mu[:, :, 0:self.pred_length]
        # lon_mean = mu[:, :, self.pred_length:]
        # lat_std = std[:, :, 0:self.pred_length]
        # lon_std = std[:, :, self.pred_length:]
        # # lat_std = torch.sqrt(std[:, :, 0:self.pred_length])
        # # lon_std = torch.sqrt(std[:, :, self.pred_length:])

        # lat = lat_mean + lat_std * epsilons_1
        # lon = lon_mean + lon_std * epsilons_1 * corr + lon_std * torch.sqrt(1 - corr ** 2) * epsilons_2
        # # print(f"lat:{lat.shape}")
        # # print(f"lon:{lon.shape}")

        # return torch.cat([lat, lon, heading_mu], dim=-1)


        # bs, _, _ = mu.shape
        # epsilons_lat = torch.tensor([random.gauss(0, 1) for _ in range(bs * self.m_tokens)]).to(device).detach()
        # epsilons_lon = torch.tensor([random.gauss(0, 1) for _ in range(bs * self.m_tokens)]).to(device).detach()
        # epsilons_lat = torch.reshape(epsilons_lat, [bs, -1, 1])
        # epsilons_lon = torch.reshape(epsilons_lon, [bs, -1, 1])

        # lat_mean = mu[:, :, 0:self.pred_length]
        # lon_mean = mu[:, :, self.pred_length:]
        # lat_std = std[:, :, 0:self.pred_length]
        # lon_std = std[:, :, self.pred_length:]

        # lat = lat_mean + epsilons_lat * lat_std
        # lon = lon_mean + epsilons_lon * lon_std

        # return torch.cat([lat, lon, heading_mu], dim=-1)


    def _forward_pass(self, batch, rollout=5):

        self.batch = batch
        self.x = batch['input'].to(device)
        self.gt = self.batch['gt'].to(device)
        print(f"self.x:{self.x.shape}")
        print(f"self.gt:{self.gt.shape}")

        self.mask = torch.ones_like(self.gt)
        self.mask[torch.isnan(self.gt)] = 0.0
        self.rollout_mask = torch.ones_like(self.gt)
        self.rollout_mask[torch.isnan(self.gt)] = 0.0
        self.rollout_mask[torch.sum(self.mask, dim=-1) > 0, :] = 1.0

        self.x[torch.isnan(self.x)] = 0.0
        self.gt[torch.isnan(self.gt)] = 0.0

        x_input = self.x
        self.G_pred_mean_lat, self.G_pred_mean_lon, self.G_pred_std, self.G_pred_corr = [], [], [], []
        self.rollout_pos = []
        self.G_pred_cos, self.G_pred_sin = [], []

        # new forward, with joint gaussian, multi-rollout
        for roll_i in range(rollout):
            mu, std, corr, cos_sin_heading = self.net_G(x_input, if_mean_grad=True, if_std_grad=False, if_corr_grad=False)
            # print(f"mu:{mu.shape}")
            # print(f"std:{std.shape}")
            # print(f"corr:{corr.shape}")
            # print(f"cos_sin_heading:{cos_sin_heading.shape}")

            pred_lat, pred_lon, pred_cos, pred_sin = self._sampling_from_mu_and_std(mu, std, cos_sin_heading, corr)
            
            pred_lat = pred_lat * (self.rollout_mask[:,:,roll_i].unsqueeze(-1))
            pred_lon = pred_lon * (self.rollout_mask[:,:,self.rollout_num + roll_i].unsqueeze(-1))
            pred_cos = pred_cos * (self.rollout_mask[:,:,self.rollout_num * 2 + roll_i].unsqueeze(-1))
            pred_sin = pred_sin * (self.rollout_mask[:,:,self.rollout_num * 3 + roll_i].unsqueeze(-1))

            print(f"pred_lat:{pred_lat.shape}")
            print(f"pred_lon:{pred_lon.shape}")
            print(f"pred_cos:{pred_cos.shape}")
            print(f"pred_sin:{pred_sin.shape}")

            lat_old = x_input[:,:,1:5]
            lon_old = x_input[:,:,6:10]
            cos_old = x_input[:,:,11:15]
            sin_old = x_input[:,:,16:20]

            x_input = torch.cat([lat_old, pred_lat, lon_old, pred_lon, cos_old, pred_cos, sin_old, pred_sin], dim=-1)

            print(f"x_input_roll_{roll_i}:{x_input.shape}")

            self.G_pred_mean_lat.append(pred_lat)
            self.G_pred_mean_lon.append(pred_lon)
            self.G_pred_cos.append(pred_cos)
            self.G_pred_sin.append(pred_sin)

        pred_traj = self.G_pred_mean_lat[0]
        for i in range(1, self.rollout_num):
            pred_traj = torch.cat([pred_traj, self.G_pred_mean_lat[i]])

        for i in range(self.rollout_num):
            pred_traj = torch.cat([pred_traj, self.G_pred_mean_lon[i]])

        for i in range(self.rollout_num):
            pred_traj = torch.cat([pred_traj, self.G_pred_cos[i]])

        for i in range(self.rollout_num):
            pred_traj = torch.cat([pred_traj, self.G_pred_sin[i]])
        
        self.rollout_pos.append(pred_traj)
            
            #print(f"x_input:{x_input.shape}")

        # # new forward, with joint gaussian
        # for _ in range(rollout):
        #     mu, std, corr, cos_sin_heading = self.net_G(x_input)
        #     # print(f"mu:{mu.shape}")
        #     # print(f"std:{std.shape}")
        #     # print(f"corr:{corr.shape}")
        #     # print(f"cos_sin_heading:{cos_sin_heading.shape}")

        #     x_input = self._sampling_from_mu_and_std(mu, std, cos_sin_heading, corr)
        #     #print(f"x_input:{x_input.shape}")
        #     x_input = x_input * self.rollout_mask

        #     self.rollout_pos.append(x_input)
        #     self.G_pred_mean.append(torch.cat([mu, cos_sin_heading], dim=-1))
        #     self.G_pred_std.append(std)
        #     self.G_pred_corr.append(corr)
            
            #print(f"x_input:{x_input.shape}")


        # #old forward, no joint gaussian
        # for _ in range(rollout):
        #     mu, std, cos_sin_heading = self.net_G(x_input)
        #     # print(f"mu:{mu.shape}")
        #     # print(f"std:{std.shape}")
        #     # print(f"cos_sin_heading:{cos_sin_heading.shape}")

        #     x_input = self._sampling_from_mu_and_std(mu, std, cos_sin_heading)
        #     # print(f"x_input:{x_input.shape}")
        #     # print(f"self.rollout_mask:{self.rollout_mask.shape}")
        #     x_input = x_input * self.rollout_mask  # For future rollouts
        #     #x_input = self.net_safety_mapper.safety_mapper_in_the_training_loop_forward(x_input) #no safety_mapping
        #     self.rollout_pos.append(x_input)
        #     self.G_pred_mean.append(torch.cat([mu, cos_sin_heading], dim=-1))
        #     self.G_pred_std.append(std)

        #     #print(f"x_input:{x_input.shape}")


    def _compute_acc(self):
        G_pred_mean_at_step0 = self.G_pred_mean[0]

        G_pred_mean_pos_at_step0 = G_pred_mean_at_step0[:, :, :int(self.output_dim / 2)]
        sample_at_stpe0 = self.rollout_pos[0][:, :, :int(self.output_dim / 2)]
        G_pred_cos_sin_heading_at_step0 = G_pred_mean_at_step0[:, :, int(self.output_dim / 2):]

        gt_pos, mask_pos = self.gt[:, :, :int(self.output_dim / 2)], self.mask[:, :, :int(self.output_dim / 2)]
        gt_cos_sin_heading, mask_heading = self.gt[:, :, int(self.output_dim / 2):], self.mask[:, :, int(self.output_dim / 2):]

        self.batch_acc_pos = self.accuracy_metric_pos(G_pred_mean_pos_at_step0.detach(), gt_pos.detach(), mask=mask_pos > 0)
        self.batch_acc_heading = self.accuracy_metric_heading(G_pred_cos_sin_heading_at_step0.detach(), gt_cos_sin_heading.detach(), mask=mask_heading > 0)
        self.batch_acc_sample = self.accuracy_metric_pos(sample_at_stpe0.detach(), gt_pos.detach(), mask=mask_pos > 0)

        G_pred_std0 = self.G_pred_std[0] * mask_pos
        G_pred_corr0 = self.G_pred_corr[0] * (self.mask[:, :, 0].unsqueeze(-1))

        self.batch_mean_std_x = torch.mean(G_pred_std0[:,:,0])
        self.batch_mean_std_y = torch.mean(G_pred_std0[:,:,1])
        self.batch_mean_corr = torch.mean(G_pred_corr0)



        # G_pred_mean_at_step0 = self.G_pred_mean[0]
        # G_pred_mean_pos_at_step0 = G_pred_mean_at_step0[:, :, :int(self.output_dim / 2)]
        # G_pred_cos_sin_heading_at_step0 = G_pred_mean_at_step0[:, :, int(self.output_dim / 2):]

        # gt_pos, mask_pos = self.gt[:, :, :int(self.output_dim / 2)], self.mask[:, :, :int(self.output_dim / 2)]
        # gt_cos_sin_heading, mask_heading = self.gt[:, :, int(self.output_dim / 2):], self.mask[:, :, int(self.output_dim / 2):]

        # self.batch_acc_pos = self.accuracy_metric_pos(G_pred_mean_pos_at_step0.detach(), gt_pos.detach(), mask=mask_pos > 0)
        # self.batch_acc_heading = self.accuracy_metric_heading(G_pred_cos_sin_heading_at_step0.detach(), gt_cos_sin_heading.detach(), mask=mask_heading > 0)

        # G_pred_std0 = self.G_pred_std[0] * mask_pos
        # self.batch_mean_std_x = torch.mean(G_pred_std0[:,:,0])
        # self.batch_mean_std_y = torch.mean(G_pred_std0[:,:,1])

        # sample_at_stpe0 = self.rollout_pos[0][:, :, :int(self.output_dim / 2)]
        # self.batch_acc_sample = self.accuracy_metric_pos(sample_at_stpe0.detach(), gt_pos.detach(), mask=mask_pos > 0)


    def _compute_loss_G(self):
        # reg loss (between pred0 and gt)
        G_pred_mean_at_step0 = self.G_pred_mean[0]
        #G_pred_std_at_step0 = self.G_pred_std[0]
        sample_at_step0 = self.rollout_pos[0]

        G_pred_pos_at_step0, G_pred_cos_sin_heading_at_step0 = G_pred_mean_at_step0[:, :, :int(self.output_dim / 2)], G_pred_mean_at_step0[:, :, int(self.output_dim / 2):]
        #G_pred_std_pos_at_step0 = G_pred_std_at_step0

        gt_pos, mask_pos = self.gt[:, :, :int(self.output_dim / 2)], self.mask[:, :, :int(self.output_dim / 2)]
        gt_cos_sin_heading, mask_cos_sin_heading = self.gt[:, :, int(self.output_dim / 2):], self.mask[:, :, int(self.output_dim / 2):]

        self.reg_loss_position = self.regression_loss_func_pos(y_pred_mean=sample_at_step0[:, :, :int(self.output_dim / 2)], y_pred_std=None, y_true=gt_pos, weight=mask_pos)
        self.reg_loss_heading = 20 * self.regression_loss_func_heading(y_pred_mean=G_pred_cos_sin_heading_at_step0, y_pred_std=None, y_true=gt_cos_sin_heading, weight=mask_cos_sin_heading)

        D_pred_fake = self.net_D(sample_at_step0)
        #print(f"D_pred_fake:{D_pred_fake.shape}")
        # Filter out ghost vehicles
        # Reformat the size into num x n, where num = bs * m_token - ghost vehicles (also for those have missing values in gt)
        ghost_vehicles_mask = (torch.sum(self.mask, dim=-1) == 4*self.rollout_num).flatten()  # bs * m_token.
        D_pred_fake_filtered = (D_pred_fake.flatten())[ghost_vehicles_mask]
        #print(f"D_pred_fake_filtered:{D_pred_fake_filtered.shape}")
        self.G_adv_loss = 0.1*self.gan_loss(D_pred_fake_filtered, True)

        self.batch_loss_G = self.reg_loss_position + self.reg_loss_heading + self.G_adv_loss
        #print(self.reg_loss_position.item(), self.reg_loss_heading.item(), self.G_adv_loss.item(), self.batch_loss_G.item())



        # # reg loss (between pred0 and gt)
        # G_pred_mean_at_step0 = self.G_pred_mean[0]
        # #G_pred_std_at_step0 = self.G_pred_std[0]
        # sample_at_step0 = self.rollout_pos[0]


        # G_pred_pos_at_step0, G_pred_cos_sin_heading_at_step0 = G_pred_mean_at_step0[:, :, :int(self.output_dim / 2)], G_pred_mean_at_step0[:, :, int(self.output_dim / 2):]
        # #G_pred_std_pos_at_step0 = G_pred_std_at_step0

        # gt_pos, mask_pos = self.gt[:, :, :int(self.output_dim / 2)], self.mask[:, :, :int(self.output_dim / 2)]
        # gt_cos_sin_heading, mask_cos_sin_heading = self.gt[:, :, int(self.output_dim / 2):], self.mask[:, :, int(self.output_dim / 2):]

        # self.reg_loss_position = self.regression_loss_func_pos(y_pred_mean=sample_at_step0[:, :, :int(self.output_dim / 2)], y_pred_std=None, y_true=gt_pos, weight=mask_pos)
        # self.reg_loss_heading = 20 * self.regression_loss_func_heading(y_pred_mean=G_pred_cos_sin_heading_at_step0, y_pred_std=None, y_true=gt_cos_sin_heading, weight=mask_cos_sin_heading)

        # D_pred_fake = self.net_D(sample_at_step0)
        # #print(f"D_pred_fake:{D_pred_fake.shape}")
        # # Filter out ghost vehicles
        # # Reformat the size into num x n, where num = bs * m_token - ghost vehicles (also for those have missing values in gt)
        # ghost_vehicles_mask = (torch.sum(self.mask, dim=-1) == 4).flatten()  # bs * m_token.
        # D_pred_fake_filtered = (D_pred_fake.flatten())[ghost_vehicles_mask]
        # #print(f"D_pred_fake_filtered:{D_pred_fake_filtered.shape}")
        # self.G_adv_loss = 0.1*self.gan_loss(D_pred_fake_filtered, True)

        # self.batch_loss_G = self.reg_loss_position + self.reg_loss_heading + self.G_adv_loss
        # #print(self.reg_loss_position.item(), self.reg_loss_heading.item(), self.G_adv_loss.item(), self.batch_loss_G.item())



        # # reg loss (between pred0 and gt)
        # G_pred_mean_at_step0 = self.G_pred_mean[0]
        # G_pred_std_at_step0 = self.G_pred_std[0]

        # G_pred_pos_at_step0, G_pred_cos_sin_heading_at_step0 = G_pred_mean_at_step0[:, :, :int(self.output_dim / 2)], G_pred_mean_at_step0[:, :, int(self.output_dim / 2):]
        # G_pred_std_pos_at_step0 = G_pred_std_at_step0

        # gt_pos, mask_pos = self.gt[:, :, :int(self.output_dim / 2)], self.mask[:, :, :int(self.output_dim / 2)]
        # gt_cos_sin_heading, mask_cos_sin_heading = self.gt[:, :, int(self.output_dim / 2):], self.mask[:, :, int(self.output_dim / 2):]

        # self.reg_loss_position = self.regression_loss_func_pos(G_pred_pos_at_step0, G_pred_std_pos_at_step0, gt_pos, weight=mask_pos)
        # self.reg_loss_heading = 20 * self.regression_loss_func_heading(y_pred_mean=G_pred_cos_sin_heading_at_step0, y_pred_std=None, y_true=gt_cos_sin_heading, weight=mask_cos_sin_heading)

        # D_pred_fake = self.net_D(self.G_pred_mean[0])
        # # Filter out ghost vehicles
        # # Reformat the size into num x n, where num = bs * m_token - ghost vehicles (also for those have missing values in gt)
        # ghost_vehicles_mask = (torch.sum(self.mask, dim=-1) == 4).flatten()  # bs * m_token.
        # D_pred_fake_filtered = (D_pred_fake.flatten())[ghost_vehicles_mask]
        # self.G_adv_loss = 0.1*self.gan_loss(D_pred_fake_filtered, True)

        # self.batch_loss_G = self.reg_loss_position + self.reg_loss_heading + self.G_adv_loss



        # # reg loss (in trial 2, only joint Gaussian)
        # G_pred_mean_at_step0 = self.G_pred_mean[0]
        # G_pred_std_at_step0 = self.G_pred_std[0]
        # G_pred_corr_at_step0 = self.G_pred_corr[0]

        # G_pred_pos_at_step0, G_pred_cos_sin_heading_at_step0 = G_pred_mean_at_step0[:, :, :int(self.output_dim / 2)], G_pred_mean_at_step0[:, :, int(self.output_dim / 2):]
        # G_pred_std_pos_at_step0 = G_pred_std_at_step0

        # gt_pos, mask_pos = self.gt[:, :, :int(self.output_dim / 2)], self.mask[:, :, :int(self.output_dim / 2)]
        # gt_cos_sin_heading, mask_cos_sin_heading = self.gt[:, :, int(self.output_dim / 2):], self.mask[:, :, int(self.output_dim / 2):]

        # self.reg_loss_position = self.regression_loss_func_pos(G_pred_pos_at_step0, G_pred_std_pos_at_step0, gt_pos, weight=mask_pos, pred_corr=G_pred_corr_at_step0)
        # self.reg_loss_heading = 20 * self.regression_loss_func_heading(y_pred_mean=G_pred_cos_sin_heading_at_step0, y_pred_std=None, y_true=gt_cos_sin_heading, weight=mask_cos_sin_heading)

        # D_pred_fake = self.net_D(self.G_pred_mean[0])
        # # Filter out ghost vehicles
        # # Reformat the size into num x n, where num = bs * m_token - ghost vehicles (also for those have missing values in gt)
        # ghost_vehicles_mask = (torch.sum(self.mask, dim=-1) == 4).flatten()  # bs * m_token.
        # D_pred_fake_filtered = (D_pred_fake.flatten())[ghost_vehicles_mask]
        # self.G_adv_loss = 0.1*self.gan_loss(D_pred_fake_filtered, True)

        # self.batch_loss_G = self.reg_loss_position + self.reg_loss_heading + self.G_adv_loss

    def _compute_loss_D(self):
        # print(f"self.G_pred_mean[0]:{self.G_pred_mean[0].shape}")
        # print(f"self.gt:{self.gt.shape}")

        D_pred_fake = self.net_D(self.rollout_pos[0].detach())
        D_pred_real = self.net_D(self.gt.detach())

        # print(f"D_pred_fake:{D_pred_fake.shape}")
        # print(f"D_pred_real:{D_pred_real.shape}")

        # Filter out ghost vehicles
        pred_fake_ghost_vehicles_mask = (torch.sum(self.mask, dim=-1) == 4*self.rollout_num).flatten()  # bs * m_token.
        D_pred_fake_filtered = (D_pred_fake.flatten())[pred_fake_ghost_vehicles_mask]

        pred_real_ghost_vehicles_mask = (torch.sum(self.mask, dim=-1) == 4*self.rollout_num).flatten()  # bs * m_token.
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

    def _backward_D(self):
        self.batch_loss_D.backward()

    def _clear_cache(self):
        self.running_acc_pos, self.running_acc_heading, self.running_acc_sample = [], [], []
        self.running_loss_G, self.running_loss_D = [], []
        self.running_loss_G_reg_pos, self.running_loss_G_reg_heading, self.running_loss_G_adv = [], [], []
        self.running_std_x, self.running_std_y, self.running_corr = [],[],[]

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
                self._forward_pass(batch, rollout=self.rollout_num)
                break
            
            break
            #     # update D
            #     set_requires_grad(self.net_D, True)
            #     self.optimizer_D.zero_grad()
            #     self._compute_loss_D()
            #     self._backward_D()
            #     self.optimizer_D.step()
                
            #     # update G
            #     set_requires_grad(self.net_D, False)
            #     self.optimizer_G.zero_grad()
            #     self._compute_loss_G()
            #     self._backward_G()
            #     self.optimizer_G.step()
            #     # evaluate acc
            #     self._compute_acc()

            #     self._collect_running_batch_states()
            #     self._update_logfile_batch_train()
            #     # if self.batch_id == 5:
            #     #     break
            # self._collect_epoch_states()


            # # ################## Eval ##################
            # ##########################################
            # print('Begin evaluation...')
            # self._clear_cache()
            # self.is_training = False
            # self.net_G.eval()  # Set model to eval mode

            # # Iterate over data.
            # for self.batch_id, batch in enumerate(self.dataloaders['val'], 0):
            #     with torch.no_grad():
            #         self._forward_pass(batch, rollout=self.rollout_num)
            #         self._compute_loss_G()
            #         self._compute_loss_D()
            #         self._compute_acc()
            #     self._collect_running_batch_states()
            #     # if self.batch_id == 5:
            #     #     break
            # self._collect_epoch_states()

            # ########### Update_Checkpoints ###########
            # ##########################################
            # if (self.epoch_id + 1) <= 200:
            #     if (self.epoch_id + 1) % 5 == 0:
            #         self._update_checkpoints(epoch_id=self.epoch_id)
            # elif (self.epoch_id + 1) <= 500:
            #     if (self.epoch_id + 1) % 10 == 0:
            #         self._update_checkpoints(epoch_id=self.epoch_id)
            # else:
            #     if (self.epoch_id + 1) % 20 == 0:
            #         self._update_checkpoints(epoch_id=self.epoch_id)

            # ########### Update_LR Scheduler ##########
            # ##########################################
            # self._update_lr_schedulers()

            # ############## Update logfile ############
            # ##########################################
            # self._update_logfile()
            # self._visualize_logfile()

            # ########### Visualize Prediction #########
            # ##########################################
            # # self._visualize_prediction()

        self.writer.close()

