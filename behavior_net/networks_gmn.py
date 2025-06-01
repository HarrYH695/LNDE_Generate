import torch
import torch.nn as nn
import torch.nn.functional as F
from .bert import Embeddings, Block, Config
from safety_mapping.safety_mapping_networks import define_safety_mapping_networks, load_pretrained_weights

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def define_G(model, input_dim, output_dim, m_tokens, n_gaussian=3):

    h_dim = 256

    # define input positional mapping
    M = PositionalMapping(input_dim=input_dim, L=4, pos_scale=0.01, heading_scale=1.0)

    # define backbone networks
    if model == 'simple_mlp':
        Backbone = SimpleMLP(input_dim=M.output_dim, h_dim=h_dim, m_tokens=m_tokens)
    elif model == 'bn_mlp':
        Backbone = BnMLP(input_dim=M.output_dim, h_dim=h_dim, m_tokens=m_tokens)
    elif model == 'transformer':
        bert_cfg = Config()
        bert_cfg.dim = h_dim
        bert_cfg.n_layers = 4
        bert_cfg.n_heads = 4
        bert_cfg.max_len = m_tokens
        Backbone = Transformer(input_dim=M.output_dim, m_tokens=m_tokens, cfg=bert_cfg)
    else:
        raise NotImplementedError(
            'Wrong backbone model name %s (choose one from [simple_mlp, bn_mlp, transformer])' % model)

    # define prediction heads
    P = PredictionsHeads(h_dim=h_dim, output_dim=output_dim, n_gaussian=n_gaussian)

    return nn.Sequential(M, Backbone, P)

class PositionalMapping(nn.Module):
    """
    Positional mapping Layer.
    This layer map continuous input coordinates into a higher dimensional space
    and enable the prediction to more easily approximate a higher frequency function.
    See NERF paper for more details (https://arxiv.org/pdf/2003.08934.pdf)
    """

    def __init__(self, input_dim, L=5, pos_scale=0.01, heading_scale=1.0):
        super(PositionalMapping, self).__init__()
        self.L = L
        self.output_dim = input_dim * (L*2 + 1)
        # self.scale = scale
        self.pos_scale = pos_scale
        self.heading_scale = heading_scale

    def forward(self, x):

        if self.L == 0:
            return x

        # x = x * self.scale
        x_scale = x.clone()
        x_scale[:, :, :10] = x_scale[:, :, :10] * self.pos_scale
        x_scale[:, :, 10:] = x_scale[:, :, 10:] * self.heading_scale

        h = [x]
        PI = 3.1415927410125732
        for i in range(self.L):
            x_sin = torch.sin(2**i * PI * x_scale)
            x_cos = torch.cos(2**i * PI * x_scale)

            x_sin[:, :, :10], x_sin[:, :, 10:] = x_sin[:, :, :10] / self.pos_scale, x_sin[:, :, 10:] / self.heading_scale
            x_cos[:, :, :10], x_cos[:, :, 10:] = x_cos[:, :, :10] / self.pos_scale, x_cos[:, :, 10:] / self.heading_scale

            h.append(x_sin)
            h.append(x_cos)

        # return torch.cat(h, dim=-1) / self.scale
        return torch.cat(h, dim=-1)


class PredictionsHeads(nn.Module):
    """
    Prediction layer with two output heads, one modeling mean and another one modeling std.
    Also prediction cos and sin headings.
    """

    def __init__(self, h_dim, output_dim, n_gaussian, tem_softmax=1.3, corr_limit=0.99):
        super().__init__()
        # number of gaussian
        self.n_gaussian = n_gaussian
        self.tem_softmax = tem_softmax
        self.corr_limit = corr_limit

        # x, y position
        self.out_net_mean = nn.Linear(in_features=h_dim, out_features=int(output_dim * self.n_gaussian / 2), bias=True)
        self.out_net_std = nn.Linear(in_features=h_dim, out_features=int(output_dim * self.n_gaussian / 2), bias=True)
        self.out_net_corr = nn.Linear(in_features=h_dim, out_features=self.n_gaussian, bias=True)

        self.elu = torch.nn.ELU()
        self.tanh = torch.nn.Tanh()
        self.softplus = torch.nn.Softplus()
        self.softmax = torch.nn.Softmax(dim=-1)
        
        # cos and sin heading
        #self.out_net_cos_sin_heading = nn.Linear(in_features=h_dim, out_features=int(output_dim * self.n_gaussian / 2), bias=True)
        self.out_net_cos_sin_heading = nn.Linear(in_features=h_dim, out_features=int(output_dim / 2), bias=True)

        # pi_i, i = range(n_gaussian)
        self.out_pi = nn.Linear(in_features=h_dim, out_features=n_gaussian, bias=True)

    def forward(self, x, if_mean_grad=True, if_std_grad=True, if_corr_grad=True, if_pi_grad=True):

        # shape x: batch_size x m_token x m_state
        # go through the network
        if not if_mean_grad:
            with torch.no_grad():
                out_mean = self.out_net_mean(x)
        else:
            out_mean = self.out_net_mean(x)

        if not if_std_grad:
            with torch.no_grad():
                out_std_raw = self.out_net_std(x)
        else:
            out_std_raw = self.out_net_std(x)

        if not if_corr_grad:
            with torch.no_grad():
                out_corr_raw = self.out_net_corr(x)
        else:
            out_corr_raw = self.out_net_corr(x)

        if not if_pi_grad:
            with torch.no_grad():
                out_pi_raw = self.out_pi(x)
        else:
            out_pi_raw = self.out_pi(x)
        

        out_cos_sin_heading = self.out_net_cos_sin_heading(x)


        out_pi = self.softmax(out_pi_raw / self.tem_softmax)  #[B, N=32, 3]
        out_corr = self.tanh(out_corr_raw) * self.corr_limit #[B, N=32, 3]
        out_std = torch.minimum(F.softplus(out_std_raw) + 1e-3, torch.full_like(out_std_raw, 0.5))
        #out_std = self.elu(out_std_raw) + 1
        #out_std = out_std_raw ** 2
        #out_std = self.softplus(out_std_raw)

        # change shape into (batch, 32, n_gaussian, 2)
        B, N, C = out_mean.shape
        out_mean = out_mean.view(B, N, self.n_gaussian, 2) #[B, 32, 3, 2]
        out_std = out_std.view(B, N, self.n_gaussian, 2)
        #out_cos_sin_heading = out_cos_sin_heading.view(B, N, self.n_gaussian, 2)

        #calculate L where cov = L @ L^T [B, 32, 3, 2, 2]
        #L = [std_x, 0; corr*std_y, sqrt(1-corr**2)*std_y]
        std_x = out_std[:,:,:,0]
        std_y = out_std[:,:,:,1]
        row1 = torch.stack([std_x, torch.zeros_like(std_x)], dim=-1)
        row2 = torch.stack([out_corr*std_y, torch.sqrt(1-out_corr**2)*std_y], dim=-1)
        out_L = torch.stack([row1, row2], dim=-2)

        return out_mean, out_std, out_corr, out_cos_sin_heading, out_pi, out_L




class Transformer(nn.Module):
    """ Transformer with Self-Attentive Blocks"""

    def __init__(self, input_dim, m_tokens, cfg=Config()):
        super().__init__()
        self.in_net = nn.Linear(input_dim, cfg.dim, bias=True)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.m_tokens = m_tokens

    def forward(self, x):
        # shape x: batch_size x m_token x m_state
        h = self.in_net(x)
        for block in self.blocks:
            h = block(h, None)
        return h

class Network_G(nn.Module):
    def __init__(self, model, input_dim, output_dim, m_tokens, n_gaussian=3):
        super().__init__()

        h_dim = 256

        # define input positional mapping
        self.M = PositionalMapping(input_dim=input_dim, L=4, pos_scale=0.01, heading_scale=1.0)

        # define backbone networks
        if model == 'simple_mlp':
            self.Backbone = SimpleMLP(input_dim=M.output_dim, h_dim=h_dim, m_tokens=m_tokens)
        elif model == 'bn_mlp':
            self.Backbone = BnMLP(input_dim=M.output_dim, h_dim=h_dim, m_tokens=m_tokens)
        elif model == 'transformer':
            bert_cfg = Config()
            bert_cfg.dim = h_dim
            bert_cfg.n_layers = 4
            bert_cfg.n_heads = 4
            bert_cfg.max_len = m_tokens
            self.Backbone = Transformer(input_dim=M.output_dim, m_tokens=m_tokens, cfg=bert_cfg)
        else:
            raise NotImplementedError(
                'Wrong backbone model name %s (choose one from [simple_mlp, bn_mlp, transformer])' % model)

        # define prediction heads
        self.P = PredictionsHeads(h_dim=h_dim, output_dim=output_dim, n_gaussian=n_gaussian)

        # self.if_mean_grad = True
        # self.if_std_grad=True
        # self.if_corr_grad=True 
        # self.if_pi_grad=True

    def forward(self, x, if_mean_grad=True, if_std_grad=True, if_corr_grad=True, if_pi_grad=True):
        x = self.M(x)
        x = self.Backbone(x)

        out_mean, out_std, out_corr, out_cos_sin_heading, out_pi, out_L = self.P(x, if_mean_grad=True, if_std_grad=True, if_corr_grad=True, if_pi_grad=True)

        return out_mean, out_std, out_corr, out_cos_sin_heading, out_pi, out_L



def define_D(input_dim):

    # define input positional mapping
    M = PositionalMapping(input_dim=input_dim, L=4, pos_scale=0.01, heading_scale=1.0)
    D = Discriminator(input_dim=M.output_dim)

    return nn.Sequential(M, D)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()

        self.mlp = nn.Sequential(
            # input is batch x m_token x input_dim
            nn.Linear(input_dim, 1024, bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512, bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256, bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1, bias=True)
        )

    def forward(self, input):
        return self.mlp(input)