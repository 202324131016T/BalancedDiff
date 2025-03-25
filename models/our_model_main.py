import shutil
import sys
import time

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_mean
from tqdm.auto import tqdm

from models.common import compose_context, ShiftedSoftplus
from models.egnn import EGNN
from models.utils import get_TSNE
from mamba_ssm import Mamba
from kan import KAN


from models.uni_transformer import UniTransformerO2TwoUpdateGeneral


def my_mamba(dim=128, d_state=16, d_conv=4, expand=2):
    # input: (N, L, C)
    my_mamba = Mamba(
        # This module uses roughly 3 * expand * d_model^2 parameters
        d_model=dim,  # Model dimension d_model
        d_state=16,  # SSM state expansion factor
        d_conv=4,  # Local convolution width
        expand=2,  # Block expansion factor
    )
    # output: (N, L, C)
    return my_mamba


def get_refine_net(refine_net_type, config):
    # print("get_refine_net")
    # h_all,  # protein_v+vt[N*E_protein+N*E_ligand, 128]
    # pos_all # protein_x+xt[N*E_protein+N*E_ligand, 3]

    if refine_net_type == 'uni_o2': # True
        refine_net = UniTransformerO2TwoUpdateGeneral(
            num_blocks=config.num_blocks,
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            n_heads=config.n_heads,
            k=config.knn,
            edge_feat_dim=config.edge_feat_dim,
            num_r_gaussian=config.num_r_gaussian,
            num_node_types=config.num_node_types,
            act_fn=config.act_fn,
            norm=config.norm,
            cutoff_mode=config.cutoff_mode,
            ew_net_type=config.ew_net_type,
            num_x2h=config.num_x2h,
            num_h2x=config.num_h2x,
            r_max=config.r_max,
            x2h_out_fc=config.x2h_out_fc,
            sync_twoup=config.sync_twoup
        )
    elif refine_net_type == 'egnn':
        refine_net = EGNN(
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            edge_feat_dim=config.edge_feat_dim,
            num_r_gaussian=1,
            k=config.knn,
            cutoff_mode=config.cutoff_mode
        )
    else:
        raise ValueError(refine_net_type)
    return refine_net


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])

    alphas = np.clip(alphas, a_min=0.001, a_max=1.)

    # Use sqrt of this, so the alpha in our paper is the alpha_sqrt from the
    # Gaussian diffusion in Ho et al.
    alphas = np.sqrt(alphas)
    return alphas


def get_distance(pos, edge_index):
    return (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)


def to_torch_const(x):
    x = torch.from_numpy(x).float()
    x = nn.Parameter(x, requires_grad=False)
    return x


def center_pos(protein_pos, ligand_pos, batch_protein, batch_ligand, mode='protein'):
    if mode == 'none':
        offset = 0.
        pass
    elif mode == 'protein':
        offset = scatter_mean(protein_pos, batch_protein, dim=0) # get center position
        protein_pos = protein_pos - offset[batch_protein] # get 相对位置
        ligand_pos = ligand_pos - offset[batch_ligand]
    else:
        raise NotImplementedError
    return protein_pos, ligand_pos, offset


# %% categorical diffusion related
def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    # permute_order = (0, -1) + tuple(range(1, len(x.size())))
    # x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x


def log_onehot_to_index(log_x):
    return log_x.argmax(1)


def categorical_kl(log_prob1, log_prob2):
    kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
    return kl


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    kl = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + (mean1 - mean2) ** 2 * torch.exp(-logvar2))
    return kl.sum(-1)


def log_normal(values, means, log_scales):
    var = torch.exp(log_scales * 2)
    log_prob = -((values - means) ** 2) / (2 * var) - log_scales - np.log(np.sqrt(2 * np.pi))
    return log_prob.sum(-1)


def log_sample_categorical(logits):
    uniform = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    sample_index = (gumbel_noise + logits).argmax(dim=-1)
    # sample_onehot = F.one_hot(sample, self.num_classes)
    # log_sample = index_to_log_onehot(sample, self.num_classes)
    return sample_index


def log_1_min_a(a):
    return np.log(1 - np.exp(a) + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    am = torch.exp(a - maximum)
    bm = torch.exp(b - maximum)
    return maximum + torch.log(am + bm)


# %%


# Time embedding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# Model
class ScorePosNet3D(nn.Module):

    def __init__(self, config, protein_atom_feature_dim, ligand_atom_feature_dim):
        super().__init__()

        path = os.path.abspath(sys.modules[UniTransformerO2TwoUpdateGeneral.__module__].__file__)
        model_path = path.split('/')[-1]
        shutil.copyfile(path, os.path.join(config.log_dir, model_path))

        self.config = config

        # variance schedule
        self.model_mean_type = config.model_mean_type  # ['noise', 'C0']
        self.loss_v_weight = config.loss_v_weight
        # self.v_mode = config.v_mode
        # assert self.v_mode == 'categorical'
        # self.v_net_type = getattr(config, 'v_net_type', 'mlp')
        # self.bond_loss = getattr(config, 'bond_loss', False)
        # self.bond_net_type = getattr(config, 'bond_net_type', 'pre_att')
        # self.loss_bond_weight = getattr(config, 'loss_bond_weight', 0.)
        # self.loss_non_bond_weight = getattr(config, 'loss_non_bond_weight', 0.)

        self.sample_time_method = config.sample_time_method  # ['importance', 'symmetric']
        # self.loss_pos_type = config.loss_pos_type  # ['mse', 'kl']
        # print(f'Loss pos mode {self.loss_pos_type} applied!')
        # print(f'Loss bond net type: {self.bond_net_type} '
        #       f'bond weight: {self.loss_bond_weight} non bond weight: {self.loss_non_bond_weight}')

        if config.beta_schedule == 'cosine':
            alphas = cosine_beta_schedule(config.num_diffusion_timesteps, config.pos_beta_s) ** 2
            # print('cosine pos alpha schedule applied!')
            betas = 1. - alphas
        else:
            betas = get_beta_schedule(
                beta_schedule=config.beta_schedule,
                beta_start=config.beta_start,
                beta_end=config.beta_end,
                num_diffusion_timesteps=config.num_diffusion_timesteps,
            )
            alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.betas = to_torch_const(betas)
        self.num_timesteps = self.betas.size(0)
        self.alphas_cumprod = to_torch_const(alphas_cumprod)
        self.alphas_cumprod_prev = to_torch_const(alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = to_torch_const(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_torch_const(np.sqrt(1. - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = to_torch_const(np.sqrt(1. / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = to_torch_const(np.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_c0_coef = to_torch_const(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.posterior_mean_ct_coef = to_torch_const(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_var = to_torch_const(posterior_variance)
        self.posterior_logvar = to_torch_const(np.log(np.append(self.posterior_var[1], self.posterior_var[1:])))

        # atom type diffusion schedule in log space
        if config.v_beta_schedule == 'cosine':
            alphas_v = cosine_beta_schedule(self.num_timesteps, config.v_beta_s)
            # print('cosine v alpha schedule applied!')
        else:
            raise NotImplementedError
        log_alphas_v = np.log(alphas_v)
        log_alphas_cumprod_v = np.cumsum(log_alphas_v)
        self.log_alphas_v = to_torch_const(log_alphas_v)
        self.log_one_minus_alphas_v = to_torch_const(log_1_min_a(log_alphas_v))
        self.log_alphas_cumprod_v = to_torch_const(log_alphas_cumprod_v)
        self.log_one_minus_alphas_cumprod_v = to_torch_const(log_1_min_a(log_alphas_cumprod_v))

        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))

        # model definition
        self.hidden_dim = config.hidden_dim
        self.num_classes = ligand_atom_feature_dim
        if self.config.node_indicator:
            emb_dim = self.hidden_dim - 1
        else:
            emb_dim = self.hidden_dim

        # atom embedding
        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, emb_dim)

        # center pos
        self.center_pos_mode = config.center_pos_mode  # ['none', 'protein']

        # time embedding
        self.time_emb_dim = config.time_emb_dim
        self.time_emb_mode = config.time_emb_mode  # ['simple', 'sin']
        if self.time_emb_dim > 0:
            if self.time_emb_mode == 'simple':
                self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + 1, emb_dim)
            elif self.time_emb_mode == 'sin':
                self.time_emb = nn.Sequential(
                    SinusoidalPosEmb(self.time_emb_dim),
                    nn.Linear(self.time_emb_dim, self.time_emb_dim * 4),
                    nn.GELU(),
                    nn.Linear(self.time_emb_dim * 4, self.time_emb_dim)
                )
                self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + self.time_emb_dim, emb_dim)
            else:
                raise NotImplementedError
        else:
            self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim, emb_dim)

        self.refine_net_type = config.model_type
        self.refine_net = get_refine_net(self.refine_net_type, config)
        self.v_inference = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, ligand_atom_feature_dim)
        )

        # deal xyz to weight
        self.mlp_x = nn.Sequential(
            nn.Linear(128+3, 32),
            ShiftedSoftplus(),
            nn.Linear(32, 64),
            ShiftedSoftplus(),
            nn.Linear(64, 128)
        )
        # KAN
        C_in = 64
        self.kan_fc_in = nn.Linear(128, C_in)
        self.kan = KAN(width=[C_in, C_in])
        self.kan_fc_out = nn.Linear(C_in, 128)
        # KAN


    def forward(self, protein_pos, protein_v, batch_protein, init_ligand_pos, init_ligand_v, batch_ligand,
                time_step=None, return_all=False, fix_x=False):

        # print("forward")
        # protein_pos,           # [N*E, 3]  protein_x
        # protein_v,             # [N*E, 27] protein_v
        # ligand_pos_perturbed,  # [N*E, 3]  xt
        # ligand_v_perturbed,    # [N*E, ]   vt
        # time_step=time_step    # [N, ]     time_step
        batch_size = batch_protein.max().item() + 1
        init_ligand_v = F.one_hot(init_ligand_v, self.num_classes).float()  # vt[N*E, 13]
        # time embedding
        if self.time_emb_dim > 0:
            if self.time_emb_mode == 'simple':
                input_ligand_feat = torch.cat([
                    init_ligand_v,
                    (time_step / self.num_timesteps)[batch_ligand].unsqueeze(-1)
                ], -1)
            elif self.time_emb_mode == 'sin':
                time_feat = self.time_emb(time_step)
                input_ligand_feat = torch.cat([init_ligand_v, time_feat], -1)
            else:
                raise NotImplementedError
        else:
            input_ligand_feat = init_ligand_v  # vt[N*E, 13]

        h_protein = self.protein_atom_emb(protein_v) # use fc to embedding  # protein_v[N*E, 127]
        init_ligand_h = self.ligand_atom_emb(input_ligand_feat) # use fc to embedding  # vt[N*E, 127]

        if self.config.node_indicator:
            # protein feature
            h_protein = torch.cat([h_protein, torch.zeros(len(h_protein), 1).to(h_protein)], -1) # embedding 128-1 再加0 to 128 # protein_v[N*E, 128]
            # ligand feature
            init_ligand_h = torch.cat([init_ligand_h, torch.ones(len(init_ligand_h), 1).to(h_protein)], -1) # embedding 128-1 再加1 to 128 # vt[N*E, 128]

        h_all, pos_all, batch_all, mask_ligand = compose_context(
            h_protein=h_protein,          # protein_v[N*E, 128]
            h_ligand=init_ligand_h,       # vt[N*E, 128]
            pos_protein=protein_pos,      # protein_x[N*E, 3]
            pos_ligand=init_ligand_pos,   # vt[N*E, 3]
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
        )

        # h_all,  # protein_v+vt[N*E_protein+N*E_ligand, 128]
        # pos_all # protein_x+xt[N*E_protein+N*E_ligand, 3]

        h_all_R = h_all
        h_all = self.kan_fc_in(h_all)
        h_all = self.kan(h_all)
        h_all = self.kan_fc_out(h_all)
        h_all = h_all * self.mlp_x(torch.cat([h_all_R, pos_all], dim=1)) + h_all_R

        if "EGNN" in str(self.refine_net):
            outputs = self.refine_net(h_all, pos_all, mask_ligand, batch_all, return_all=return_all)
        else:
            outputs = self.refine_net(h_all, pos_all, mask_ligand, batch_all, return_all=return_all, fix_x=fix_x)
        final_pos, final_h = outputs['x'], outputs['h']  # position and feature pre_x0[N*E_all,3] pre_v0[N*E_all,128]
        final_ligand_pos, final_ligand_h = final_pos[mask_ligand], final_h[mask_ligand] # ligand position and ligand feature pre_x0[N*E,3] pre_v0[N*E,128]
        final_ligand_v = self.v_inference(final_ligand_h)  # ligand feature 128 to 13 pre_v0[N*E,13]
        # ligand position and ligand feature pre_x0[N*E,3] pre_v0[N*E,13]
        preds = {
            'pred_ligand_pos': final_ligand_pos, # pre_x0[N*E,3]
            'pred_ligand_v': final_ligand_v, # pre_v0[N*E,13]
            'final_h': final_h, # all feature 128
            'final_ligand_h': final_ligand_h # pre_v0[N*E,128]
        }
        if return_all:
            final_all_pos, final_all_h = outputs['all_x'], outputs['all_h']
            final_all_ligand_pos = [pos[mask_ligand] for pos in final_all_pos]
            final_all_ligand_v = [self.v_inference(h[mask_ligand]) for h in final_all_h]
            preds.update({
                'layer_pred_ligand_pos': final_all_ligand_pos,
                'layer_pred_ligand_v': final_all_ligand_v
            })
        return preds

    # atom type diffusion process
    def q_v_pred_one_timestep(self, log_vt_1, t, batch):
        # q(vt | vt-1)
        log_alpha_t = extract(self.log_alphas_v, t, batch)
        log_1_min_alpha_t = extract(self.log_one_minus_alphas_v, t, batch)

        # alpha_t * vt + (1 - alpha_t) 1 / K
        log_probs = log_add_exp(
            log_vt_1 + log_alpha_t,
            log_1_min_alpha_t - np.log(self.num_classes)
        )
        return log_probs # get vt from vt-1

    def q_v_pred(self, log_v0, t, batch): # NE, 13
        # compute q(vt | v0) add noise t:step_time
        log_cumprod_alpha_t = extract(self.log_alphas_cumprod_v, t, batch) # alpha_t
        log_1_min_cumprod_alpha = extract(self.log_one_minus_alphas_cumprod_v, t, batch) # 1-alpha_t
        # alpha_t * v0 + (1 - alpha_t) 1 / K
        log_probs = log_add_exp(
            log_v0 + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - np.log(self.num_classes)
        )
        return log_probs # get vt from v0

    def q_v_sample(self, log_v0, t, batch): # NE, 13
        log_qvt_v0 = self.q_v_pred(log_v0, t, batch) # get vt by v0 and K
        sample_index = log_sample_categorical(log_qvt_v0) # [N*E, ]
        log_sample = index_to_log_onehot(sample_index, self.num_classes) # [N*E, 13]
        return sample_index, log_sample # get vt

    # atom type generative process
    def q_v_posterior(self, log_v0, log_vt, t, batch):
        # q(vt-1 | vt, v0) = q(vt | vt-1, x0) * q(vt-1 | x0) / q(vt | x0)
        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_qvt1_v0 = self.q_v_pred(log_v0, t_minus_1, batch)
        # q_v_pred add noise for t-1
        unnormed_logprobs = log_qvt1_v0 + self.q_v_pred_one_timestep(log_vt, t, batch) # vt-1 + vt+1
        log_vt1_given_vt_v0 = unnormed_logprobs - torch.logsumexp(unnormed_logprobs, dim=-1, keepdim=True)
        return log_vt1_given_vt_v0

    def kl_v_prior(self, log_x_start, batch):
        num_graphs = batch.max().item() + 1
        log_qxT_prob = self.q_v_pred(log_x_start, t=[self.num_timesteps - 1] * num_graphs, batch=batch)
        log_half_prob = -torch.log(self.num_classes * torch.ones_like(log_qxT_prob))
        kl_prior = categorical_kl(log_qxT_prob, log_half_prob)
        kl_prior = scatter_mean(kl_prior, batch, dim=0)
        return kl_prior

    def _predict_x0_from_eps(self, xt, eps, t, batch):
        pos0_from_e = extract(self.sqrt_recip_alphas_cumprod, t, batch) * xt - \
                      extract(self.sqrt_recipm1_alphas_cumprod, t, batch) * eps
        return pos0_from_e

    def q_pos_posterior(self, x0, xt, t, batch):
        # Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        pos_model_mean = extract(self.posterior_mean_c0_coef, t, batch) * x0 + \
                         extract(self.posterior_mean_ct_coef, t, batch) * xt
        return pos_model_mean

    def kl_pos_prior(self, pos0, batch):
        num_graphs = batch.max().item() + 1
        a_pos = extract(self.alphas_cumprod, [self.num_timesteps - 1] * num_graphs, batch)  # (num_ligand_atoms, 1)
        pos_model_mean = a_pos.sqrt() * pos0
        pos_log_variance = torch.log((1.0 - a_pos).sqrt())
        kl_prior = normal_kl(torch.zeros_like(pos_model_mean), torch.zeros_like(pos_log_variance),
                             pos_model_mean, pos_log_variance)
        kl_prior = scatter_mean(kl_prior, batch, dim=0)
        return kl_prior

    def sample_time(self, num_graphs, device, method='symmetric'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(num_graphs, device, method='symmetric')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            time_step = torch.multinomial(pt_all, num_samples=num_graphs, replacement=True)
            pt = pt_all.gather(dim=0, index=time_step)
            return time_step, pt

        elif method == 'symmetric':
            time_step = torch.randint(
                0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=device) # [0, num_timesteps=1000]之间的bs//2+1个随机数
            time_step = torch.cat(
                [time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]
            pt = torch.ones_like(time_step).float() / self.num_timesteps
            return time_step, pt # bs 个 随机数，bs 个 1/num_timesteps=1000

        else:
            raise ValueError

    def compute_pos_Lt(self, pos_model_mean, x0, xt, t, batch):
        # fixed pos variance
        pos_log_variance = extract(self.posterior_logvar, t, batch)
        pos_true_mean = self.q_pos_posterior(x0=x0, xt=xt, t=t, batch=batch)
        kl_pos = normal_kl(pos_true_mean, pos_log_variance, pos_model_mean, pos_log_variance)
        kl_pos = kl_pos / np.log(2.)

        decoder_nll_pos = -log_normal(x0, means=pos_model_mean, log_scales=0.5 * pos_log_variance)
        assert kl_pos.shape == decoder_nll_pos.shape
        mask = (t == 0).float()[batch]
        loss_pos = scatter_mean(mask * decoder_nll_pos + (1. - mask) * kl_pos, batch, dim=0)
        return loss_pos

    def compute_v_Lt(self, log_v_model_prob, log_v0, log_v_true_prob, t, batch):
        kl_v = categorical_kl(log_v_true_prob, log_v_model_prob)  # [num_atoms, ]
        decoder_nll_v = -log_categorical(log_v0, log_v_model_prob)  # L0
        assert kl_v.shape == decoder_nll_v.shape
        mask = (t == 0).float()[batch]
        loss_v = scatter_mean(mask * decoder_nll_v + (1. - mask) * kl_v, batch, dim=0)
        return loss_v

    def get_diffusion_loss(
            self, protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand, pid=None, lid=None, info_path=None, time_step=None
    ):
        # protein_pos, [x,y,z]
        # protein_v, feature [27]
        # batch_protein, batch
        # ligand_pos, [x,y,z]
        # ligand_v, feature [1]
        # batch_ligand,
        # print("get_diffusion_loss")
        num_graphs = batch_protein.max().item() + 1
        protein_pos, ligand_pos, _ = center_pos(
            protein_pos, ligand_pos, batch_protein, batch_ligand, mode=self.center_pos_mode)

        # 1. sample noise levels
        if time_step is None:
            time_step, pt = self.sample_time(num_graphs, protein_pos.device, self.sample_time_method)
        else:
            pt = torch.ones_like(time_step).float() / self.num_timesteps
        a = self.alphas_cumprod.index_select(0, time_step)  # (num_graphs, )

        # 2. perturb pos and v
        a_pos = a[batch_ligand].unsqueeze(-1)  # (num_ligand_atoms, 1)
        pos_noise = torch.zeros_like(ligand_pos)
        pos_noise.normal_() # get noise eps mean=0, std=1 for ligand
        # Xt = a.sqrt() * X0 + (1-a).sqrt() * eps
        ligand_pos_perturbed = a_pos.sqrt() * ligand_pos + (1.0 - a_pos).sqrt() * pos_noise  # get xt by x0 and eps
        # Vt = a * V0 + (1-a) / K
        log_ligand_v0 = index_to_log_onehot(ligand_v, self.num_classes) # log and onehot
        ligand_v_perturbed, log_ligand_vt = self.q_v_sample(log_ligand_v0, time_step, batch_ligand) # get vt by v0 and K

        # 3. forward-pass NN, feed perturbed pos and v, output noise
        preds = self(
            protein_pos=protein_pos,  # [N*E, 3]
            protein_v=protein_v,  # [N*E, 27]
            batch_protein=batch_protein,

            init_ligand_pos=ligand_pos_perturbed,  # [N*E, 3]
            init_ligand_v=ligand_v_perturbed,  # [N*E, ]
            batch_ligand=batch_ligand,
            time_step=time_step  # [N, ]
        ) # get pre_x0 and pre_v0 by protein_x protein_v xt

        pred_ligand_pos, pred_ligand_v = preds['pred_ligand_pos'], preds['pred_ligand_v'] # pre_x0[N*E,3] pre_v0[N*E,13]
        pred_pos_noise = pred_ligand_pos - ligand_pos_perturbed # pre_eps[N*E,3]
        # atom position
        if self.model_mean_type == 'noise':
            pos0_from_e = self._predict_x0_from_eps(
                xt=ligand_pos_perturbed, eps=pred_pos_noise, t=time_step, batch=batch_ligand)
            pos_model_mean = self.q_pos_posterior(
                x0=pos0_from_e, xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)
        elif self.model_mean_type == 'C0':
            pos_model_mean = self.q_pos_posterior(
                x0=pred_ligand_pos, xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)
        else:
            raise ValueError

        # atom pos loss
        if self.model_mean_type == 'C0':
            target, pred = ligand_pos, pred_ligand_pos # 根据true position和pre position计算loss
        elif self.model_mean_type == 'noise':
            target, pred = pos_noise, pred_pos_noise # 根据正态分布的随机噪声 和 预测的噪声 计算loss
        else:
            raise ValueError
        loss_pos = scatter_mean(((pred - target) ** 2).sum(-1), batch_ligand, dim=0) # x0[N*E,3] pre_x0[N*E,3]

        # atom type loss
        log_ligand_v_recon = F.log_softmax(pred_ligand_v, dim=-1) # pre feature  pre_v0[N*E,13]
        log_v_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_vt, time_step, batch_ligand) # pre and diffusion pre_vt-1[N*E,13]
        log_v_true_prob = self.q_v_posterior(log_ligand_v0, log_ligand_vt, time_step, batch_ligand) # true and diffusion vt-1[N*E,13]
        kl_v = self.compute_v_Lt(log_v_model_prob=log_v_model_prob, log_v0=log_ligand_v0, # v0[N*E,13]
                                 log_v_true_prob=log_v_true_prob, t=time_step, batch=batch_ligand) # pre and true calculate loss
        loss_v = kl_v * self.loss_v_weight

        if info_path is not None:
            # print('-----------modify loss---------------')
            # QED*w_Q + SA*w_S
            import pickle
            with open(info_path, 'rb') as file:
                info = pickle.load(file)
            QED = torch.tensor([0.0 for i in range(len(pid))]).to('cuda')
            SA = torch.tensor([0.0 for i in range(len(pid))]).to('cuda')
            VS = torch.tensor([0.0 for i in range(len(pid))]).to('cuda')
            for i in range(len(pid)):
                QED[i] = info[pid[i] + '-' + lid[i]]['QED']
                SA[i] = info[pid[i] + '-' + lid[i]]['SA']
                VS[i] = min(0, info[pid[i] + '-' + lid[i]]['affinity_vina'])

            loss = loss_pos + loss_v
            loss_pos = torch.mean(loss_pos)
            loss_v = torch.mean(loss_v)  # feature loss
            QED_map = {0.9: 0.80, 0.8: 0.72, 0.75: 0.69, 0.7: 0.65, 0.65: 0.61, 0.6: 0.58, 0.5: 0.52}
            SA_map = {0.9: 0.87, 0.8: 0.81, 0.75: 0.78, 0.7: 0.76, 0.65: 0.74, 0.6: 0.72, 0.5: 0.69}
            VS_map = {0.9: -11.04, 0.8: -10.05, 0.75: -9.69, 0.7: -9.37, 0.65: -9.08, 0.6: -8.82, 0.5: -8.32}
            QED_threshold = QED_map[0.7]
            SA_threshold = SA_map[0.7]
            VS_threshold = VS_map[0.7]
            QED = QED / QED_threshold
            SA = SA / SA_threshold
            VS = VS / VS_threshold
            a = 0.3
            w = (QED*SA*VS) ** a
            loss = loss * w
            # print(loss)
            loss = torch.mean(loss)
        else:
            loss_pos = torch.mean(loss_pos)
            loss_v = torch.mean(loss_v)  # feature loss
            loss = loss_pos + loss_v

        return {
            'loss_pos': loss_pos,
            'loss_v': loss_v,
            'loss': loss,
            'x0': ligand_pos,
            'pred_ligand_pos': pred_ligand_pos,
            'pred_ligand_v': pred_ligand_v,
            'pred_pos_noise': pred_pos_noise,
            'ligand_v_recon': F.softmax(pred_ligand_v, dim=-1)
        }

    @torch.no_grad()
    def likelihood_estimation(
            self, protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand, time_step
    ):
        protein_pos, ligand_pos, _ = center_pos(
            protein_pos, ligand_pos, batch_protein, batch_ligand, mode='protein')
        assert (time_step == self.num_timesteps).all() or (time_step < self.num_timesteps).all()
        if (time_step == self.num_timesteps).all():
            kl_pos_prior = self.kl_pos_prior(ligand_pos, batch_ligand)
            log_ligand_v0 = index_to_log_onehot(batch_ligand, self.num_classes)
            kl_v_prior = self.kl_v_prior(log_ligand_v0, batch_ligand)
            return kl_pos_prior, kl_v_prior

        # perturb pos and v
        a = self.alphas_cumprod.index_select(0, time_step)  # (num_graphs, )
        a_pos = a[batch_ligand].unsqueeze(-1)  # (num_ligand_atoms, 1)
        pos_noise = torch.zeros_like(ligand_pos)
        pos_noise.normal_()
        # Xt = a.sqrt() * X0 + (1-a).sqrt() * eps
        ligand_pos_perturbed = a_pos.sqrt() * ligand_pos + (1.0 - a_pos).sqrt() * pos_noise  # pos_noise * std
        # Vt = a * V0 + (1-a) / K
        log_ligand_v0 = index_to_log_onehot(ligand_v, self.num_classes)
        ligand_v_perturbed, log_ligand_vt = self.q_v_sample(log_ligand_v0, time_step, batch_ligand)

        preds = self(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,

            init_ligand_pos=ligand_pos_perturbed,
            init_ligand_v=ligand_v_perturbed,
            batch_ligand=batch_ligand,
            time_step=time_step
        )

        pred_ligand_pos, pred_ligand_v = preds['pred_ligand_pos'], preds['pred_ligand_v']
        if self.model_mean_type == 'C0':
            pos_model_mean = self.q_pos_posterior(
                x0=pred_ligand_pos, xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)
        else:
            raise ValueError

        # atom type
        log_ligand_v_recon = F.log_softmax(pred_ligand_v, dim=-1)
        log_v_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_vt, time_step, batch_ligand)
        log_v_true_prob = self.q_v_posterior(log_ligand_v0, log_ligand_vt, time_step, batch_ligand)

        # t = [T-1, ... , 0]
        kl_pos = self.compute_pos_Lt(pos_model_mean=pos_model_mean, x0=ligand_pos,
                                     xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)
        kl_v = self.compute_v_Lt(log_v_model_prob=log_v_model_prob, log_v0=log_ligand_v0,
                                 log_v_true_prob=log_v_true_prob, t=time_step, batch=batch_ligand)
        return kl_pos, kl_v

    @torch.no_grad()
    def fetch_embedding(self, protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand):
        preds = self(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,

            init_ligand_pos=ligand_pos,
            init_ligand_v=ligand_v,
            batch_ligand=batch_ligand,
            fix_x=True
        )
        return preds

    @torch.no_grad()
    def sample_diffusion(self, protein_pos, protein_v, batch_protein,
                         init_ligand_pos, init_ligand_v, batch_ligand,
                         num_steps=None, center_pos_mode=None, pos_only=False):

        # protein_pos            px0[100*E, 3] raw not rel
        # protein_atom_feature   pv0[100*E, 27]
        # init_ligand_pos         xt[100*E, 3] raw not rel
        # init_ligand_v           vt[100*E, 1]

        if num_steps is None:
            num_steps = self.num_timesteps
        num_graphs = batch_protein.max().item() + 1

        protein_pos, init_ligand_pos, offset \
            = center_pos(protein_pos, init_ligand_pos, batch_protein, batch_ligand, mode=center_pos_mode)
        # px0[100*E, 3]  xt[100*E, 3]  protein center pos pcx0[100, 3]

        pos_traj, v_traj = [], []
        v0_pred_traj, vt_pred_traj = [], []
        ligand_pos, ligand_v = init_ligand_pos, init_ligand_v # xt[100*E, 3]  vt[100*E, 1]
        # time sequence
        time_seq = list(reversed(range(self.num_timesteps - num_steps, self.num_timesteps)))
        # time_seq = [1, 0] # for debug
        for i in tqdm(time_seq, desc='sampling', total=len(time_seq)):
            t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=protein_pos.device)
            preds = self(
                protein_pos=protein_pos,     # px0[100*E, 3]   每一轮都不变
                protein_v=protein_v,         # pv0[100*E, 27]  每一轮都不变
                batch_protein=batch_protein,

                init_ligand_pos=ligand_pos,  # xt[100*E, 3]   每一轮都更新 pre_xt-1[100*E, 3]
                init_ligand_v=ligand_v,      # vt[100*E, 1]   每一轮都更新 pre_vt-1[100*E, 1]
                batch_ligand=batch_ligand,
                time_step=t
            )
            # Compute posterior mean and variance
            if self.model_mean_type == 'noise':
                pred_pos_noise = preds['pred_ligand_pos'] - ligand_pos
                pos0_from_e = self._predict_x0_from_eps(xt=ligand_pos, eps=pred_pos_noise, t=t, batch=batch_ligand)
                v0_from_e = preds['pred_ligand_v']
            elif self.model_mean_type == 'C0':
                pos0_from_e = preds['pred_ligand_pos']  # pre_x0[100*E, 3]
                v0_from_e = preds['pred_ligand_v']      # pre_v0[100*E, 13]
            else:
                raise ValueError

            # pos xt-1
            pos_model_mean = self.q_pos_posterior(x0=pos0_from_e, xt=ligand_pos, t=t, batch=batch_ligand) # pre_x_mean[100*E, 3]
            pos_log_variance = extract(self.posterior_logvar, t, batch_ligand)  # pre_x_variance[100*E, 1]
            # no noise when t == 0
            nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)   # pre_x_mask[100*E, 1]
            ligand_pos_next \
                = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(ligand_pos)
            # pre_xt-1[100*E, 3]  from pre_x0 and xt
            ligand_pos = ligand_pos_next # xt = pre_xt-1[100*E, 3]  from pre_x0 and xt

            if not pos_only:
                # feature vt-1
                log_ligand_v_recon = F.log_softmax(v0_from_e, dim=-1) # pre_v0[100*E, 13]
                log_ligand_v = index_to_log_onehot(ligand_v, self.num_classes) # vt[100*E, 13]
                log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand) # pre_vt-1[100*E, 13] from pre_v0 and vt
                ligand_v_next = log_sample_categorical(log_model_prob) # pre_vt-1[100*E, 1]

                # # post vt-1
                # a1 = torch.exp(log_model_prob)[0:10]
                # a2 = F.softmax(a1, dim=-1)
                # a3 = torch.argmax(a2, -1)
                # # vt
                # c1 = torch.exp(log_ligand_v)[0:10]
                # # pred pr
                # b1 = torch.exp(log_ligand_v_recon)[0:10]
                # b2 = F.softmax(v0_from_e, dim=-1)[0:10]
                # b3 = torch.argmax(b2, -1)

                v0_pred_traj.append(log_ligand_v_recon.clone().cpu())  # pre_v0[100*E, 13]
                vt_pred_traj.append(log_model_prob.clone().cpu())      # pre_vt-1[100*E, 13] from pre_v0 and vt
                ligand_v = ligand_v_next  # pre_vt-1[100*E, 1]

            ori_ligand_pos = ligand_pos + offset[batch_ligand] # raw pos not rel_pos # xt = pre_xt-1[100*E, 3]
            pos_traj.append(ori_ligand_pos.clone().cpu())   # raw pos  xt = pre_xt-1[100*E, 3]
            v_traj.append(ligand_v.clone().cpu())  # pre_vt-1[100*E, 1]

        ligand_pos = ligand_pos + offset[batch_ligand]
        return {
            'pos': ligand_pos,   # abs_pos pre_xt-1[100*E, 3]  from pre_x0 and xt 最后一轮的
            'v': ligand_v,       # pre_vt-1[100*E, 1] 最后一轮的
            'pos_traj': pos_traj,# abs pos pre_xt-1[100*E, 3] 的 ls 每一轮
            'v_traj': v_traj,    # pre_vt-1[100*E, 1] 的 ls 每一轮
            'v0_traj': v0_pred_traj, # pre_v0[100*E, 13] 的 ls 每一轮
            'vt_traj': vt_pred_traj  # pre_vt-1[100*E, 13]  的 ls 每一轮
        }


def extract(coef, t, batch):
    out = coef[t][batch]
    return out.unsqueeze(-1)
