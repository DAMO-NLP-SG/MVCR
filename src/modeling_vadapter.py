from turtle import forward
import torch
import torch.utils.checkpoint
from torch import layer_norm, nn
from torch.nn import CrossEntropyLoss, MSELoss
import configparser
import math
import os
import random
import global_var

def load_adapter_config(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

def add_adapter_config(config):
    if 'select_adapter' not in config.to_dict():
        try:
            select_adapter  = os.environ['USE_ADAPTER']
        except:
            select_adapter = None

        try:
            adapter_hidden = os.environ['ADAPTER_H']
        except:
            adapter_hidden = None

        try:
            adapter_num = os.environ['ADAPTER_NUM']
        except:
            adapter_num = None

        config.select_adapter = None 
        config.adapter_hidden = None
        config.adapter_num = None
       
        if select_adapter in ADAPTERS:
            config.select_adapter = select_adapter
            config.adapter_hidden = 64 # default set to 64
            config.adapter_num = [64] # default set to 1 adapter with 64 dim
            if adapter_hidden is not None:
                config.adapter_hidden = int(adapter_hidden)

            if adapter_num is not None:
                config.adapter_num = [int(x) for x in adapter_num.split(',')]

        elif select_adapter is not None:
            print('*' * 20)
            print('\n\ninvalid choice of adapter: %s\n\n' % select_adapter)
            print('*' * 20)
            raise NotImplementedError
    return config 


class Adapter_Org2(nn.Module):
    def __init__(self, config, layerid):
        super().__init__()
        hidden_size = config.hidden_size
        adapter_hidden = config.adapter_hidden
        self.dense_down = nn.Linear(hidden_size, adapter_hidden)
        self.dense_up = nn.Linear(adapter_hidden, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        return self.dense_up(self.activation(self.dense_down(hidden_states))) + hidden_states


class InfoBottle(nn.Module):
    def __init__(self, config, compress_dim):
        super().__init__()
        self.dense_down = nn.Linear(config.hidden_size, compress_dim)
        self.dense_up = nn.Linear(compress_dim, config.hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, hidden_states):
        out = self.dense_down(hidden_states)
        return self.dense_up(self.dropout(self.activation(out)))


class InfoBottleSub(nn.Module):
    def __init__(self, hidden_size, compress_dim):
        super().__init__()
        self.dense_down = nn.Linear(hidden_size, compress_dim)
        self.dense_up = nn.Linear(compress_dim, hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, hidden_states):
        out = self.dense_down(hidden_states)
        return self.dense_up(self.dropout(self.activation(out)))



class InfoBottle2(nn.Module):
    def __init__(self, config, compress_dim, num_aes=2):
        super().__init__()
        self.dense_down = nn.Linear(config.hidden_size, compress_dim)
        self.dense_up = nn.Linear(compress_dim, config.hidden_size)
        self.num_aes = num_aes
        self.aes = nn.ModuleList([InfoBottleSub(compress_dim, dim) for dim in [int(compress_dim/2)] * 2])
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, hidden_states):
        out = self.dense_down(hidden_states)
        out = self.dropout(self.activation(out))
        random_ae = random.randint(0, self.num_aes)
        if random_ae < self.num_aes:
            out = self.aes[random_ae](out)
            out = self.activation(out)
        return self.dense_up(out)


class AE2TK(nn.Module):
    def __init__(self, config, layerid):
        super().__init__()
        self.dim_list = config.vae_sets
        self.aes = nn.ModuleList([InfoBottle2(config, dim) for dim in self.dim_list])
        self.num_aes = len(config.vae_sets)
        self.layerid = layerid
        self.alpha = config.recon_loss_weight

    def forward(self, hidden_states):
        if self.training:
            batch_size, seq_len, dim = hidden_states.shape
            hs_ae = []
            hidden_states = hidden_states + torch.randn(hidden_states.shape).cuda() * 0.002
            for i in range(self.num_aes):
                hs_ae.append(self.aes[i](hidden_states))
            hs_ae.append(hidden_states)
            weights_list = []
            for i in range(self.num_aes + 1):
                weights_list.append(torch.zeros(batch_size, seq_len, 1).cuda())
            for i in range(seq_len):
                random_ae = random.randint(0, (self.num_aes - 1) * 2)
                if random_ae < self.num_aes:
                    weights_list[random_ae][:,i] = 1.0
                else:
                    weights_list[self.num_aes][:,i] = 1.0

            opt = sum([weights_list[i] * hs_ae[i] for i in range(self.num_aes + 1)])

            # recon_loss
            recon_criterion = MSELoss()
            recon_loss = []
            for i in range(self.num_aes):
                recon_loss.append(recon_criterion(hidden_states, hs_ae[i]))
            global_var.recon_loss[self.layerid] = sum(recon_loss) / float(self.num_aes)
        else:
            opt = hidden_states
        return opt


class GN(nn.Module):
    def __init__(self, config, layerid):
        super().__init__()

    def forward(self, hidden_states):
        if self.training:
            opt = hidden_states + torch.randn(hidden_states.shape).cuda() * 0.002
        else:
            opt = hidden_states
        return opt


class AE(nn.Module):
    def __init__(self, config, layerid):
        super().__init__()
        self.dense_down = nn.ModuleList([nn.Linear(config.hidden_size, x) for x in config.vae_sets])
        self.dense_up = nn.ModuleList([nn.Linear(x, config.hidden_size) for x in config.vae_sets])

        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(p=0.2)

        self.num_aes = len(config.vae_sets)
        self.layerid = layerid
        self.recon_loss_weight = config.recon_loss_weight

    def forward(self, hidden_states):
        if self.training and not global_var.last_epoch_training:
            a = self.recon_loss_weight
            weight_list = [0] * (self.num_aes + 1)
            random_layer = random.randint(0, self.num_aes)
            weight_list[random_layer] = 1

            opt_list = [hidden_states]
            for i, l in enumerate(self.dense_down):
                opt_list.append(self.dense_up[i](self.dropout(self.activation(self.dense_down[i](hidden_states)))) + hidden_states)

            opt = 0
            for i in range(len(weight_list)):
                opt += weight_list[i] * opt_list[i]
            opt = (1 - a) * opt + a * hidden_states

            # recon_loss
            recon_criterion = MSELoss()
            recon_loss = recon_criterion(hidden_states, opt)
            global_var.recon_loss[self.layerid] = recon_loss * 0

        else:
            opt = hidden_states
        return opt


class OriAdapter(nn.Module):
    def __init__(self, config, layerid):
        super().__init__()
        self.dense_down = nn.ModuleList([nn.Linear(config.hidden_size, x) for x in config.vae_sets])
        self.dense_up = nn.ModuleList([nn.Linear(x, config.hidden_size) for x in config.vae_sets])

        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(p=0.2)

        self.num_aes = len(config.vae_sets)
        self.layerid = layerid
        self.recon_loss_weight = config.recon_loss_weight

    def forward(self, hidden_states):
        a = self.recon_loss_weight
        weight_list = [0] * (self.num_aes + 1)
        random_layer = 1
        weight_list[random_layer] = 1

        opt_list = [hidden_states]
        for i, l in enumerate(self.dense_down):
            opt_list.append(self.dense_up[i](self.dropout(self.activation(self.dense_down[i](hidden_states)))) + hidden_states)

        opt = 0
        for i in range(len(weight_list)):
            opt += weight_list[i] * opt_list[i]

        recon_criterion = MSELoss()
        recon_loss = recon_criterion(hidden_states, opt)
        global_var.recon_loss[self.layerid] = recon_loss * 0

        return opt

class VAE(nn.Module):
    def __init__(self, config, layerid):
        super().__init__()

        self.beta = config.adapter_loss_weight
        self.sample_size = 5
        self.layerid = layerid
        self.num_aes = len(config.vae_sets)
        self.recon_loss_weight = config.recon_loss_weight

        self.dense_up = nn.ModuleList([nn.Linear(x, config.hidden_size) for x in config.vae_sets])

        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(p=0.2)

        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size, 600),
                self.activation,
                nn.Linear(600, 512),
                self.activation) for i in range(len(config.vae_sets))])

        self.emb2mu = nn.ModuleList([nn.Linear(512, x) for x in config.vae_sets])
        self.emb2std = nn.ModuleList([nn.Linear(512, x) for x in config.vae_sets])

        self.mu_p = nn.ParameterList([nn.Parameter(torch.randn(128, x)) for x in config.vae_sets])
        self.std_p = nn.ParameterList([nn.Parameter(torch.randn(128, x)) for x in config.vae_sets])


    def estimate(self, emb, emb2mu, emb2std):
        """Estimates mu and std from the given input embeddings."""
        mean = emb2mu(emb)
        std = torch.nn.functional.softplus(emb2std(emb))
        return mean, std

    def kl_div(self, mu_q, std_q, mu_p, std_p):
        """Computes the KL divergence between the two given variational distribution.\
           This computes KL(q||p), which is not symmetric. It quantifies how far is\
           The estimated distribution q from the true distribution of p."""
        k = mu_q.size(1)
        mu_diff = mu_p - mu_q
        mu_diff_sq = torch.mul(mu_diff, mu_diff)
        logdet_std_q = torch.sum(2 * torch.log(torch.clamp(std_q, min=1e-8)), dim=1)
        logdet_std_p = torch.sum(2 * torch.log(torch.clamp(std_p, min=1e-8)), dim=1)
        fs = torch.sum(torch.div(std_q ** 2, std_p ** 2), dim=1) + torch.sum(torch.div(mu_diff_sq, std_p ** 2), dim=1)
        kl_divergence = (fs - k + logdet_std_p - logdet_std_q)*0.5
        return kl_divergence.mean()

    def reparameterize(self, mu, std):
        batch_size = mu.shape[0]
        z = torch.randn(self.sample_size, batch_size, mu.shape[1], mu.shape[2]).cuda()
        # take mean of the samples
        z = z.mean(dim=0)
        return mu + std * z


    def vae(self, hidden_states, mlp, emb2mu, emb2std, mu_p, std_p, beta):
        pooled_hidden = mlp(hidden_states) 
        batch_size = pooled_hidden.shape[0]
        mu, std = self.estimate(pooled_hidden, emb2mu, emb2std)
        mu_p = mu_p.view(1, 128, -1).expand(batch_size, -1, -1)
        std_p = torch.nn.functional.softplus(std_p.view(1, 128, -1).expand(batch_size, -1, -1))
        kl_loss = self.kl_div(mu, std, mu_p, std_p)
        z = self.reparameterize(mu, std)

        return z, beta * kl_loss

    def forward(self, hidden_states):
        if self.training and not global_var.last_epoch_training:
            weight_list = [0] * (self.num_aes + 1)
            random_layer = random.randint(0, self.num_aes)
            weight_list[random_layer] = 1

            # if direct pass through, kl loss is 0
            kl_loss_list = [0]
            opt_list = [hidden_states]
            
            for i, l in enumerate(self.dense_up):
                tmp_z, tmp_kl_loss = self.vae(hidden_states, self.mlp[i], self.emb2mu[i], self.emb2std[i], self.mu_p[i], self.std_p[i], self.beta)
                tmp_opt = self.dense_up[i](self.dropout(tmp_z))
                
                kl_loss_list.append(tmp_kl_loss)
                opt_list.append(tmp_opt)

            opt = 0
            kl_loss = 0
            for i in range(len(weight_list)):
                opt += weight_list[i] * opt_list[i]
                kl_loss += weight_list[i] * kl_loss_list[i]
            
            opt /= sum(weight_list)

            opt = 0.5 * opt + 0.5 * hidden_states

            kl_loss /= sum(weight_list)
            global_var.kl_loss[self.layerid] = kl_loss

            # recon_loss
            recon_criterion = MSELoss()
            recon_loss = recon_criterion(hidden_states, opt)
            global_var.recon_loss[self.layerid] = recon_loss * self.recon_loss_weight

        else:
            opt = hidden_states
        return opt



ADAPTERS = {
    'AE': AE,
    'AE2TK': AE2TK,
    'VAE': VAE,
    'GN': GN,
    'OriAdapter': OriAdapter,
    'Adapter_Org2': Adapter_Org2,
}
