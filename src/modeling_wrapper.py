import random

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import configparser
import math
import os


def load_wrapper_config(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    return config


def add_wrapper_config(config):

    # If 'select_wrapper' does not exits in config, check whether the relevant env variables are defined 
    # If these args exist in config, the env variables will be ignored.
    if 'select_wrapper' not in config.to_dict():
        try:
            select_wrapper = os.environ['USE_WRAPPER']
        except:
            select_wrapper = None

        try:
            wrapper_config_path = os.environ['WRAPPER_CONFIG_PATH']
        except:
            wrapper_config_path = None

        config.select_wrapper = None 
        config.wrapper_config = None

        if select_wrapper in WRAPPERS:
            config.select_wrapper = select_wrapper
            if wrapper_config_path is not None:
                if not os.path.isfile(wrapper_config_path):
                    print('WRAPPER_CONFIG_PATH is not found: %s' % wrapper_config_path)
                    raise FileNotFoundError
                wrapper_config = load_wrapper_config(wrapper_config_path)
                if select_wrapper in wrapper_config:
                    config.wrapper_config = dict(wrapper_config[select_wrapper])

        elif select_wrapper is not None:
            print('*' * 20)
            print('\n\ninvalid choice of wrapper: %s\n\n' % select_wrapper)
            print('*' * 20)
            raise NotImplementedError
    return config 




class HorizontalPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
 
    def forward(self, hidden_states, attention_mask):
        hidden_states_wo_mask = ()
        for layer_hidden_states in hidden_states:
            hidden_states_wo_mask += (layer_hidden_states * attention_mask.unsqueeze(2),)
        hidden_states_wo_mask = torch.stack(hidden_states_wo_mask)
        hidden_states_sum = torch.sum(hidden_states_wo_mask, dim=2)
        attention_sum = torch.sum(attention_mask, dim=1)
        pooled_mean = hidden_states_sum / torch.unsqueeze(attention_sum, 1)
        pooled_output = torch.cat((torch.unsqueeze(pooled_mean, 2), hidden_states[:,:,1:]), 2)
        return pooled_output


class InfoBottleNeck(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bottle_neck_size = int(config.wrapper_config['bottle_neck_size'])
        self.dense_down = nn.Linear(config.hidden_size, self.bottle_neck_size)
        self.dense_up = nn.Linear(self.bottle_neck_size, config.hidden_size)
        self.activation = nn.Tanh()
 
    def forward(self, hidden_states, attention_mask):
        return self.dense_up(self.activation(self.dense_down(hidden_states)))


class HorizontalAttentionPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_layers = config.num_hidden_layers
        n_dim = config.hidden_size
        stdv = 1./math.sqrt(n_dim)
        self.activation = nn.Tanh()
        #use different attention vectors for different layers
        self.att_vectors = nn.ParameterList([nn.Parameter(torch.FloatTensor(n_dim, 1).uniform_(-stdv, stdv), 
                requires_grad=True) for _ in range(num_layers)])

        self.h_pooler = HorizontalPooler(config)


    def forward(self, hidden_states, attention_mask):
        hidden_states_att = ()
        for i, features in enumerate(hidden_states):
            weights = torch.bmm(features, # (batch_size, seq_len, hidden_size)
                                self.att_vectors[i]  # (hidden_size, 1)
                                .unsqueeze(0)  # (1, hidden_size, 1)
                                .repeat(features.shape[0], 1, 1) # (batch_size, hidden_size, 1)
                                )

            weights = self.activation(weights.squeeze())
            weights = torch.exp(weights) * attention_mask
            weights = weights / torch.sum(weights, dim=1, keepdim=True)
            features = features * weights.unsqueeze(2)
            hidden_states_att += (features,)

        hidden_states_att = torch.stack(hidden_states_att)        
        pooled_output = self.h_pooler(hidden_states_att, attention_mask)
        return pooled_output




class HorizontalCNNPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_layers = config.num_hidden_layers
        in_dim = config.hidden_size
        out_dim = config.hidden_size
        # default kernel size is 3
        kernel_size = 3
        if config.wrapper_config is not None:
            if "cnn_kernel_size" in config.wrapper_config:
                kernel_size = int(config.wrapper_config['cnn_kernel_size'])

        #use different CNNs for different layers
        self.cnns = nn.ModuleList([nn.Conv1d(in_dim, out_dim, kernel_size, 
                        padding=int(kernel_size/2)) for _ in range(num_layers)])
        self.activation = nn.ReLU()
        self.h_pooler = HorizontalPooler(config)


    def forward(self, hidden_states, attention_mask):
        hidden_states_cnn = ()
        for i, features in enumerate(hidden_states):
            # convert shape (batch_size, seq_len, hidden_size) to (batch_size, hidden_size, seq_len)
            features = torch.transpose(features, 1, 2)
            features = self.cnns[i](features)
            features = self.activation(features)
            # convert shape back to (batch_size, seq_len, hidden_size)
            features = torch.transpose(features, 1, 2)
            hidden_states_cnn += (features,)

        hidden_states_cnn = torch.stack(hidden_states_cnn)        
        pooled_output = self.h_pooler(hidden_states_cnn, attention_mask)
        return pooled_output


class BertWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.h_pooler = HorizontalPooler(config)
        if config.wrapper_config is not None:
            if "wrapper_layers" in config.wrapper_config:
                self.wrapper_layers = list(map(int, config.wrapper_config['wrapper_layers'].split(',')))
        else:
            if config.select_wrapper is None:
                None
            else:
                raise Exception('wrapper config is not given')

    def forward(self, hidden_states, attention_mask):
        raise NotImplementedError


class BertWrapperLastLayer(BertWrapper):

    def __init__(self, config):
        super().__init__(config)

    def forward(self, hidden_states, attention_mask):
        return hidden_states[-1]


class BertWrapperMean(BertWrapper):

    def __init__(self, config):
        super().__init__(config)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, attention_mask):
        hidden_states_layers = torch.stack(hidden_states[self.wrapper_layers])
        output = torch.mean(hidden_states_layers, axis=0)
        output = self.activation(output)
        return output


class BertWrapperMeanBN(BertWrapper):

    def __init__(self, config):
        super().__init__(config)
        self.info_bn = nn.ModuleList([InfoBottleNeck(config) for _ in self.wrapper_layers])
        self.bn_keep_n = None
        if 'bn_keep_n' in config.wrapper_config:
            self.bn_keep_n = int(config.wrapper_config['bn_keep_n'])
        self.activation = nn.Tanh()

    def forward(self, hidden_states, attention_mask):
        bn_info = [self.info_bn[i](hidden_states[layer], attention_mask) for i,layer in enumerate(self.wrapper_layers)]
        if self.training:
            if self.bn_keep_n is not None:
                random.shuffle(bn_info)
                bn_info = bn_info[:self.bn_keep_n]
        hidden_states_layers = torch.stack(bn_info)
        output = torch.mean(hidden_states_layers, axis=0)
        output = self.activation(output)
        return output


class BertWrapperMeanBN2(BertWrapper):
    def __init__(self, config):
        super().__init__(config)
        self.info_bn = nn.ModuleList([InfoBottleNeck(config) for _ in self.wrapper_layers])
        self.bn_keep_n = int(config.wrapper_config['bn_keep_n'])
        self.bn_keep_step = int(config.wrapper_config['bn_keep_step'])
        self.count = self.bn_keep_step
        self.bn_keep_tmp = []
        self.activation = nn.Tanh()

    def forward(self, hidden_states, attention_mask):
        bn_info = [self.info_bn[i](hidden_states[layer], attention_mask) for i,layer in enumerate(self.wrapper_layers)]
        #bn_info = [hidden_states[layer] for i,layer in enumerate(self.wrapper_layers)]
        if self.training:
            if self.count >= self.bn_keep_step:
                tmp_layers = list(range(len(self.wrapper_layers)))
                random.shuffle(tmp_layers)
                self.bn_keep_tmp = tmp_layers[:random.randint(self.bn_keep_n, len(self.wrapper_layers))]
                self.count = 0
            bn_info = [bn_info[i] for i in self.bn_keep_tmp]
            self.count += 1
        hidden_states_layers = torch.stack(bn_info)
        output = torch.mean(hidden_states_layers, axis=0)
        output = self.activation(output)
        return output


class BertWrapperMeanHori(BertWrapper):

    def __init__(self, config):
        super().__init__(config)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, attention_mask):
        hidden_states_top_layers = torch.stack(hidden_states[self.wrapper_layers])
        if self.h_pooler is not None:
            hidden_states_top_layers = self.h_pooler(hidden_states_top_layers, attention_mask)
        output = torch.mean(hidden_states_top_layers, axis=0)
        output = self.activation(output)
        return output


class BertWrapperMeanHoriNoActivation(BertWrapper):

    def __init__(self, config):
        super().__init__(config)

    def forward(self, hidden_states, attention_mask):
        hidden_states_top_layers = torch.stack(hidden_states[self.wrapper_layers])
        if self.h_pooler is not None:
            hidden_states_top_layers = self.h_pooler(hidden_states_top_layers, attention_mask)
        output = torch.mean(hidden_states_top_layers, axis=0)
        return output

class ProductWeight(nn.Module):
    def __init__(self, init_val):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor([float(init_val)]), requires_grad=True)

    def forward(self, hidden_states):
        return self.weight * hidden_states


class BertWrapperWeight(BertWrapper):
    def __init__(self, config):
        super().__init__(config)
        self.layer_weight = nn.ModuleList([ProductWeight(1.0 / len(self.wrapper_layers)) for _ in self.wrapper_layers])
        self.activation = nn.Tanh()

    def forward(self, hidden_states, attention_mask):
        hidden_states_top_layers = torch.stack(hidden_states[self.wrapper_layers])
        if self.h_pooler is not None:
            hidden_states_top_layers = self.h_pooler(hidden_states_top_layers, attention_mask)
        output = torch.zeros_like(hidden_states_top_layers[0])
        for i in range(len(hidden_states_top_layers)):
            output += self.layer_weight[i](hidden_states_top_layers[i])
        #output = self.activation(output)
        return output


class BertWrapperCnnMEAN(BertWrapper):

    def __init__(self, config):
        super().__init__(config)
        self.activation = nn.Tanh()
        self.h_pooler = HorizontalCNNPooler(config)

    def forward(self, hidden_states, attention_mask):
        hidden_states_top_layers = torch.stack(hidden_states[self.wrapper_layers])
        if self.h_pooler is not None:
            hidden_states_top_layers = self.h_pooler(hidden_states_top_layers, attention_mask)
        output = torch.mean(hidden_states_top_layers, axis=0)
        output = self.activation(output)
        return output



class BertWrapperAttMean(BertWrapper):

    def __init__(self, config):
        super().__init__(config)
        self.activation = nn.Tanh()
        self.h_pooler = HorizontalAttentionPooler(config)

    def forward(self, hidden_states, attention_mask):
        hidden_states_top_layers = torch.stack(hidden_states[self.wrapper_layers])
        if self.h_pooler is not None:
            hidden_states_top_layers = self.h_pooler(hidden_states_top_layers, attention_mask)
        output = torch.mean(hidden_states_top_layers, axis=0)
        output = self.activation(output)
        return output


WRAPPERS = {'mean': BertWrapperMean,
            'mean_bn': BertWrapperMeanBN,
            'mean_bn2': BertWrapperMeanBN2,
            'mean_hori': BertWrapperMeanHori,
            'mean_hori_na': BertWrapperMeanHoriNoActivation,
            'weight': BertWrapperWeight,
            'last_layer': BertWrapperLastLayer,
            'cnn_mean': BertWrapperCnnMEAN,
            'att_mean': BertWrapperAttMean
           }
