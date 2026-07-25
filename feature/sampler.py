import torch.nn as nn
from functools import partial


SAMPLER = {
    "constant": partial(nn.init.constant_, val = 0.01), 
    "zeros": nn.init.zeros_,
    "xavier_uniform": nn.init.xavier_uniform_,
    "xavier_normal": nn.init.xavier_normal_,
    "kaiming_uniform": partial(nn.init.kaiming_uniform_, nonlinearity = 'relu'),
    "kaiming_normal": partial(nn.init.kaiming_normal_, nonlinearity= 'relu')
}

