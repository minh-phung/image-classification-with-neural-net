import torch.nn as nn
from functools import partial


SAMPLER = {
    "constant": partial(nn.init.constant_, val = 0.01), 
    "zeros": nn.init.zeros_,
    "xavieruniform": nn.init.xavier_uniform_,
    "xaviernormal": nn.init.xavier_normal_,
    "kaiminguniform": partial(nn.init.kaiming_uniform_, nonlinearity = 'relu'),
    "kaimingnormal": partial(nn.init.kaiming_normal_, nonlinearity= 'relu')
}

