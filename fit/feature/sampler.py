import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy

#--------------------------------------

def init_xavier_uniform_weight(module):
    
    if isinstance(module, torch.nn.Linear):

        torch.nn.init.xavier_uniform_(module.weight)
        torch.nn.init.zeros_(module.bias)
        
        print(module.weight)
        
#--------------------------------------

def init_kaiming_uniform_weight(module):
    
    if isinstance(module, torch.nn.Linear):
        
        torch.nn.init.kaiming_uniform_(module.weight)
        torch.nn.init.constant_(module.bias, 0.01)
    
    return
    
#-----------------------------------------------------------------


SAMPLER = {
    "xavier_uniform": init_xavier_uniform_weight,
    "kaiming_uniform": init_kaiming_uniform_weight
}
