import torch
import numpy as np

from feature.sampler import SAMPLER
from feature.activation import ACTIVATION

class Net6(torch.nn.Module):
    
    def __init__(self):
        
        