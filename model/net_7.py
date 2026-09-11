import torch
import numpy as np

from feature.sampler import SAMPLER
from feature.activation import ACTIVATION


class Net5(torch.nn.Module):
    
# input -> 
# [[conv]*n -> conv stride 2]*m
# conv(1*1)*k -> gap ->
# output
 

# Springenberg All Convolutional table 2 all-cnn-c

    def __init__(
        self,
    ):
        
        print("\ninit - Net7")

