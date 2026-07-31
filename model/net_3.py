import torch

from feature.sampler import SAMPLER
from feature.activation import ACTIVATION

class Net3(torch.nn.Module):
    
    # input -> [conv -> relu]*n -> fc -> output
    
    def __init__(
        self,
        num_conv_relu,
        conv_kernel_size,
        conv_stride,
        conv_padding
    ):
        
        print("\ninit - Net3")
        
        super().__init__()
        
        #---------------------------------------------------
        