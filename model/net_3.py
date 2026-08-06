import torch
import numpy as np

from feature.sampler import SAMPLER
from feature.activation import ACTIVATION

class Net3(torch.nn.Module):
    
    # input -> [conv -> relu]*n  -> gap (feature) -> output
    # 
    # gap: global average pooling
    
    # Springenberg All Convolutional 
    
    def __init__(
        self,
        lay_conv_number,
        lay_conv_kernel
    ):
        
        print("\ninit - Net3")
        
        super().__init__()
        
        input_count = 3*64*64
        
        #---------------------------------------------------
        in_channel = 3
        out_channel = np.exp(np.log(input_count)/4).astype(int)
        
        width = 64
                
        self.lay_conv = [None]*lay_conv_number
        self.lay_conv_act = ACTIVATION["relu"]
        
        for i in range(lay_conv_number):
            
            print("\nlayer - conv", i)
            print("feature", out_channel)
            print("kernel", lay_conv_kernel)
            
            self.lay_conv[i] = torch.nn.Conv2d(
                in_channels = in_channel,
                out_channels = out_channel,
                kernel_size = lay_conv_kernel,
                stride = 1,
                padding = 0,
                groups = 1
            )
            
            print("weight", "kaiming_uniform")
            SAMPLER["kaiming_uniform"](self.lay_conv[i].weight)
            
            print("bias", "constant")
            SAMPLER["constant"](self.lay_conv[i].bias)
            
            print("activation", self.lay_conv_act)
            
            width = ((width - lay_conv_kernel + 2*0) /(1)) + 1
            
            in_channel = out_channel
        
        self.out_conv = int(width**2 * out_channel)
        
        
        
    
    def forward(self, x):
        
        print(x.shape)
        
        for each_conv in self.lay_conv:
            
            x = self.lay_conv_act(each_conv(x))
        
        print(x.shape)
        
        quit()
        
        return