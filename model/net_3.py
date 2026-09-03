import torch
import numpy as np

from feature.sampler import SAMPLER
from feature.activation import ACTIVATION

class Net3(torch.nn.Module):
    
    # input -> [conv -> relu]*n  -> conv(1*1), 1 feat -> gap -> output
    # 
    # gap: global average pooling 
    
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
                
        self.lay_conv = torch.nn.ModuleList()
        self.lay_conv_act = ACTIVATION["relu"]
        
        for i in range(lay_conv_number):
            
            print("\nlayer - conv", i)
            print("feature", out_channel)
            print("kernel", lay_conv_kernel)
            
            conv = torch.nn.Conv2d(
                in_channels = in_channel,
                out_channels = out_channel,
                kernel_size = lay_conv_kernel,
                stride = 1,
                padding = 0,
                groups = 1
            )
            
            print("weight", "kaiming_uniform")
            SAMPLER["kaiming_uniform"](conv.weight)
            
            print("bias", "constant")
            SAMPLER["constant"](conv.bias)

            self.lay_conv.append(conv)
            
            print("activation", self.lay_conv_act)
            
            width = ((width - lay_conv_kernel + 2*0) /(1)) + 1
            
            in_channel = out_channel
        
        #---------------------------------------------------
        
        print("\nlayer - conv - 1*1")
        
        self.lay_conv_feat = torch.nn.Conv2d(
            in_channels = out_channel,
            out_channels = 1,
            kernel_size = 1
        )
        
        print("weight", "xavier_uniform")
        SAMPLER["xavier_uniform"](self.lay_conv_feat.weight)
        
        print("bias", "zeros")
        SAMPLER["zeros"](self.lay_conv_feat.bias)
        
        #---------------------------------------------------
        
        print("\nlayer - pool average global")
        
        self.lay_pool_global = torch.nn.AvgPool2d(
            kernel_size = int(width)
        )
        
    
    def forward(self, x):
        
        for each_conv in self.lay_conv:
            
            x = self.lay_conv_act(each_conv(x))
        
        x = self.lay_conv_feat(x)
        
        x = self.lay_pool_global(x)
        
        return x.squeeze((1, 2, 3))
        
        