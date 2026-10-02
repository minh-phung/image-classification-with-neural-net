import torch
import numpy as np

from feature.sampler import SAMPLER
from feature.activation import ACTIVATION


class Net8(torch.nn.Module):
    
# input -> 
# [[conv -> relu]*n -> conv stride 2 -> relu]*m
# [fc -> relu]*k ->
# output

    def __init__(
        self,
        lay_conv_number, #n
        lay_conv_kernel,
        lay_stride2_number, #m
        lay_fc_number, #k
    ):
        
        print("\ninit - Net7")

        super().__init__()
        
        width = 64
        
        input_count = 3*width*width
        
        #---------------------------------------------------
        in_channel = 3
        out_channel = np.exp(np.log(input_count)/4).astype(int)
        
        
        self.lay_conv = torch.nn.ModuleList()
        self.lay_conv_act = ACTIVATION["relu"]
        
        self.lay_conv_number = lay_conv_number
        
        for i in range(lay_conv_number * lay_stride2_number):
            
            print("\nlayer - conv", i)
            print("feature", out_channel)
            print("kernel", lay_conv_kernel)
            
            conv = torch.nn.Conv2d(
                in_channels = in_channel,
                out_channels = out_channel,
                kernel_size = lay_conv_kernel,
                stride = 1,
                padding = int((lay_conv_kernel - 1)/2),
                groups = 1
            )
            
            print("weight", "kaiming_uniform")
            SAMPLER["kaiming_uniform"](conv.weight)
            
            print("bias", "constant")
            SAMPLER["constant"](conv.bias)
            
            self.lay_conv.append(conv)
            
            in_channel = out_channel
            
            print("activation", self.lay_conv_act)
            
        #---------------------------------------------------
        
        self.lay_conv_stride2 = torch.nn.ModuleList()
        self.lay_conv_stride2_act = ACTIVATION["relu"]
        
        self.lay_conv_stride2_number = lay_stride2_number
        
        for i in range(lay_stride2_number):
            
            print("\nlayer - conv stride 2:", i)
            print("feature", out_channel)
            print("kernel", lay_conv_kernel)
            
            conv = torch.nn.Conv2d(
                in_channels = out_channel,
                out_channels = out_channel,
                kernel_size = lay_conv_kernel,
                stride = 2,
                padding = 0,
                groups = 1
            )
        
            print("weight", "kaiming_uniform")
            SAMPLER["kaiming_uniform"](conv.weight)
            
            print("bias", "constant")
            SAMPLER["constant"](conv.bias)
            
            self.lay_conv_stride2.append(conv)
        
        
        for i in range(lay_stride2_number):
            width = int((width - lay_conv_kernel)/2 + 1)
        
        self.post_conv_feat_count = width*width*out_channel
        
        #---------------------------------------------------
        
        print("\nlayer - fc")
        
        self.lay_fc = torch.nn.ModuleList()
        
        in_feature = self.post_conv_feat_count
        out_feature = np.exp(np.log(input_count)/2).astype(int)
        
        self.lay_fc_act = ACTIVATION["relu"]
        
        for i in range(lay_fc_number):
            
            fc = torch.nn.Linear(in_feature, out_feature)
            
            in_feature = out_feature
            
            print("weight", "kaiming_uniform")
            SAMPLER["kaiming_uniform"](fc.weight)
            
            print("bias", "constant")
            SAMPLER["constant"](fc.bias)
            
            self.lay_fc.append(fc)
            
            print("activation", self.lay_fc_act)
            
        #---------------------------------------------------
        
        self.fc_last = torch.nn.Linear(out_feature, 1)
        
        print("\nlayer last")
        print("weight", "xavier_uniform")
        SAMPLER["xavier_uniform"](self.fc_last.weight)
        
        print("bias", "zeros")
        SAMPLER["zeros"](self.fc_last.bias)
        
        print("\n-------------\n")
    
        
    
    def forward(self, x):
        
        for i, each_conv in enumerate(self.lay_conv):
            
            x = self.lay_conv_act(each_conv(x))
        
            if (i+1) % self.lay_conv_number == 0:
                
                count = int((i+1)/self.lay_conv_number - 1) 
                
                x = self.lay_conv_stride2_act(
                    self.lay_conv_stride2[count](x)
                )
        
        x = x.view(-1, self.post_conv_feat_count)
        
        for each_fc in self.lay_fc:
            
            x = self.lay_fc_act(each_fc(x))
            
        x = self.fc_last(x)
        
        return x.squeeze(1)
        
        