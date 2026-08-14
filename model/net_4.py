import torch
import numpy as np


from feature.sampler import SAMPLER
from feature.activation import ACTIVATION


class Net4(torch.nn.Module):
    
# input -> 
# [[conv -> relu]*n -> max pool]*m  ->
# [fc -> relu]*k -> 
# output
    
    def __init__(
        self,
        lay_conv_number, #n
        lay_conv_kernel,
        lay_pool_number, #m
        lay_fc_number #k
    ):
        
        print("\ninit - Net4")
        
        super().__init__() 
        
        input_count = 3*64*64
        
        
        #---------------------------------------------------
        in_channel = 3
        out_channel = np.exp(np.log(input_count)/4).astype(int)

        
        self.lay_conv = torch.nn.ModuleList()
        self.lay_conv_act = ACTIVATION["relu"]
        
        self.lay_conv_number = lay_conv_number
        
        for i in range(lay_conv_number * lay_pool_number):
            
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
        
        self.lay_pool = torch.nn.MaxPool2d(
            kernel_size = 3,
            stride = 2
        )
        
        width = 64 
        
        for i in range(lay_pool_number):
            width = int((width - 3)/2 + 1)
        
        #---------------------------------------------------
        
        self.out_pool = int(width**2 * out_channel)
        
        in_feature = self.out_pool
        out_feature = np.exp(2*np.log(input_count)/3).astype(int)
        
        self.lay_fc = torch.nn.ModuleList()
        self.lay_fc_act = ACTIVATION["relu"]
        
        for i in range(lay_fc_number):
            
            print("\nlayer - fc", i)
            print("input", in_feature, "output", out_feature)
            
            fc = torch.nn.Linear(
                in_features = in_feature,
                out_features = out_feature
            )
            
            print("weight", "kaiming_uniform")
            SAMPLER["kaiming_uniform"](fc.weight)
            
            print("bias", "constant")
            SAMPLER["constant"](fc.bias)
            
            self.lay_fc.append(fc)
            
            print("activation", self.lay_fc_act)
            
            in_feature = out_feature
        
        
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
                
                x = self.lay_pool(x)
            
        x = x.view(-1, self.out_pool)
        
        for each_fc in self.lay_fc:
            
            x = self.lay_fc_act(each_fc(x))
            
        x = self.fc_last(x)
        
        return x.squeeze(1)