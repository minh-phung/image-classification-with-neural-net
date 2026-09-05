import torch
import numpy as np

from feature.sampler import SAMPLER
from feature.activation import ACTIVATION



class Net5(torch.nn.Module):
    
# input -> 
# [[conv -> relu]*n -> max pool]*m  ->
# [conv(1*1), 1 feat -> relu]*k -> gap ->
# output

# Springenberg All Convolutional
    
    def __init__(
        self,
        lay_conv_number, #n
        lay_conv_kernel,
        lay_pool_number, #m
        lay_conv1_number #k
    ):
        
        print("\ninit - Net5")
        
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
        
        print("\npooling", self.lay_pool)
        
        width = 64 
        
        for i in range(lay_pool_number):
            width = int((width - 3)/2 + 1)
        
        #---------------------------------------------------
        
        print("\nlayer - conv - 1*1")
        
        self.lay_conv1 = torch.nn.ModuleList()
        
        #self.lay_conv1_act = ACTIVATION["relu"]
        
        in_channel = out_channel
        out_channel = out_channel
        
        for i in range(lay_conv1_number):
            
            if i+1 == lay_conv1_number:
                out_channel = 1
            
            conv1 = torch.nn.Conv2d(
                in_channels = in_channel,
                out_channels = out_channel,
                kernel_size = 1
            )
            
            print("weight", "xavier_uniform")
            SAMPLER["xavier_uniform"](conv1.weight)
        
            print("bias", "zeros")
            SAMPLER["zeros"](conv1.bias)
            
            self.lay_conv1.append(conv1)
        
            #print("activation", self.lay_conv1_act)
        
        #---------------------------------------------------
        
        print("\nlayer - pool average global")
        
        self.lay_pool_global = torch.nn.AvgPool2d(
            kernel_size = int(width)
        )
        
        
        
        
        
    def forward(self, x):
        
        for i, each_conv in enumerate(self.lay_conv):
            
            x = self.lay_conv_act(each_conv(x))
            
            if (i+1) % self.lay_conv_number == 0:
                
                x = self.lay_pool(x)
        
        for each_conv1 in self.lay_conv1:
            
            #x = self.lay_conv1_act(each_conv1(x))
            x = each_conv1(x)
        
        x = self.lay_pool_global(x)
        
        return x.squeeze((1, 2, 3))