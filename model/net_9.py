import torch
import numpy as np

from feature.sampler import SAMPLER
from feature.activation import ACTIVATION


class Net9(torch.nn.Module):
    
# input ->
# [
#   [ conv(1 by 1) -> relu ]*n 
#   [ conv(3 by 3) -> relu ]*n
#   [ conv(5 by 5) -> relu ]*n
#   [ conv(7 by 7) -> relu ]*n
#   -> conv(1 by 1) -> relu       ] * m ->
# gap -> output

# inception cell
    
    def __init__(
        self,
        cell_number, # n
        conv1_number, #m
    ):
        
        print("\ninit - Net9")

        super().__init__()
        
        width = 64
        input_count = 3*width*width
        
        #---------------------------------------------------
        in_channel = 3
        out_channel = np.exp(np.log(input_count)/4).astype(int)
        
        self.cell_number = cell_number
        self.conv1_number = conv1_number
        
        self.cell_kernel = [1, 3, 5, 7]
        
        
        self.cell_conv = torch.nn.ModuleList()
        
        for i in range(len(self.cell_kernel)):
            self.cell_conv.append(torch.nn.ModuleList())
        
        
        self.conv_count = cell_number*conv1_number # within cells only
        
        for i in range(self.conv_count):
            
            for j, each_kernel in enumerate(self.cell_kernel):
                
                conv = torch.nn.Conv2d(
                    in_channels = in_channel,
                    out_channels = out_channel,
                    kernel_size = each_kernel,
                    stride = 1,
                    padding = int((each_kernel - 1)/2),
                    groups = 1
                )
                
                SAMPLER["kaiming_uniform"](conv.weight)
                SAMPLER["constant"](conv.bias)
                
                self.cell_conv[j].append(conv)
                
            if (i+1) % self.cell_number == 0:
                in_channel = 1
                
            else:
                in_channel = out_channel
                
        
        self.cell_conv_act = ACTIVATION["relu"]
        
        
        for i in range(len(self.cell_kernel)):
            
            print("\nkernel", i)
            
            print(self.cell_conv[i])
        
            
        #---------------------------------------------------

        self.conv1_lay = torch.nn.ModuleList()
        
        for i in range(conv1_number):
            
            conv = torch.nn.Conv2d(
                in_channels = out_channel * len(self.cell_kernel),
                out_channels = 1,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                groups = 1
            )
        
            SAMPLER["kaiming_uniform"](conv.weight)
            SAMPLER["constant"](conv.bias)
    
            self.conv1_lay.append(conv)
        
        #---------------------------------------------------

        self.pool_global =  torch.nn.AvgPool2d(
            kernel_size = int(width)
        )
        
            
    
    def forward(self, x):
        
        branch = [x]*len(self.cell_kernel)
        
        for i in range(self.conv_count):
            
            for j in range(len(self.cell_kernel)):
                
                branch[j] = self.cell_conv_act(self.cell_conv[j][i](branch[j]))
                
            if (i+1) % self.cell_number == 0:
                
                count = int(((i+1) / self.cell_number) - 1)
                
                x = torch.cat(branch, dim = 1)
                
                x = self.cell_conv_act(self.conv1_lay[count](x))
                
                branch = [x]*len(self.cell_kernel)
            
        
        x = self.pool_global(x)
        
        return x.squeeze(1, 2, 3)