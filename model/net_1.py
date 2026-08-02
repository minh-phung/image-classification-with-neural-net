import torch
import numpy as np

from feature.sampler import SAMPLER
from feature.activation import ACTIVATION

class Net1(torch.nn.Module):
    
    # input -> [ fc -> activation ]*n -> output
    
    
    def __init__(
        self,
        lay_hidden_number, # n
        lay_sampler_weight,
        lay_sampler_bias,
        lay_activation
    ):
        
        print("\ninit - NET1")

        super().__init__()
        
        #---------------------------------------------------
        self.lay = [None]*lay_hidden_number
        self.activation = ACTIVATION[lay_activation]
        
        input_count = 3*64*64
        
        lay_count = np.exp( np.linspace(
            np.log(input_count),
            0,
            lay_hidden_number + 2
        )[1:-1]).astype(int)
        
        
        lay_in = 3*64*64
        
        for i in range(lay_hidden_number):
            print("\nlayer", i)
            print("input", lay_in, "output", lay_count[i])
            
            self.lay[i] = torch.nn.Linear(
                lay_in,
                lay_count[i]
            )
            
            print("weight", SAMPLER[lay_sampler_weight])
            SAMPLER[lay_sampler_weight](self.lay[i].weight)
            
            print("bias", SAMPLER[lay_sampler_bias])
            SAMPLER[lay_sampler_bias](self.lay[i].bias)
            
            print("activation", self.activation)
            
            
            lay_in = lay_count[i]
        
        #---------------------------------------------------
        
        self.fc_last = torch.nn.Linear(lay_count[-1], 1)
        
        print("\nlayer last")
        print("weight", "xavier_uniform")
        SAMPLER["xavier_uniform"](self.fc_last.weight)
        
        print("bias", "zeros")
        SAMPLER["zeros"](self.fc_last.bias)
        
        print("\n-------------\n")
        
        
    def forward(self, x):
        
        x = x.view(-1, 3 * 64 * 64)
        
        for each_lay in self.lay:
            
            x = self.activation((each_lay(x)))
        
        x = self.fc_last(x)
        
        return x.squeeze(1)
