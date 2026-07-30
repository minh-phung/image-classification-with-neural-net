import torch

from feature.sampler import SAMPLER
from feature.activation import ACTIVATION

class Net1(torch.nn.Module):
    
    # ~ 3 * 10^7 parameters
    
    def __init__(
        self,
        lay_1_sampler_weight,
        lay_1_sampler_bias,
        lay_1_activation,
        lay_2_sampler_weight,
        lay_2_sampler_bias
    ):
        
        print("\ninit - NET1")

        super().__init__()
        
        #---------------------------------------------------
        self.fc1 = torch.nn.Linear(3 * 64 * 64, 50 * 50)
        
        print("layer 1 weight", SAMPLER[lay_1_sampler_weight])
        SAMPLER[lay_1_sampler_weight](self.fc1.weight)
        
        print("layer 1 bias", SAMPLER[lay_1_sampler_bias])
        SAMPLER[lay_1_sampler_bias](self.fc1.bias)
        
        print("layer 1 activation", ACTIVATION[lay_1_activation])
        self.act1 = ACTIVATION[lay_1_activation]
        
        #---------------------------------------------------
        
        self.fc2 = torch.nn.Linear(50 * 50, 1)
        
        print("layer 2 weight", SAMPLER[lay_2_sampler_weight])
        SAMPLER[lay_2_sampler_weight](self.fc2.weight)
        
        print("layer 2 bias", SAMPLER[lay_2_sampler_bias])
        SAMPLER[lay_2_sampler_bias](self.fc2.bias)
        
    def forward(self, x):
        
        x = x.view(-1, 3 * 64 * 64)
        
        x = self.act1(self.fc1(x))
        
        x = self.fc2(x)
        
        return x.squeeze(1)
