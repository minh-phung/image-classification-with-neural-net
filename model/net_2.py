import torch

from feature.sampler import SAMPLER
from feature.activation import ACTIVATION

class Net2(torch.nn.Module):
    
    def __init__(
        self,
        lay_1_sampler_weight,
        lay_1_sampler_bias,
        lay_1_activation,
        lay_2_sampler_weight,
        lay_2_sampler_bias,
        lay_2_activation,
        lay_3_sampler_weight,
        lay_3_sampler_bias
    ):
        
        print("\ninit - Net2")
        
        super().__init__()
        
        #---------------------------------------------------
        self.fc1 = torch.nn.Linear(3 * 64 * 64, 9 * 9)
        
        print("layer 1 weight", SAMPLER[lay_1_sampler_weight])
        SAMPLER[lay_1_sampler_weight](self.fc1.weight)
        
        print("layer 1 bias", SAMPLER[lay_1_sampler_bias])
        SAMPLER[lay_1_sampler_bias](self.fc1.bias)
        
        print("layer 1 activation", ACTIVATION[lay_1_activation])
        self.act1 = ACTIVATION[lay_1_activation]
        
        #---------------------------------------------------
        self.fc2 = torch.nn.Linear(9 * 9, 5 * 5)
        
        print("layer 2 weight", SAMPLER[lay_2_sampler_weight])
        SAMPLER[lay_2_sampler_weight](self.fc2.weight)
        
        print("layer 2 bias", SAMPLER[lay_2_sampler_bias])
        SAMPLER[lay_2_sampler_bias](self.fc2.bias)
        
        print("layer 2 activation", ACTIVATION[lay_2_activation])
        self.act2 = ACTIVATION[lay_2_activation]
        
        #---------------------------------------------------
        self.fc3 = torch.nn.Linear(5 * 5, 1)
        
        print("layer 3 weight", SAMPLER[lay_3_sampler_weight])
        SAMPLER[lay_3_sampler_weight](self.fc3.weight)
        
        print("layer 3 bias", SAMPLER[lay_3_sampler_bias])
        SAMPLER[lay_3_sampler_bias](self.fc3.bias)
        
        
    def forward(self, x):
        
        x = x.view(-1, 3 * 64 * 64)
        
        x = self.act1(self.fc1(x))
        
        x = self.act2(self.fc2(x))
        
        x = self.fc3(x)
        
        return x.squeeze(1)
        
        
