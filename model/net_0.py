import torch

from feature.sampler import SAMPLER


class Net0(torch.nn.Module):
    
    def __init__(
        self,
        lay_1_sampler_weight,
        lay_1_sampler_bias
    ):
        
        print("\ninit - NET0")
        
        super().__init__()
        
        self.fc1 = torch.nn.Linear(3 * 64 * 64, 1)
        
        print("layer 1 weight", SAMPLER[lay_1_sampler_weight])
        SAMPLER[lay_1_sampler_weight](self.fc1.weight)
        
        print("layer 1 bias", SAMPLER[lay_1_sampler_bias])
        SAMPLER[lay_1_sampler_bias](self.fc1.bias)
        
        print("\n")
    
    def forward(self, x):
        
        x = x.view(-1, 3 * 64 * 64)
        
        x = self.fc1(x)
        
        return x.squeeze(1)