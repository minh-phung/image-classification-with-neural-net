import torch

from feature.sampler import SAMPLER


class Net0(torch.nn.Module):
    
    # input -> output
    
    def __init__(
        self
    ):
        
        print("\ninit - NET0")
        
        super().__init__()
        
        self.fc1 = torch.nn.Linear(3 * 64 * 64, 1)
        
        print("layer 1 weight", "xavier_uniform")
        SAMPLER["xavier_uniform"](self.fc1.weight)
        
        print("layer 1 bias", "zeros")
        SAMPLER["zeros"](self.fc1.bias)
        
    
    def forward(self, x):
        
        x = x.view(-1, 3 * 64 * 64)
        
        x = self.fc1(x)
        
        return x.squeeze(1)