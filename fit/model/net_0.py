import torch

class Net0(torch.nn.Module):
    
    def __init__(self):
        
        print("init - NET0")
        
        super().__init__()
        
        self.fc1 = torch.nn.Linear(3 * 64 * 64)
        
    def forward(self, x):
        
        x = x.view(-1, 3 * 64 * 64)
        
        x = self.fc1
        
        return x.squeeze(1)