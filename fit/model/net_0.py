import torch

class Net0(torch.nn.Module):
    
    def __init__(self):
        
        print("init - NET0")
        
        super().__init__()
        
        self.fc1 = torch.nn.Linear(3 * 64 * 64, 1)
        
    def forward(self, x):
        print(x)
        x = x.view(-1, 3 * 64 * 64)
        
        x = self.fc1(x)
        
        print(x)
        
        return x.squeeze(1)