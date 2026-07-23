import torch

class Net1(torch.nn.Module):
    
    def __init__(self, act_func):
        
        super().__init__()
        
        self.fc1 = torch.nn.Linear(3 * 64 * 64, 65 * 65)
        
        self.act_func_2 = act_func
        
        self.fc_3 = torch.nn.Linear(65 * 65, 1)
        
    def forward(self, x):
        
        x = x.view(-1, 3 * 64 * 64)
        
        x = self.fc_1(x)
        
        x = act_func_2(x)
        
        x = self.fc_3(x)
        
        return 
