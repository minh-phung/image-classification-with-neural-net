import torch

class Net4(torch.nn.Module):
    
    # input -> [conv -> relu]*n -> [fc -> relu]*m -> output
    
    
    def __init__(self):
        
        print("\ninit - Net4")
        
        super().__init__()
        
        