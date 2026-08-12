import torch

class Net5(torch.nn.Module):
    
# input -> 
# [[conv -> relu]*n -> max pool]*m  ->
#  -> conv(1*1), 1 feat -> gap ->
# output
    
    def __init__(
        self,
        lay_conv_number, #n
        lay_conv_kernel,
        lay_pool_number, #m
        lay_fc_number
    ):
        
        print("\ninit - Net4")
        
        super().__init__() 
        