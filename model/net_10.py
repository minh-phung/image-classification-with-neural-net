import torch
import numpy as np

from feature.sampler import SAMPLER
from feature.activation import ACTIVATION


class Net10(torch.nn.Module):
    
# input ->
# [
#   [ conv(1 by 1) -> relu ]*n 
#   [ conv(3 by 3) -> relu ]*n
#   [ conv(5 by 5) -> relu ]*n
#   [ conv(7 by 7) -> relu ]*n
#   max across feature -> relu  ] * m ->
# gap -> output

# inception cell
    
    def __init__(
        self
    ):
        
        print("\ninit - Net10")

        super().__init__()
        