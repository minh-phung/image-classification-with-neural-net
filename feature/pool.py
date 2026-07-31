import torch

POOL = {
    "max":  torch.nn.MaxPool2d(),
    "avg":  torch.nn.AvgPool2d()
}