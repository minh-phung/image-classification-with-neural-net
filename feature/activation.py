import torch

ACTIVATION = {
    "identity": torch.nn.Identity(),
    "sigmoid": torch.nn.Sigmoid(),
    "tanh": torch.nn.Tanh(),
    "relu": torch.nn.ReLU()
}