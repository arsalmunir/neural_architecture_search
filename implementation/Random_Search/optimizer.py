import torch

def get_optimizer(parameters, learning_rate, **kwargs):
    return torch.optim.Adam(parameters, lr=learning_rate)

