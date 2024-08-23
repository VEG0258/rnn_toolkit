import numpy as np
import torch
import torch.nn as nn

def optimizer(model, lr):
    return torch.optim.Adam(model.parameters(), lr=1e-5)