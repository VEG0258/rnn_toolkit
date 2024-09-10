import numpy as np
import torch
import torch.nn as nn

# When you initialize the optimizer (e.g., Adam), it gets references to the model's parameters (model.parameters()). These references remain valid throughout training.
def optimizer(model, lr):
    return torch.optim.Adam(model.parameters(), lr)