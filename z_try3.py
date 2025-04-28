import numpy as np
import torch

a = torch.zeros((32, 32, 2))
b = a[:, :, 1:]
print(b.shape)