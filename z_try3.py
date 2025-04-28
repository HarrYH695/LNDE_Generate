import numpy as np
import torch

a = torch.zeros((32, 32, 3))
b = a[:, :, :-1]
print(b.shape)