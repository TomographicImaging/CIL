import torch
from cil.framework import DataContainer

a = torch.zeros([2, 4], dtype=torch.float32)

print (a[0], a[1])

cil_a = DataContainer

