import torch
import numpy 
from cil.framework import DataContainer


b = numpy.ones([2,4], dtype=numpy.float32)
cil_np0 = DataContainer(b, deep_copy=True)
cil_np1 = DataContainer(b, deep_copy=False)

numpy.testing.assert_allclose(numpy.zeros([2,4], dtype=numpy.float32), 
                              (cil_np0-cil_np1).as_array())
a = torch.zeros([2, 4], dtype=torch.float32)

print (f"torch array has __array_interface__? {hasattr(a, '__array_interface__')}")
# https://github.com/pytorch/pytorch/issues/54138

# print (a[0], a[1])
cil_tc0 = DataContainer(a, deep_copy=True)
cil_tc1 = DataContainer(a, deep_copy=False)

numpy.testing.assert_allclose(numpy.zeros([2,4], dtype=numpy.float32), 
                              (cil_tc0-cil_tc1).as_array().numpy())
