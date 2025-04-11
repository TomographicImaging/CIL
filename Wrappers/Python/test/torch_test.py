import torch
import numpy 
from cil.framework import DataContainer


b = numpy.ones([2,4], dtype=numpy.float32)
cil_np0 = DataContainer(b, deep_copy=True)
cil_np1 = DataContainer(b, deep_copy=False)

numpy.testing.assert_allclose(numpy.zeros([2,4], dtype=numpy.float32), 
                              (cil_np0-cil_np1).as_array())



a = [ torch.ones([2, 4], dtype=torch.float32),
      torch.ones([2, 4], dtype=torch.float32),
      torch.ones([2, 4], dtype=torch.float32)]


print (f"torch array has __array_interface__? {hasattr(a, '__array_interface__')}")
# https://github.com/pytorch/pytorch/issues/54138

# print (a[0], a[1])
cil_tc0 = DataContainer(a[0], deep_copy=True)
cil_tc1 = DataContainer(a[0], deep_copy=False)

numpy.testing.assert_allclose(numpy.zeros([2,4], dtype=numpy.float32), 
                              (cil_tc0-cil_tc1).as_array().numpy())

# Torch - NumPy = Torch
c = a[0] - b
print(c)

# NumPy - Torch = TypeError: unsupported operand type(s) for -: 'numpy.ndarray' and 'Tensor'
# d = b - a = TypeError: unsupported operand type(s) for -: 'numpy.ndarray' and 'Tensor'
# in CIL we do subtraction as below and it works
d = numpy.subtract(b,a[0])
print("d", d)

# CIL DataContainer and mixed stuff
mixed_diff = cil_np0.subtract(cil_tc0)
print (mixed_diff.array)
print(f"Type of mixed_diff cil_np0.subtract(cil_tc0) {type(mixed_diff.array)}")
numpy.testing.assert_allclose(numpy.zeros([2,4], dtype=numpy.float32), 
                              (mixed_diff).as_array().numpy())

cil_tc2 = DataContainer(a[1], deep_copy=False)
print(f"cil_tc2.array ype of mixed_diff {type(cil_tc2.array)}")

# cil_tc0.subtract(cil_tc1, out=cil_tc2.array)
from array_api_compat import array_namespace
# array_namespace will raise a TypeError if multiple namespaces for array input
xp = array_namespace(*a)
print(xp)
xp.subtract(a[1], a[0], out=a[2])
print ("after subtract", a[2])

cil_tc2 = DataContainer(a[2], deep_copy=False)
cil_tc2 += 1

print("################")
print (cil_tc2.array)
cil_tc0.subtract(cil_tc1, out=cil_tc2)

print (cil_tc2.array)
# from array_api_compat import array_namespace
# array_namespace([1,2])
cil_tc2+=1
# mixed types with output
# output must match the type of the first array
print("################")
print (cil_np1.array)
mixed_diff = cil_np0.subtract(cil_tc0, out=cil_np1)
print (cil_np1.array)

print("########## test sapyb with pytorch ########## ")
a = [ torch.ones([2, 4], dtype=torch.float32),
      torch.ones([2, 4], dtype=torch.float32),
      torch.ones([2, 4], dtype=torch.float32)]

cil_tc = [ DataContainer(el, deep_copy=False) for el in a]

# 2 * 1 + 3 * 1
res = cil_tc[1].sapyb(2, cil_tc[1], 3)
cil_tc[1].sapyb(2, cil_tc[1], 3, out=cil_tc[2])

numpy.testing.assert_allclose(cil_tc[2].as_array().numpy(), 
                              (res).as_array().numpy())

print(res.array)

print("########## test sapyb with numpy ########## ")
b = [ numpy.ones([2, 4], dtype=numpy.float32),
      numpy.ones([2, 4], dtype=numpy.float32),
      numpy.ones([2, 4], dtype=numpy.float32)]

cil_np = [ DataContainer(el, deep_copy=False) for el in b]

# 2 * 1 + 3 * 1
res = cil_np[1].sapyb(2, cil_np[1], 3)
cil_np[1].sapyb(2, cil_np[1], 3, out=cil_np[2])

numpy.testing.assert_allclose(cil_np[2].as_array(), 
                              (res).as_array())

print(res.array)