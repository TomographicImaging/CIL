import numpy
from PIL import Image

import matplotlib.pyplot as plt
import time
from ccpi.framework import ImageGeometry
from ccpi.optimisation.operators import FiniteDiff

import os
#os.environ["OMP_NUM_THREADS"] = "8"

import ctypes, platform

# check for the extension
if platform.system() == 'Linux':
    dll = 'libcilacc.so'
elif platform.system() == 'Windows':
    dll = 'cilacc.dll'
elif platform.system() == 'Darwin':
    dll = 'libcilacc.dylib'
else:
    raise ValueError('Not supported platform, ', platform.system())

#print ("dll location", dll)
FD = ctypes.cdll.LoadLibrary(dll)

c_float_p = ctypes.POINTER(ctypes.c_float)
FD.openMPtest.restypes = ctypes.c_int32
FD.fdiff2D.argtypes = [ctypes.POINTER(ctypes.c_float),
                       ctypes.POINTER(ctypes.c_float),
                       ctypes.POINTER(ctypes.c_float),
                       ctypes.c_long,
                       ctypes.c_long,
                       ctypes.c_int32,
                       ctypes.c_int32]
#set up
nx = 2048
ny = 2048

niters = 10

print("thread count: ", FD.openMPtest())
print("Iterations: ", niters)

dim = [ny, nx]

size = nx * ny

nDim = len(dim)
#
arr = numpy.random.random_integers(0, 65535, size).reshape(dim)
arr = arr.astype(numpy.float32)

ig = ImageGeometry(voxel_num_x=nx, voxel_num_y=ny)

data = ig.allocate()

out_dx_gold = ig.allocate()
out_dy_gold = ig.allocate()
adjoint_gold = ig.allocate()

out_dx = numpy.empty_like(arr)
out_dy = numpy.empty_like(arr)
adjoint_out = numpy.empty_like(arr)

arr_p = arr.ctypes.data_as(c_float_p)

out_dx_p = out_dx.ctypes.data_as(c_float_p)
out_dy_p = out_dy.ctypes.data_as(c_float_p)
adjoint_out_p = adjoint_out.ctypes.data_as(c_float_p)

data.fill(arr)


print("2D")

print("Array size: ", ny, " x ", nx)
#print("input")
#print(arr)

print("Neumann")
fdx = FiniteDiff(data.geometry, direction=1, bnd_cond='Neumann')
fdy = FiniteDiff(data.geometry, direction=0, bnd_cond='Neumann')


#framework, neumman, direct
t0 = time.time()
for i in range(niters):
    fdx.direct(data, out=out_dx_gold)
    fdy.direct(data, out=out_dy_gold)

t_fw_d = (time.time() - t0)

if t_fw_d:
    print("Framework, direct", ", ", 1, ", ", 1000*t_fw_d / niters)

#framework, neumman, adjoint
t0 = time.time()
for i in range(niters):
    adjoint_gold = fdx.adjoint(out_dx_gold)
    adjoint_gold += fdy.adjoint(out_dy_gold)

t_fw_a = (time.time() - t0)

if t_fw_a:
    print("Framework, adjoint", ", ", 1, ", ", 1000*t_fw_a / niters)

#C, neumann, direct
out_dx.fill(0)
out_dy.fill(0)

t0 = time.time()
for i in range(niters):
    FD.fdiff2D(arr_p, out_dx_p, out_dy_p, nx, ny, 0, 1)

t1 = time.time() - t0

if t1:
    print("C, direct", ", ", t_fw_d/t1, ", ", 1000*t1 / niters)

try:
    numpy.testing.assert_array_almost_equal(out_dx, out_dx_gold.array)
    numpy.testing.assert_array_almost_equal(out_dy, out_dy_gold.array)
    print("Passed")
except AssertionError as ae:
    print ("The output does not match the framework")
    print (ae)

t0 = time.time()
for i in range(niters):
    FD.fdiff2D(adjoint_out_p, out_dx_p, out_dy_p, nx, ny, 0, 0)

t1 = time.time() - t0

if t1:
    print("C, adjoint", ", ", t_fw_a/t1, ", ", 1000*t1 / niters)

try:
    numpy.testing.assert_array_almost_equal(adjoint_out, adjoint_gold.array)
    print("Passed")
except AssertionError as ae:
    print ("The output does not match the framework")
    print (ae)



#####

print("Periodic")
fdx = FiniteDiff(data.geometry, direction=1, bnd_cond='Periodic')
fdy = FiniteDiff(data.geometry, direction=0, bnd_cond='Periodic')


#framework, Periodic, direct
t0 = time.time()
for i in range(niters):
    fdx.direct(data, out=out_dx_gold)
    fdy.direct(data, out=out_dy_gold)

t_fw_d = (time.time() - t0)

if t_fw_d:
    print("Framework, direct", ", ", 1, ", ", 1000*t_fw_d / niters)

#framework, neumman, adjoint
t0 = time.time()
for i in range(niters):
    adjoint_gold = fdx.adjoint(out_dx_gold)
    adjoint_gold += fdy.adjoint(out_dy_gold)

t_fw_a = (time.time() - t0)

if t_fw_a:
    print("Framework, adjoint", ", ", 1, ", ", 1000*t_fw_a / niters)


#C, Periodic, direct
out_dx.fill(0)
out_dy.fill(0)

t0 = time.time()
for i in range(niters):
    FD.fdiff2D(arr_p, out_dx_p, out_dy_p, nx, ny, 1, 1)

t1 = time.time() - t0

if t1:
    print("C, direct", ", ", t_fw_d/t1, ", ", 1000*t1 / niters)

try:
    numpy.testing.assert_array_almost_equal(out_dx, out_dx_gold.array)
    numpy.testing.assert_array_almost_equal(out_dy, out_dy_gold.array)
    print("Passed")
except AssertionError as ae:
    print ("The output does not match the framework")
    print (ae)


t0 = time.time()
for i in range(niters):
    FD.fdiff2D(adjoint_out_p, out_dx_p, out_dy_p, nx, ny, 1, 0)

t1 = time.time() - t0

if t1:
    print("C, adjoint", ", ", t_fw_a/t1, ", ", 1000*t1 / niters)

try:
    numpy.testing.assert_array_almost_equal(adjoint_out, adjoint_gold.array)
    print("Passed")
except AssertionError as ae:
    print ("The output does not match the framework")
    print (ae)
