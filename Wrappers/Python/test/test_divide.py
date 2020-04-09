from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import numpy as np
import sys
from datetime import timedelta, datetime
import warnings
from functools import reduce
from numbers import Number
import ctypes, platform
from ccpi.utilities import NUM_THREADS

# dll = os.path.abspath(os.path.join( 
#          os.path.abspath(os.path.dirname(__file__)),
#          'libfdiff.dll')
# )

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
cilacc = ctypes.cdll.LoadLibrary(dll)

def divide(x, y, out, default_value, is_0by0=False, dtype=numpy.float32, num_threads=NUM_THREADS):
    '''performs division with cilacc C library
    
    Does the operation .. math:: a*x+b*y and stores the result in out, where x is self

    :param a: scalar
    :type a: float
    :param b: scalar
    :type b: float
    :param y: DataContainer
    :param out: DataContainer instance to store the result
    :param dtype: data type of the DataContainers
    :type dtype: numpy type, optional, default numpy.float32
    :param num_threads: number of threads to run on
    :type num_threads: int, optional, default 1/2 CPU of the system
    '''

    c_float_p = ctypes.POINTER(ctypes.c_float)
    c_double_p = ctypes.POINTER(ctypes.c_double)
    c_int_p = ctypes.POINTER(ctypes.c_int)
    # get the reference to the data
    # ndx = x.as_array()
    # ndy = y.as_array()
    # ndout = out.as_array()
    ndx = x
    ndy = y
    ndout = out

    if is_0by0:
        zero_by_zero = numpy.int(1)
    else:
        zero_by_zero = numpy.int(0)

    if ndx.dtype != dtype:
        ndx = ndx.astype(dtype)
    if ndy.dtype != dtype:
        ndy = ndy.astype(dtype)
    
    if dtype == numpy.float32:
        x_p = ndx.ctypes.data_as(c_float_p)
        y_p = ndy.ctypes.data_as(c_float_p)
        out_p = ndout.ctypes.data_as(c_float_p)
        default_value = numpy.float32(default_value)
        f = cilacc.fdivide

    elif dtype == numpy.float64:
        default_value = numpy.float64(default_value)
        x_p = ndx.ctypes.data_as(c_double_p)
        y_p = ndy.ctypes.data_as(c_double_p)
        out_p = ndout.ctypes.data_as(c_double_p)
        f = cilacc.ddivide
    elif dtype == numpy.int32:
        default_value = numpy.int(default_value)
        x_p = ndx.ctypes.data_as(c_int_p)
        y_p = ndy.ctypes.data_as(c_int_p)
        out_p = ndout.ctypes.data_as(c_int_p)
        f = cilacc.idivide
    else:
        raise TypeError('Unsupported type {}. Expecting numpy.float32 or numpy.float64 or numpy.int32'.format(dtype))

    #out = numpy.empty_like(a)

    
    # int psaxpby(float * x, float * y, float * out, float a, float b, long size)
    cilacc.fdivide.argtypes = [ctypes.POINTER(ctypes.c_float),  # pointer to the first array 
                                ctypes.POINTER(ctypes.c_float),  # pointer to the second array 
                                ctypes.POINTER(ctypes.c_float),  # pointer to the third array 
                                ctypes.c_float,                  # type of default value
                                ctypes.c_int,                  # type of is_zero_by_zero (int)
                                ctypes.c_long,                   # type of size of first array 
                                ctypes.c_int]                    # number of threads
    cilacc.ddivide.argtypes = [ctypes.POINTER(ctypes.c_double),  # pointer to the first array 
                                ctypes.POINTER(ctypes.c_double),  # pointer to the second array 
                                ctypes.POINTER(ctypes.c_double),  # pointer to the third array 
                                ctypes.c_double,                  # type of default value
                                ctypes.c_int,                  # type of is_zero_by_zero (int)
                                ctypes.c_long,                   # type of size of first array 
                                ctypes.c_int]                    # number of threads
    cilacc.idivide.argtypes = [ctypes.POINTER(ctypes.c_int),  # pointer to the first array 
                                ctypes.POINTER(ctypes.c_int),  # pointer to the second array 
                                ctypes.POINTER(ctypes.c_int),  # pointer to the third array 
                                ctypes.c_int,                  # type of default value
                                ctypes.c_int,                  # type of is_zero_by_zero (int)
                                ctypes.c_long,                   # type of size of first array 
                                ctypes.c_int]                    # number of threads

    if f(x_p, y_p, out_p, default_value, zero_by_zero, ndx.size, num_threads) != 0:
        raise RuntimeError('axpby execution failed')

if __name__ == '__main__':

    dtypes = [np.float32, np.float64, np.int32]

    for dtype in dtypes:
        a = np.asarray([0,-2,3,0], dtype=dtype)
        b = np.asarray([-4,0,3,0], dtype=dtype)

        c1 = np.divide(a,b)
        print (c1)
        ## handle division by zero and 0/0
        default_div_by_0 = 5
        c = np.divide(np.where(b == 0, np.ones_like(b) * default_div_by_0, a),
                    np.where(b == 0, np.ones_like(b), b ))
        print (c)

        ## handle 0/0
        default_div_0by0 = 10
        c = np.where(np.isnan(c1), np.ones_like(c1) * default_div_0by0, c1)

        print (c)

        ## handle division by zero
        c = np.where(np.isinf(c1), np.ones_like(c1) * default_div_by_0, c1)

        print (c)
        out = np.empty_like(a)
        divide(a,b,out, 11, False, a.dtype)
        print ("out", out)

        divide(a,b,out, 12, True, a.dtype)
        print ("out", out)