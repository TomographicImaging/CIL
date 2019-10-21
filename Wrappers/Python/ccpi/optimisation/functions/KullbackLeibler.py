# -*- coding: utf-8 -*-
# Copyright 2019 Science Technology Facilities Council
# Copyright 2019 University of Manchester
#
# This work is part of the Core Imaging Library developed by Science Technology
# Facilities Council and University of Manchester
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy
from ccpi.optimisation.functions import Function
from ccpi.optimisation.functions.ScaledFunction import ScaledFunction 
import functools
import scipy.special
try:
    import numba
    from numba import jit, prange
    has_numba = True
    '''Some parallelisation of KL calls'''
    @jit(nopython=True)
    def kl_proximal(x,b, bnoise, tau, out):
            for i in prange(x.size):
                out.flat[i] = 0.5 *  ( 
                    ( x.flat[i] - bnoise.flat[i] - tau ) +\
                    numpy.sqrt( (x.flat[i] + bnoise.flat[i] - tau)**2. + \
                        (4. * tau * b.flat[i]) 
                    )
                )
    @jit(nopython=True)
    def kl_proximal_conjugate(x, b, bnoise, tau, out):
        #z = x + tau * self.bnoise
        #return 0.5*((z + 1) - ((z-1)**2 + 4 * tau * self.b).sqrt())

        for i in prange(x.size):
            z = x.flat[i] + ( tau * bnoise.flat[i] )
            out.flat[i] = 0.5 * ( 
                (z + 1) - numpy.sqrt((z-1)*(z-1) + 4 * tau * b.flat[i])
                )
    @jit(nopython=True)
    def kl_gradient(x, b, bnoise, out):
        for i in prange(x.size):
            out.flat[i] = 1 - b.flat[i]/(x.flat[i] + bnoise.flat[i])

    @jit(nopython=True)
    def kl_div(x, y, out):
        for i in prange(x.size):
            X = x.flat[i]
            Y = y.flat[i]    
            if x.flat[i] > 0 and y.flat[i] > 0:
                out.flat[i] = X * numpy.log(X/Y) - X + Y
            elif X == 0 and Y >= 0:
                out.flat[i] = Y
            else:
                out.flat[i] = numpy.inf
    
    # force a jit
    x = numpy.asarray(numpy.random.random((10,10)), dtype=numpy.float32)
    b = numpy.asarray(numpy.random.random((10,10)), dtype=numpy.float32)
    bnoise = numpy.zeros_like(x)
    out = numpy.empty_like(x)
    tau = 1.
    kl_div(b,x,out)
    kl_gradient(x,b,bnoise,out)
    kl_proximal(x,b, bnoise, tau, out)
    kl_proximal_conjugate(x,b, bnoise, tau, out)
    
except ImportError as ie:
    has_numba = False

class KullbackLeibler(Function):
    
    r'''Kullback-Leibler divergence function
    
         .. math::
              f(x, y) = \begin{cases} x \log(x / y) - x + y & x > 0, y > 0 \\ 
                                    y & x = 0, y \ge 0 \\
                                    \infty & \text{otherwise} 
                       \end{cases}
            
    '''
    
    def __init__(self, data, **kwargs):
        
        super(KullbackLeibler, self).__init__()
        
        self.b = data    
        self.bnoise = data * 0.
        
                                                    
    def __call__(self, x):
        

        '''Evaluates KullbackLeibler at x'''
        if has_numba:
            tmp = numpy.empty_like(x.as_array())
            kl_div(self.b.as_array(), x.as_array(), tmp)
        else:
            ind = x.as_array()>0
            tmp = scipy.special.kl_div(self.b.as_array()[ind], x.as_array()[ind])                
        return numpy.sum(tmp) 

    def log(self, datacontainer):
        '''calculates the in-place log of the datacontainer'''
        if not functools.reduce(lambda x,y: x and y>0, datacontainer.as_array().ravel(), True):
            raise ValueError('KullbackLeibler. Cannot calculate log of negative number')
        datacontainer.fill( numpy.log(datacontainer.as_array()) )

        
    def gradient(self, x, out=None):
        
        '''Evaluates gradient of KullbackLeibler at x'''
        if has_numba:
            if out is None:
                out = (x * 0.)
                out_np = out.as_array()
                kl_gradient(x.as_array(), self.b.as_array(), self.bnoise.as_array(), out_np)
                # out.fill(out_np)
                return out
            else:
                out_np = out.as_array()
                kl_gradient(x.as_array(), self.b.as_array(), self.bnoise.as_array(), out_np)
                # out.fill(out_np)
        else:
            if out is None:
                return 1 - self.b/(x + self.bnoise)
            else:

                x.add(self.bnoise, out=out)
                self.b.divide(out, out=out)
                out.subtract(1, out=out)
                out *= -1
            
    def convex_conjugate(self, x):
        
        '''Convex conjugate of KullbackLeibler at x'''
        
        xlogy = - scipy.special.xlogy(self.b.as_array(), 1 - x.as_array())
        return numpy.sum(xlogy)
            
    def proximal(self, x, tau, out=None):
        
        r'''Proximal operator of KullbackLeibler at x
           
           .. math::     prox_{\tau * f}(x)

        '''
        if has_numba:
            if out is None:
                out = (x * 0.)
                # out_np = numpy.empty_like(out.as_array(), dtype=numpy.float64)
                out_np = out.as_array()
                kl_proximal(x.as_array(), self.b.as_array(), self.bnoise.as_array(), tau, out_np)
                out.fill(out_np)
                return out
            else:
                out_np = out.as_array()
                kl_proximal(x.as_array(), self.b.as_array(), self.bnoise.as_array(), tau, out_np)
                # out.fill(out_np)                    
        else:
            if out is None:        
                return 0.5 *( (x - self.bnoise - tau) + ( (x + self.bnoise - tau)**2 + 4*tau*self.b   ) .sqrt() )
            else:
                
                tmp =  0.5 *( (x - self.bnoise - tau) + 
                            ( (x + self.bnoise - tau)**2 + 4*tau*self.b   ) .sqrt()
                            )
                x.add(self.bnoise, out=out)
                out -= tau
                out *= out
                tmp = self.b * (4 * tau)
                out.add(tmp, out=out)
                out.sqrt(out=out)
                
                x.subtract(self.bnoise, out=tmp)
                tmp -= tau
                
                out += tmp
                
                out *= 0.5
                         
    def proximal_conjugate(self, x, tau, out=None):
        
        r'''Proximal operator of the convex conjugate of KullbackLeibler at x:
           
           .. math::     prox_{\tau * f^{*}}(x)
        '''

        if has_numba:
            if out is None:
                out = (x * 0.)
                out_np = out.as_array()
                kl_proximal_conjugate(x.as_array(), self.b.as_array(), self.bnoise.as_array(), tau, out_np)
                # out.fill(out_np)
                return out
            else:
                out_np = out.as_array()
                kl_proximal_conjugate(x.as_array(), self.b.as_array(), self.bnoise.as_array(), tau, out_np)
                # out.fill(out_np)                    
        else:
            if out is None:
                z = x + tau * self.bnoise
                return 0.5*((z + 1) - ((z-1)**2 + 4 * tau * self.b).sqrt())
            else:
                
                tmp = tau * self.bnoise
                tmp += x
                tmp -= 1
                
                self.b.multiply(4*tau, out=out)    
                
                out.add(tmp.power(2), out=out)
                out.sqrt(out=out)
                out *= -1
                tmp += 2
                out += tmp
                out *= 0.5

    def __rmul__(self, scalar):
        
        '''Multiplication of KullbackLeibler with a scalar        
            
            Returns: ScaledFunction
        '''
        
        return ScaledFunction(self, scalar) 


if __name__ == '__main__':
    
    from ccpi.framework import ImageGeometry
    import numpy
    
    M, N =  2,3
    ig = ImageGeometry(voxel_num_x=M, voxel_num_y = N)
    u = ig.allocate('random_int')
    b = ig.allocate('random_int')
    u.as_array()[1,1]=0
    u.as_array()[2,0]=0
    b.as_array()[1,1]=0
    b.as_array()[2,0]=0    
    
    f = KullbackLeibler(b)
    
    
#    longest = reduce(lambda x, y: len(x) if len(x) > len(y) else len(y), strings)


#    tmp = functools.reduce(lambda x, y: \
#                           0 if x==0 and not numpy.isnan(y) else x * numpy.log(y), \
#                           zip(b.as_array().ravel(), u.as_array().ravel()),0)
    
    
#    np.multiply.reduce(X, 0)
    
    
#                sf = reduce(lambda x,y: x + y[0]*y[1],
#                            zip(self.as_array().ravel(),
#                                other.as_array().ravel()),
#                            0)        
#cdef inline number_t xlogy(number_t x, number_t y) nogil:
#    if x == 0 and not zisnan(y):
#        return 0
#    else:
#        return x * zlog(y)        
    
#    if npy_isnan(x):
#        return x
#    elif x > 0:
#        return -x * log(x)
#    elif x == 0:
#        return 0
#    else:
#        return -inf    
    
#        cdef inline double kl_div(double x, double y) nogil:
#    if npy_isnan(x) or npy_isnan(y):
#        return nan
#    elif x > 0 and y > 0:
#        return x * log(x / y) - x + y
#    elif x == 0 and y >= 0:
#        return y
#    else:
#        return inf    

    
    
    
#    def xlogy(self, dc1, dc2):
        
#        return numpy.sum(numpy.where(dc1.as_array() != 0, dc2.as_array() * numpy.log(dc2.as_array() / dc1.as_array()), 0))
        
           
    
#    f.xlog(u, b)
    
            

    
#    tmp1 = b.as_array()
#    tmp2 = u.as_array()
#    
#    zz = scipy.special.xlogy(tmp1, tmp2)
#
#    print(np.sum(zz))
    
    
#    ww = f.xlogy(b, u)
    
#    print(ww)
    
    
#cdef inline double kl_div(double x, double y) nogil:
  
    
    
        

    
    
