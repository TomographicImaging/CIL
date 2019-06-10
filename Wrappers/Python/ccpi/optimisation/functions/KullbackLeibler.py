# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library developed by
#   Visual Analytics and Imaging System Group of the Science Technology
#   Facilities Council, STFC

#   Copyright 2018-2019 Evangelos Papoutsellis and Edoardo Pasca

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy
from ccpi.optimisation.functions import Function
from ccpi.optimisation.functions.ScaledFunction import ScaledFunction 
import functools
import scipy.special

class KullbackLeibler(Function):
    
    ''' 
    
    KL_div(x, y + back) = int x * log(x/(y+back)) - x + (y+back)
    
    Assumption: y>=0
                back>=0
                
    '''
    
    def __init__(self, data, **kwargs):
        
        super(KullbackLeibler, self).__init__()
        
        self.b = data    
        self.bnoise = 0
        
                                                    
    def __call__(self, x):
        
        
        '''
        
            x - y * log( x + bnoise) + y * log(y) - y + bnoise
        
        
        '''
        
        ind = x.as_array()>0
        tmp = scipy.special.kl_div(self.b.as_array()[ind], x.as_array()[ind])                
        return numpy.sum(tmp) 
          

    def log(self, datacontainer):
        '''calculates the in-place log of the datacontainer'''
        if not functools.reduce(lambda x,y: x and y>0,
                                datacontainer.as_array().ravel(), True):
            raise ValueError('KullbackLeibler. Cannot calculate log of negative number')
        datacontainer.fill( numpy.log(datacontainer.as_array()) )

        
    def gradient(self, x, out=None):
        
        if out is None:
            return 1 - self.b/(x + self.bnoise)
        else:

            x.add(self.bnoise, out=out)
            self.b.divide(out, out=out)
            out.subtract(1, out=out)
            out *= -1
            
    def convex_conjugate(self, x):
        
        xlogy = - scipy.special.xlogy(self.b.as_array(), 1 - x.as_array())
        return numpy.sum(xlogy)
            
    def proximal(self, x, tau, out=None):
        
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

                
        if out is None:
            z = x + tau * self.bnoise
            return 0.5*((z + 1) - ((z-1)**2 + 4 * tau * self.b).sqrt())
        else:
            
            #tmp = x + tau * self.bnoise
            tmp = tau * self.bnoise
            tmp += x
            tmp -= 1
            
            self.b.multiply(4*tau, out=out)    
            
            out.add((tmp)**2, out=out)
            out.sqrt(out=out)
            out *= -1
            tmp += 2
            out += tmp
            out *= 0.5

    def __rmul__(self, scalar):
        
        ''' Multiplication of L2NormSquared with a scalar
        
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
  
    
    
        

    
        
