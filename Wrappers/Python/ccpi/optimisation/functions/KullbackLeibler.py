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

class KullbackLeibler(Function):
    
    r'''Kullback-Leibler divergence function
    
         .. math::
              f(x, y) = \begin{cases} x \log(x / y) - x + y & x > 0, y > 0 \\ 
                                    y & x = 0, y \ge 0 \\
                                    \infty & \text{otherwise} 
                       \end{cases}
            
    '''
    
    def __init__(self, data = None, **kwargs):
        
        super(KullbackLeibler, self).__init__()
        
        self.data = data    
        self.background_term = kwargs.get('background_term',data * 0.0)
        
        if self.data is None:
            raise ValueError('Please define data')
            
        if self.data.as_array().any()<0:
            raise ValueError('Data is larger or equal to 0')
            
        
                                                    
    def __call__(self, x):
        

        '''Evaluates KullbackLeibler at x'''
        
        tmp_sum = (x + self.background_term).as_array()
        ind = tmp_sum > 0
        tmp = scipy.special.kl_div(self.data.as_array()[ind], tmp_sum[ind])                
        return numpy.sum(tmp) 

    def log(self, datacontainer):
        '''calculates the in-place log of the datacontainer'''
        if not functools.reduce(lambda x,y: x and y>0, datacontainer.as_array().ravel(), True):
            raise ValueError('KullbackLeibler. Cannot calculate log of negative number')
        datacontainer.fill( numpy.log(datacontainer.as_array()) )

        
    def gradient(self, x, out=None):
        
        '''Evaluates gradient of KullbackLeibler at x'''
        
        tmp_sum = (x + self.background_term).as_array()
        ind = tmp_sum>0 
        
        if out is None:            
            return 1 - self.data[ind]/tmp_sum[ind]
        else:
            
            # TODO not working with ind
            x.add(self.background_term, out=out)
            self.data.divide(out, out=out)
            out.subtract(1, out=out)
            out *= -1
            
    def convex_conjugate(self, x):
        
        '''Convex conjugate of KullbackLeibler at x'''
        
        xlogy = - scipy.special.xlogy(self.data.as_array(), 1 - x.as_array())         
        return numpy.sum(xlogy) - (self.background_term * x).sum()
            
    def proximal(self, x, tau, out=None):
        
        r'''Proximal operator of KullbackLeibler at x
           
           .. math::     prox_{\tau * f}(x)

        '''

        if out is None:        
            return 0.5 *( (x - self.background_term - tau) + ( (x + self.background_term - tau)**2 + 4*tau*self.data   ) .sqrt() )
        else:
            
            tmp =  0.5 *( (x - self.background_term - tau) + 
                        ( (x + self.background_term - tau)**2 + 4*tau*self.data   ) .sqrt()
                        )
            x.add(self.background_term, out=out)
            out -= tau
            out *= out
            tmp = self.data * (4 * tau)
            out.add(tmp, out=out)
            out.sqrt(out=out)
            
            x.subtract(self.background_term, out=tmp)
            tmp -= tau
            
            out += tmp
            
            out *= 0.5
                            
    def proximal_conjugate(self, x, tau, out=None):
        
        r'''Proximal operator of the convex conjugate of KullbackLeibler at x:
           
           .. math::     prox_{\tau * f^{*}}(x)
        '''

                
        if out is None:
            z = x + tau * self.background_term
            return 0.5*((z + 1) - ((z-1)**2 + 4 * tau * self.data).sqrt())
        else:
            
            tmp = tau * self.background_term
            tmp += x
            tmp -= 1
            
            self.data.multiply(4*tau, out=out)    
            
            out.add((tmp)**2, out=out)
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
    
    # if both u1, g1 > 0
    u1 = ig.allocate('random_int', seed = 5)
    g1 = ig.allocate('random_int', seed = 10)
    
    f = KullbackLeibler(g1)
#    print(f(g1))
#    print(f(u1))
    
    g2 = g1.clone()
    g2.as_array()[0,1] = 0
#    print(f(g2))


    tmp = scipy.special.kl_div(g1.as_array(), g2.as_array())  
    
        

    
        
