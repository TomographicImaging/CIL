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
    
    def __init__(self,  **kwargs):
        
        super(KullbackLeibler, self).__init__(L = None)        
        self.b = kwargs.get('b', None) 
        self.eta = kwargs.get('eta', None) 
        
#        if self.b is None:
#            self.eta = kwargs.get('eta',0)
#        else:
#            self.eta = kwargs.get('eta',self.b.allocate())
        if self.b is not None:
            if self.b.as_array().any()<0:
                raise ValueError('Data should be is larger or equal to 0')              
                                
        #if self.b is None:
        #    raise ValueError('Please define data')
                
                                      
                                                    
    def __call__(self, x):
        

        '''Evaluates KullbackLeibler at x'''
        
        tmp_sum = x.as_array()
        tmp_data = x.geometry.allocate()
        if self.eta is not None:
            tmp_sum += self.eta.as_array()
            
        if self.b is not None:
            tmp_data = self.b          
                                                     
        ind = tmp_sum > 0
        tmp = scipy.special.kl_div(tmp_data.as_array()[ind], tmp_sum[ind])                
        return numpy.sum(tmp) 

    def log(self, datacontainer):
        '''calculates the in-place log of the datacontainer'''
        if not functools.reduce(lambda x,y: x and y>0, datacontainer.as_array().ravel(), True):
            raise ValueError('KullbackLeibler. Cannot calculate log of negative number')
        datacontainer.fill( numpy.log(datacontainer.as_array()) )

        
    def gradient(self, x, out=None):
        
        '''Evaluates gradient of KullbackLeibler at x'''
        
        tmp_sum = x.as_array()
        tmp_data = x.geometry.allocate()
        if self.eta is not None:
            tmp_sum += self.eta.as_array()
            
        if self.b is not None:
            tmp_data = self.b              

        tmp_sum_array = tmp_sum  
        tmp_out = x.geometry.allocate()        
        tmp_out.as_array()[tmp_sum_array>0] = 1 - tmp_data.as_array()[tmp_sum_array>0]/tmp_sum_array[tmp_sum_array>0]        
                             
        if out is None: 
                          
            return tmp_out
        
        else:    
            
            out.fill(tmp_out)
            
#            # TODO not working with ind
#            x.add(self.background_term, out = out)            
#            self.data.divide(out, out=out)
#            out.as_array()[np.isinf(out.as_array())]=0
#            out.subtract(1, out=out)
#            out *= -1
            
            
    def convex_conjugate(self, x):
        
        '''Convex conjugate of KullbackLeibler at x'''
        
        if self.eta is None:
            tmp_part = 0
        else:            
            tmp_part = self.eta.dot(x)
        xlogy = - scipy.special.xlogy(self.b.as_array(), 1 - x.as_array())         
        return numpy.sum(xlogy) - tmp_part
            
    def proximal(self, x, tau, out=None):
        
        r'''Proximal operator of KullbackLeibler at x
           
           .. math::     prox_{\tau * f}(x)

        '''

        if self.eta is None:
            tmp_part = 0
        else:
            self.eta = tmp_part            

        if out is None:        
            return 0.5 *( (x - tmp_part - tau) + ( (x + tmp_part - tau)**2 + 4*tau*self.b   ) .sqrt() )
        else:
            
            tmp =  0.5 *( (x - tmp_part - tau) + 
                        ( (x + tmp_part - tau)**2 + 4*tau*self.b  ) .sqrt()
                        )
            x.add(tmp_part, out=out)
            out -= tau
            out *= out
            tmp = self.b * (4 * tau)
            out.add(tmp, out=out)
            out.sqrt(out=out)
            
            x.subtract(tmp_part, out=tmp)
            tmp -= tau
            
            out += tmp
            
            out *= 0.5
                            
#    def proximal_conjugate(self, x, tau, out=None):
#        
#        r'''Proximal operator of the convex conjugate of KullbackLeibler at x:
#           
#           .. math::     prox_{\tau * f^{*}}(x)
#        '''
#
#                
#        if out is None:
#            z = x + tau * self.background_term
#            return 0.5*((z + 1) - ((z-1)**2 + 4 * tau * self.data).sqrt())
#        else:
#            
#            tmp = tau * self.background_term
#            tmp += x
#            tmp -= 1
#            
#            self.data.multiply(4*tau, out=out)    
#            
#            out.add((tmp)**2, out=out)
#            out.sqrt(out=out)
#            out *= -1
#            tmp += 2
#            out += tmp
#            out *= 0.5
#
#    def __rmul__(self, scalar):
#        
#        '''Multiplication of KullbackLeibler with a scalar        
#            
#            Returns: ScaledFunction
#        '''
#        
#        return ScaledFunction(self, scalar) 


if __name__ == '__main__':
    
    from ccpi.framework import ImageGeometry
    import numpy as np
    
    M, N =  2,3
    ig = ImageGeometry(voxel_num_x=M, voxel_num_y = N)
    
    # if both u1, g1 > 0
    u1 = ig.allocate('random_int', seed = 5)
    g1 = ig.allocate('random_int', seed = 10)
    b1 = ig.allocate('random_int', seed = 100)
    
    f = KullbackLeibler(b=g1)   
    res = f(g1)     
    numpy.testing.assert_equal(0.0, f(g1)) 
    
    background_term = b1
    background_term.as_array()[0,0] = -10000
    background_term.as_array()[0,1] = - u1.as_array()[0,1]    
    
    f1 = KullbackLeibler(b=g1, eta = background_term)   
    res_gradient = f1.gradient(u1)
#    print(res_gradient.as_array())
#    
    res_gradient_out = u1.geometry.allocate()
    f1.gradient(u1, out = res_gradient_out)
#    print(res_gradient_out.as_array())
#    
#    u1.add(background_term, out = div)
#    g1.divide(div, out=div)
##    div.subtract(1, out=div)
##    div *= -1
##    div.as_array()[numpy.isinf(div.as_array())] = 0
#
##    
##    tmp_sum = u1 + background_term
##    tmp_sum_array = tmp_sum.as_array()
#
##    
###    div.as_array()[tmp_sum_array>0] = g1.as_array()[tmp_sum_array>0]/tmp_sum_array[tmp_sum_array>0]
###    
###        
##    g1.divide(tmp_sum, out=g1)
##    g1.add(1)
##    print(g1.as_array())
#    
#    
##    np.copyto(div.as_array(),g1.as_array()[tmp_sum_array>0]/tmp_sum_array[tmp_sum_array>0])
#    
#    
##    div.as_array()[tmp_sum_array>0].fill((g1.as_array()[tmp_sum_array>0]/tmp_sum_array[tmp_sum_array>0]))
#            
##            )
#    
##    np.copyto(tmp.array[k], t.array)
#    
##    print(div.as_array())
##    
#    
##    ind = tmp_sum>0 
##    
##    out_grad = u1.geometry.allocate()
##    
##    np.copyto(out_grad.as_array()[ind], (1 - (g1.as_array()/tmp_sum))[ind])
##    
#    
##        ind = tmp_sum>0 
##        
##        if out is None: 
##            
##            self.out_grad.fill(1 - self.data.as_array()[ind]/tmp_sum[ind])
##    
#    
#    
##    f1 = KullbackLeibler(g1, background_term = u1)        
##    numpy.testing.assert_equal(0.0, f1(g1))     
#    
#    
#    
##    print(f(g1))
##    print(f(u1))
#    
##    g2 = g1.clone()
##    g2.as_array()[0,1] = 0
###    print(f(g2))
##
##
##    tmp = scipy.special.kl_div(g1.as_array(), g2.as_array())  
##    
##    
##    res_grad = f.gradient()
#    
#        
#
#    
#        