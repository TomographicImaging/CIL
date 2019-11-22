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
    
    r""" Kullback Leibler divergence function is defined as:
            
    .. math:: F(u, v)
            = \begin{cases} 
            u \log(\frac{u}{v}) - u + v & \mbox{ if } u > 0, v > 0\\
            v & \mbox{ if } u = 0, v \ge 0 \\
            \infty, & \mbox{otherwise}
            \end{cases}  
       
    For data :math:`b` that follow Poisson distribution and a background term :math:`\eta`,
    we use the above definition, provided that :math:`v+\eta>0`.
    
    .. math:: F(b, v + \eta)
    
    This is related to PoissonLogLikelihoodWithLinearModelForMean definition that is used in PET reconstruction
    in the PET-MR software , see https://github.com/CCPPETMR and for more details in
    
    http://stir.sourceforge.net/documentation/doxy/html/classstir_1_1PoissonLogLikelihoodWithLinearModelForMean.html
                        
    """            
    
    
    def __init__(self,  **kwargs):
        
        super(KullbackLeibler, self).__init__(L = None)  
        
        self.b = kwargs.get('b', None)
        
        if self.b is None:
            raise ValueError('Please define data, as b = ...')
            
        if self.b.as_array().any()<0:
            raise ValueError('Data should be larger or equal to 0')              
         
        self.eta = kwargs.get('eta',self.b * 0.0)
        
                                                    
    def __call__(self, x):
        

        r"""Returns the value of the KullbackLeibler function at :math:`(b, x + \eta)`.
        """
                
        tmp_sum = (x + self.eta).as_array()
        ind = tmp_sum > 0
        tmp = scipy.special.kl_div(self.b.as_array()[ind], tmp_sum[ind])                
        return numpy.sum(tmp)         
        
    def log(self, datacontainer):
        '''calculates the in-place log of the datacontainer'''
        if not functools.reduce(lambda x,y: x and y>0, datacontainer.as_array().ravel(), True):
            raise ValueError('KullbackLeibler. Cannot calculate log of negative number')
        datacontainer.fill( numpy.log(datacontainer.as_array()) )

        
    def gradient(self, x, out=None):
        
        r"""Returns the value of the gradient of the KullbackLeibler function at :math:`(b, x + \eta)`.                
        
        .. math:: F'(b, x + \eta) = 1 - \frac{b}{x+\eta}
        
        """        
        
        tmp_sum_array = (x + self.eta).as_array()
        if out is None:   
            tmp_out = x.geometry.allocate() 
            tmp_out.as_array()[tmp_sum_array>0] = 1 - self.b.as_array()[tmp_sum_array>0]/tmp_sum_array[tmp_sum_array>0]         
            return tmp_out        
        else:                 
            x.add(self.eta, out=out)
            out.as_array()[tmp_sum_array>0] = 1 - self.b.as_array()[tmp_sum_array>0]/tmp_sum_array[tmp_sum_array>0]                                

            
    def convex_conjugate(self, x):
        
        r"""Returns the value of the convex conjugate of the KullbackLeibler function at :math:`(b, x + \eta)`.                
        
        .. math:: F^{*}(b, x + \eta) = - b \log(1-x^{*}) - <x^{*}, \eta> 
        
        """  
        
        xlogy = - scipy.special.xlogy(self.b.as_array(), 1 - x.as_array())         
        return numpy.sum(xlogy) - self.eta.dot(x)
            
    def proximal(self, x, tau, out=None):
        
        r"""Returns the value of the proximal operator of the KullbackLeibler function at :math:`(b, x + \eta)`.
        
        .. math:: \mathrm{prox}_{\tau F}(x) = \frac{1}{2}\bigg( (x - \eta - \tau) + \sqrt{ (x + \eta - \tau)^2 + 4\tau b} \bigg)
        
        The proximal for the convex conjugate of :math:`F` is 
        
        .. math:: \mathrm{prox}_{\tau F^{*}}(x) = 0.5*((z + 1) - \sqrt{(z-1)^2 + 4 * \tau b})
        
        where :math:`z = x + \tau * \eta`
                    
        """
          
        if out is None:        
            return 0.5 *( (x - self.eta - tau) + ( (x + self.eta - tau)**2 + 4*tau*self.b   ) .sqrt() )        
        else:                      
            x.add(self.eta, out=out)
            out -= tau
            out *= out
            out.add(self.b * (4 * tau), out=out)
            out.sqrt(out=out)  
            out.subtract(tau, out = out)
            out.add(x, out=out)         
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
    
    M, N, K =  20, 30, 40
    ig = ImageGeometry(N, M, K)
    
    u1 = ig.allocate('random_int', seed = 5)    
    g1 = ig.allocate('random_int', seed = 10)
    b1 = ig.allocate('random_int', seed = 100)
    
    # with no data
    try:
        f = KullbackLeibler()   
    except ValueError:
        print('Give data b=...\n')
        
    print('With data, no background\n')        
    f = KullbackLeibler(b=g1)
    
    print('Check KullbackLeibler(x,x)=0\n') 
    numpy.testing.assert_equal(0.0, f(g1))
    
    print('Check gradient\n')
    res_gradient = f.gradient(u1)
    res_gradient_out = u1.geometry.allocate()
    f.gradient(u1, out = res_gradient_out) 
    numpy.testing.assert_array_almost_equal(res_gradient.as_array(), \
                                            res_gradient_out.as_array(),decimal = 4)  
    
    print('Check proximal\n')    
    
    tau = 0.4
    res_proximal = f.proximal(u1, tau)
    res_proximal_out = u1.geometry.allocate()   
    f.proximal(u1, tau, out = res_proximal_out)
    numpy.testing.assert_array_almost_equal(res_proximal.as_array(), \
                                            res_proximal_out.as_array(), decimal =5)        
    
    

               
                
    
    
#    
#    res = f(g1)     
#    numpy.testing.assert_equal(0.0, f(g1)) 
#    
##    eta = b1
##    eta.as_array()[0,0] = -10000
##    eta.as_array()[0,1] = - u1.as_array()[0,1]    
#    
#    res_gradient = f.gradient(u1)
#    
#    res_gradient_out = u1.geometry.allocate()
#    f.gradient(u1, out = res_gradient_out)
#    
#    numpy.testing.assert_array_almost_equal(res_gradient.as_array(), \
#                                            res_gradient_out.as_array(),decimal = 4)  
#
#    tau = 0.4
#    res_proximal = f.proximal(u1, tau)
#    res_proximal_out = u1.geometry.allocate()   
#    f.proximal(u1, tau, out = res_proximal_out)
#    numpy.testing.assert_array_almost_equal(res_proximal.as_array(), \
#                                            res_proximal_out.as_array(), decimal =5)      
#    
#    
#    
#    
#    
##    print(res_gradient_out.as_array())
##    
##    u1.add(background_term, out = div)
##    g1.divide(div, out=div)
###    div.subtract(1, out=div)
###    div *= -1
###    div.as_array()[numpy.isinf(div.as_array())] = 0
##
###    
###    tmp_sum = u1 + background_term
###    tmp_sum_array = tmp_sum.as_array()
##
###    
####    div.as_array()[tmp_sum_array>0] = g1.as_array()[tmp_sum_array>0]/tmp_sum_array[tmp_sum_array>0]
####    
####        
###    g1.divide(tmp_sum, out=g1)
###    g1.add(1)
###    print(g1.as_array())
##    
##    
###    np.copyto(div.as_array(),g1.as_array()[tmp_sum_array>0]/tmp_sum_array[tmp_sum_array>0])
##    
##    
###    div.as_array()[tmp_sum_array>0].fill((g1.as_array()[tmp_sum_array>0]/tmp_sum_array[tmp_sum_array>0]))
##            
###            )
##    
###    np.copyto(tmp.array[k], t.array)
##    
###    print(div.as_array())
###    
##    
###    ind = tmp_sum>0 
###    
###    out_grad = u1.geometry.allocate()
###    
###    np.copyto(out_grad.as_array()[ind], (1 - (g1.as_array()/tmp_sum))[ind])
###    
##    
###        ind = tmp_sum>0 
###        
###        if out is None: 
###            
###            self.out_grad.fill(1 - self.data.as_array()[ind]/tmp_sum[ind])
###    
##    
##    
###    f1 = KullbackLeibler(g1, background_term = u1)        
###    numpy.testing.assert_equal(0.0, f1(g1))     
##    
##    
##    
###    print(f(g1))
###    print(f(u1))
##    
###    g2 = g1.clone()
###    g2.as_array()[0,1] = 0
####    print(f(g2))
###
###
###    tmp = scipy.special.kl_div(g1.as_array(), g2.as_array())  
###    
###    
###    res_grad = f.gradient()
##    
##        
##
##    
##        