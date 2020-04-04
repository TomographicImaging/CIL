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

from ccpi.optimisation.functions import Function
from ccpi.framework import DataContainer

class L2NormSquared(Function):
    
    r""" L2NormSquared function: :math:`F(x) = \| x\|^{2}_{2} = \underset{i}{\sum}x_{i}^{2}`
          
    Following cases are considered:
                
        a) :math:`F(x) = \|x\|^{2}_{2}`
        b) :math:`F(x) = \|x - b\|^{2}_{2}`
        
    .. note::  For case b) case we can use :code:`F = L2NormSquared().centered_at(b)`,
               see *TranslateFunction*.
        
    :Example:
        
        >>> F = L2NormSquared()
        >>> F = L2NormSquared(b=b) 
        >>> F = L2NormSquared().centered_at(b)
                                                          
    """    
    
    def __init__(self, **kwargs):
        '''creator

        Cases considered (with/without data):            
                a) .. math:: f(x) = \|x\|^{2}_{2} 
                b) .. math:: f(x) = \|\|x - b\|\|^{2}_{2}

        :param b:  translation of the function
        :type b: :code:`DataContainer`, optional
        '''                        
        super(L2NormSquared, self).__init__(L = 2)
        self.b = kwargs.get('b',None) 
        
        #if self.b is not None:
        #    self.domain = self.b.geometry  
                            
    def __call__(self, x):

        r"""Returns the value of the L2NormSquared function at x.
        
        Following cases are considered:
            
            a) :math:`F(x) = \|x\|^{2}_{2}`
            b) :math:`F(x) = \|x - b\|^{2}_{2}`
    
        :param: :math:`x`
        :returns: :math:`\underset{i}{\sum}x_{i}^{2}`
                
        """          
            
        y = x
        if self.b is not None: 
            y = x - self.b
        try:
            return y.squared_norm()
        except AttributeError as ae:
            # added for compatibility with SIRF 
            return (y.norm()**2)
                
    def gradient(self, x, out=None):        
        
        r"""Returns the value of the gradient of the L2NormSquared function at x.
        
        Following cases are considered:
                
            a) :math:`F'(x) = 2x`
            b) :math:`F'(x) = 2(x-b)`
                
        """
                
        if out is not None:
            
            out.fill(x)
            if self.b is not None:
                out -= self.b
            out *= 2
            
        else:
            
            y = x
            if self.b is not None:
                y = x - self.b
            return 2*y
        
                                                       
    def convex_conjugate(self, x):
        
        r"""Returns the value of the convex conjugate of the L2NormSquared function at x.
        
        Consider the following cases:
                
                a) .. math:: F^{*}(x^{*}) = \frac{1}{4}\|x^{*}\|^{2}_{2} 
                b) .. math:: F^{*}(x^{*}) = \frac{1}{4}\|x^{*}\|^{2}_{2} + <x^{*}, b>
                
        """                
        tmp = 0
        
        if self.b is not None:
            tmp = x.dot(self.b) 
            
        return (1./4.) * x.squared_norm() + tmp


    def proximal(self, x, tau, out = None):
        
        r"""Returns the value of the proximal operator of the L2NormSquared function at x.
        
        
        Consider the following cases:
                
                a) .. math:: \mathrm{prox}_{\tau F}(x) = \frac{x}{1+2\tau}
                b) .. math:: \mathrm{prox}_{\tau F}(x) = \frac{x-b}{1+2\tau} + b      
                        
        """            

        if out is None:
            
            if self.b is None:
                return x/(1+2*tau)
            else:
                tmp = x.subtract(self.b)
                tmp /= (1+2*tau)
                tmp += self.b
                return tmp

        else:
            if self.b is not None:
                x.subtract(self.b, out=out)
                out /= (1+2*tau)
                out += self.b
            else:
                x.divide((1+2*tau), out=out)

# TODO
# We do not need this, because it is computed directly by the method proximal_conjugate
# of parent class "Function"
# However, we should add the expression below to the docs    
                
#    def proximal_conjugate(self, x, tau, out=None):
#        
#        r'''Proximal operator of the convex conjugate of L2NormSquared at x:
#           
#           .. math::  prox_{\tau * f^{*}}(x)'''
#        
#        if out is None:
#            if self.b is not None:
#                return (x - tau*self.b)/(1 + tau/2) 
#            else:
#                return x/(1 + tau/2)
#        else:
#            if self.b is not None:
#                x.subtract(tau*self.b, out=out)
#                out.divide(1+tau/2, out=out)
#            else:
#                x.divide(1 + tau/2, out=out)                                        

# TODO
# it's easier if we define a weighted space
# otherwise we need to define for every function a weighted version
# if we need it. However, for that we need to define a space for the functions __init__
                
from ccpi.optimisation.operators import LinearOperatorMatrix
                
class WeightedL2NormSquared(Function):
    
   def __init__(self, **kwargs):
                         
    # Weight treated as Linear operator, 
    # in order to compute the lispchitz constant L = 2 *||weight||
    
    
    self.weight = kwargs.get('weight', 1.0) 
    self.b = kwargs.get('b', None) 
    tmp_norm = 1.0  
    
    # Need this to make it behave similarly as the L2NormSquared
    self.weight_sqrt = 1.0
          
    if isinstance(self.weight, DataContainer):
        op_weight = LinearOperatorMatrix(self.weight.as_array())  
        tmp_norm = op_weight.norm() 
        self.weight_sqrt = self.weight.sqrt()
        if (self.weight<0).any():
            raise ValueError('Weigth contains negative values')
    super(WeightedL2NormSquared, self).__init__(L = 2 * tmp_norm  )        
    
    
   def __call__(self, x):
        
        y = self.weight_sqrt * x
        if self.b is not None: 
            y = self.weight_sqrt * (x - self.b)
        try:
            return y.squared_norm()
        except AttributeError as ae:
            # added for compatibility with SIRF 
            return (y.norm()**2)        
                
   def gradient(self, x, out=None):        
        
                
        if out is not None:
            
            out.fill(x)        
            if self.b is not None:
                out -= self.b
            out *= self.weight                
            out *= 2
            
        else:
            
            y = x
            if self.b is not None:
                y = x - self.b
            return 2*self.weight*y
    
   def convex_conjugate(self, x):
                      
        tmp = 0
        
        if self.b is not None:
            tmp = x.dot(self.b) 
            
        return (1./4) * (x/self.weight_sqrt).squared_norm() + tmp
    
   def proximal(self, x, tau, out = None):
                  

        if out is None:
            
            if self.b is None:
                return x/(1+2*tau*self.weight)
            else:
                tmp = x.subtract(self.b)
                tmp /= (1+2*tau*self.weight)
                tmp += self.b
                return tmp

        else:
            if self.b is not None:
                x.subtract(self.b, out=out)
                out /= (1+2*tau*self.weight)
                out += self.b
            else:
                x.divide((1+2*tau*self.weight), out=out)
                
    #TODO add test for the proximal conjugate                
                                               
