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
from ccpi.optimisation.operators import DiagonalOperator

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
                

                
class WeightedL2NormSquared(Function):
    
    r""" WeightedL2NormSquared function: :math:`F(x) = \| x\|_{w}^{2}_{2} = \underset{i}{\sum}w_{i}*x_{i}^{2} = <x, w*x> = x^{T}*w*x`
                                                                    
    """                 
              
    def __init__(self, **kwargs):
        
       # Weight can be either a scalar or a DataContainer
       # Lispchitz constant L = 2 *||weight||         
                   
       self.weight = kwargs.get('weight', 1.0) 
       self.b = kwargs.get('b', None) 
       tmp_norm = 1.0  
       self.tmp_space = self.weight*0. 
                            
       if isinstance(self.weight, DataContainer):
           self.operator_weight = DiagonalOperator(self.weight) 
           tmp_norm = self.operator_weight.norm() 
           self.tmp_space = self.operator_weight.domain_geometry().allocate()  
           
           if (self.weight<0).any():
               raise ValueError('Weigth contains negative values')             

       super(WeightedL2NormSquared, self).__init__(L = 2 * tmp_norm  ) 
    
    def __call__(self, x):
       
       self.operator_weight.direct(x, out = self.tmp_space)
       y = x.dot(self.tmp_space)
       
       if self.b is not None:
           self.operator_weight.direct(x - self.b, out = self.tmp_space)
           y = (x - self.b).dot(self.tmp_space)
       return y
     
                
    def gradient(self, x, out=None):        
        
                
        if out is not None:
            
            out.fill(x)        
            if self.b is not None:
                out -= self.b
            self.operator_weight.direct(out, out=out)                            
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
            
        return (1./4) * (x/self.weight.sqrt()).squared_norm() + tmp
    
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
                

if __name__ == '__main__':
    
    print("Checks for WeightedL2NormSquared")
    
    from ccpi.framework import ImageGeometry
    from ccpi.optimisation.functions import TranslateFunction
    import numpy
    
    # Tests for weighted L2NormSquared
    ig = ImageGeometry(voxel_num_x = 3, voxel_num_y = 3)
    weight = ig.allocate('random')
    
    f = WeightedL2NormSquared(weight=weight)                                              
    x = ig.allocate(0.4)
    
    res1 = f(x)
    res2 = (weight * (x**2)).sum()
    numpy.testing.assert_almost_equal(res1, res2, decimal=4)
    
    print("Call of WeightedL2NormSquared is ... ok")
    
    # gradient for weighted L2NormSquared    
    res1 = f.gradient(x)
    res2 = 2*weight*x
    out = ig.allocate()
    f.gradient(x, out = out)
    numpy.testing.assert_array_almost_equal(res1.as_array(), \
                                            out.as_array(), decimal=4)  
    numpy.testing.assert_array_almost_equal(res2.as_array(), \
                                            out.as_array(), decimal=4)  
    
    print("Gradient of WeightedL2NormSquared is ... ok")    
    
    # convex conjugate for weighted L2NormSquared       
    res1 = f.convex_conjugate(x)
    res2 = 1/4 * (x/weight.sqrt()).squared_norm()
    numpy.testing.assert_array_almost_equal(res1, \
                                            res2, decimal=4)   
    
    print("Convex conjugate of WeightedL2NormSquared is ... ok")        
    
    # proximal for weighted L2NormSquared       
    tau = 0.3
    out = ig.allocate()
    res1 = f.proximal(x, tau)
    f.proximal(x, tau, out = out)
    res2 = x/(1+2*tau*weight)
    numpy.testing.assert_array_almost_equal(res1.as_array(), \
                                            res2.as_array(), decimal=4)  
    
    print("Proximal of WeightedL2NormSquared is ... ok")  
    
    
    tau = 0.3
    out = ig.allocate()
    res1 = f.proximal_conjugate(x, tau)   
    res2 = x/(1 + tau/(2*weight))    
    numpy.testing.assert_array_almost_equal(res1.as_array(), \
                                            res2.as_array(), decimal=4)  
    
    print("Proximal conjugate of WeightedL2NormSquared is ... ok")  


    b = ig.allocate('random')
    f1 = TranslateFunction(WeightedL2NormSquared(weight=weight), b) 
    f2 = WeightedL2NormSquared(weight = weight, b=b)
    res1 = f1(x)
    res2 = f2(x)
    numpy.testing.assert_almost_equal(res1, res2, decimal=4)
    
    print("Call of WeightedL2NormSquared vs TranslateFunction is ... ok") 
    
    f1 = WeightedL2NormSquared(b=b)
    f2 = L2NormSquared(b=b)
    
    numpy.testing.assert_almost_equal(f1.L, f2.L, decimal=4)
    numpy.testing.assert_almost_equal(f1.L, 2, decimal=4)
    numpy.testing.assert_almost_equal(f2.L, 2, decimal=4)
    
    print("Check Lip constants ... ok")     
    
    
    
    
    
    
        

     
    
        
    
    