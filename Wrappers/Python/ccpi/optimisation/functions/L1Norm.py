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
from ccpi.optimisation.operators import ShrinkageOperator 
import numpy as np
 

class L1Norm(Function):
    
    r"""L1Norm function
            
        Consider the following cases:           
            a) .. math:: F(x) = ||x||_{1}
            b) .. math:: F(x) = ||x - b||_{1}
                                
    """   
           
    def __init__(self, **kwargs):
        '''creator

        Cases considered (with/without data):            
        a) :math:`f(x) = ||x||_{1}`
        b) :math:`f(x) = ||x - b||_{1}`

        :param b: translation of the function
        :type b: :code:`DataContainer`, optional
        '''
        super(L1Norm, self).__init__()
        self.b = kwargs.get('b',None)
        self.shinkage_operator = ShrinkageOperator()
        
    def __call__(self, x):
        
        r"""Returns the value of the L1Norm function at x.
        
        Consider the following cases:           
            a) .. math:: F(x) = ||x||_{1}
            b) .. math:: F(x) = ||x - b||_{1}        
        
        """
        
        y = x
        if self.b is not None: 
            y = x - self.b
        return y.abs().sum()  
          
    def convex_conjugate(self,x):
        
        r"""Returns the value of the convex conjugate of the L1Norm function at x.
        Here, we need to use the convex conjugate of L1Norm, which is the Indicator of the unit 
        :math:`L^{\infty}` norm
        
        Consider the following cases:
                
                a) .. math:: F^{*}(x^{*}) = \mathbb{I}_{\{\|\cdot\|_{\infty}\leq1\}}(x^{*}) 
                b) .. math:: F^{*}(x^{*}) = \mathbb{I}_{\{\|\cdot\|_{\infty}\leq1\}}(x^{*}) + <x^{*},b>      
        
    
        .. math:: \mathbb{I}_{\{\|\cdot\|_{\infty}\leq1\}}(x^{*}) 
            = \begin{cases} 
            0, \mbox{if } \|x^{*}\|_{\infty}\leq1\\
            \infty, \mbox{otherwise}
            \end{cases}
    
        """        
        tmp = (np.abs(x.as_array()).max() - 1)
        if tmp<=1e-5:            
            if self.b is not None:
                return self.b.dot(x)
            else:
                return 0.
        return np.inf        
                    
    def proximal(self, x, tau, out=None):
        
        r"""Returns the value of the proximal operator of the L1Norm function at x.
        
        
        Consider the following cases:
                
                a) .. math:: \mathrm{prox}_{\tau F}(x) = \mathrm{ShinkOperator}(x)
                b) .. math:: \mathrm{prox}_{\tau F}(x) = \mathrm{ShinkOperator}(x) + b   
    
        where,
        
        .. math :: \mathrm{prox}_{\tau F}(x) = \mathrm{ShinkOperator}(x) = sgn(x) * \max\{ |x| - \tau, 0 \}
                            
        """  
            
        if out is None:
            if self.b is not None:
                return self.b + self.shinkage_operator(x - self.b, tau)
            else:
                return self.shinkage_operator(x, tau)             
        else:
            if self.b is not None:
                out.fill(self.b + self.shinkage_operator(x - self.b, tau))
            else:
                out.fill(self.shinkage_operator(x, tau))
                                    
#    def proximal_conjugate(self, x, tau, out=None):
#        
#        r'''Proximal operator of the convex conjugate of L1Norm at x:
#                
#            .. math:: prox_{\tau * f^{*}}(x)
#                
#        '''          
#        
#        if out is None:
#            if self.b is not None:
#                return (x - tau*self.b).divide((x - tau*self.b).abs().maximum(1.0))
#            else:
#                return x.divide(x.abs().maximum(1.0))
#        else:
#            if self.b is not None:
#                out.fill((x - tau*self.b).divide((x - tau*self.b).abs().maximum(1.0)))
#            else:
#                out.fill(x.divide(x.abs().maximum(1.0)) )                
            
#    def __rmul__(self, scalar):
#        
#        '''Multiplication of L2NormSquared with a scalar        
#            
#            Returns: ScaledFunction
#        '''
#        
#        return ScaledFunction(self, scalar)


if __name__ == '__main__':   
    
    from ccpi.framework import ImageGeometry
    import numpy
    N, M = 400,400
    ig = ImageGeometry(N, M)
    scalar = 10
    b = ig.allocate('random')
    u = ig.allocate('random') 
    
    f = L1Norm()
    f_scaled = scalar * L1Norm()

    f_b = L1Norm(b=b)
    f_scaled_b = scalar * L1Norm(b=b)
    
    # call  
        
    a1 = f(u)
    a2 = f_scaled(u)
    numpy.testing.assert_equal(scalar * a1, a2)
    
    a3 = f_b(u)
    a4 = f_scaled_b(u)
    numpy.testing.assert_equal(scalar * a3, a4) 
    
    # proximal
    tau = 0.4
    b1 = f.proximal(u, tau*scalar)
    b2 = f_scaled.proximal(u, tau)
        
    numpy.testing.assert_array_almost_equal(b1.as_array(), b2.as_array(), decimal=4)
    
    b3 = f_b.proximal(u, tau*scalar)
    b4 = f_scaled_b.proximal(u, tau)
    
    z1 = b + (u-b).sign() * ((u-b).abs() - tau * scalar).maximum(0)
        
    numpy.testing.assert_array_almost_equal(b3.as_array(), b4.as_array(), decimal=4)    
#        
#    #proximal conjugate
#    
    c1 = f_scaled.proximal_conjugate(u, tau)
    c2 = u.divide((u.abs()/scalar).maximum(1.0))
    
    numpy.testing.assert_array_almost_equal(c1.as_array(), c2.as_array(), decimal=4) 
    
    c3 = f_scaled_b.proximal_conjugate(u, tau)
    c4 = (u - tau*b).divide( ((u-tau*b).abs()/scalar).maximum(1.0) )
    
    numpy.testing.assert_array_almost_equal(c3.as_array(), c4.as_array(), decimal=4)     
    
    
    



    
            
        

        
        
        
      
