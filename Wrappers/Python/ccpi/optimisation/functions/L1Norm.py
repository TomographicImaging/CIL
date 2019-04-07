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
"""
Created on Wed Mar  6 19:42:34 2019

@author: evangelos
"""

#import numpy as np
##from ccpi.optimisation.funcs import Function
#from ccpi.optimisation.functions import Function
#from ccpi.framework import DataContainer, ImageData 
#
#
#############################   L1NORM FUNCTIONS   #############################
#class SimpleL1Norm(Function):
#    
#    def __init__(self, alpha=1):
#        
#        super(SimpleL1Norm, self).__init__()         
#        self.alpha = alpha
#        
#    def __call__(self, x):
#        return self.alpha * x.abs().sum()
#    
#    def gradient(self,x):
#        return ValueError('Not Differentiable')
#            
#    def convex_conjugate(self,x):
#        return 0
#    
#    def proximal(self, x, tau):
#        ''' Soft Threshold'''
#        return x.sign() * (x.abs() - tau * self.alpha).maximum(0)
#        
#    def proximal_conjugate(self, x, tau):
#        return x.divide((x.abs()/self.alpha).maximum(1.0))
    
#class L1Norm(SimpleL1Norm):
#    
#    def __init__(self, alpha=1, **kwargs):
#        
#        super(L1Norm, self).__init__()         
#        self.alpha = alpha 
#        self.b = kwargs.get('b',None)
#        
#    def __call__(self, x):
#        
#        if self.b is None:
#            return SimpleL1Norm.__call__(self, x)
#        else:
#            return SimpleL1Norm.__call__(self, x - self.b)
#            
#    def gradient(self, x):
#        return ValueError('Not Differentiable')
#            
#    def convex_conjugate(self,x):
#        if self.b is None:
#            return SimpleL1Norm.convex_conjugate(self, x)
#        else:
#            return SimpleL1Norm.convex_conjugate(self, x) + (self.b * x).sum()
#    
#    def proximal(self, x, tau):
#        
#        if self.b is None:
#            return SimpleL1Norm.proximal(self, x, tau)
#        else:
#            return self.b + SimpleL1Norm.proximal(self, x - self.b , tau)
#        
#    def proximal_conjugate(self, x, tau):
#        
#        if self.b is None:
#            return SimpleL1Norm.proximal_conjugate(self, x, tau)
#        else:
#            return SimpleL1Norm.proximal_conjugate(self, x - tau*self.b, tau)
#        

###############################################################################                
from ccpi.optimisation.functions import Function
from ccpi.optimisation.functions.ScaledFunction import ScaledFunction        
from ccpi.optimisation.operators import ShrinkageOperator 
 

class L1Norm(Function):
    
    def __init__(self, **kwargs):
        
        super(L1Norm, self).__init__()
        self.b = kwargs.get('b',None) 
        
    def __call__(self, x):
        
        y = x
        if self.b is not None: 
            y = x - self.b
        return y.abs().sum()  
    
    def gradient(self,x):
        #TODO implement subgradient???
        return ValueError('Not Differentiable')   
    
    def convex_conjugate(self,x):
        #TODO implement Indicator infty???

        y = 0        
        if self.b is not None:
            y =  0 + (self.b * x).sum()
        return y  
    
    def proximal(self, x, tau, out=None):
        
        # TODO implement shrinkage operator, we will need it later e.g SplitBregman
        
        if out is None:
            if self.b is not None:
                return self.b + ShrinkageOperator.__call__(self, x - self.b, tau)
            else:
                return ShrinkageOperator.__call__(self, x, tau)             
        else:
            if self.b is not None:
                out.fill(self.b + ShrinkageOperator.__call__(self, x - self.b, tau))
            else:
                out.fill(ShrinkageOperator.__call__(self, x, tau))
                                    
    def proximal_conjugate(self, x, tau, out=None):
        
        if out is None:
            if self.b is not None:
                return (x - tau*self.b).divide((x - tau*self.b).abs().maximum(1.0))
            else:
                return x.divide(x.abs().maximum(1.0))
        else:
            if self.b is not None:
                out.fill((x - tau*self.b).divide((x - tau*self.b).abs().maximum(1.0)))
            else:
                out.fill(x.divide(x.abs().maximum(1.0)) )                
            
    def __rmul__(self, scalar):
        return ScaledFunction(self, scalar)             
                


if __name__ == '__main__':   
    
    from ccpi.framework import ImageGeometry
    import numpy
    N, M = 40,40
    ig = ImageGeometry(N, M)
    scalar = 10
    b = ig.allocate('random_int')
    u = ig.allocate('random_int') 
    
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
    
    
    



    
            
        

        
        
        
      