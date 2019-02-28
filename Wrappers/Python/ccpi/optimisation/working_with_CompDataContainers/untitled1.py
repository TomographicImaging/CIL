#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 21:16:56 2019

@author: evangelos
"""

import numpy as np
from ccpi.optimisation.funcs import Function
from ccpi.framework import DataContainer, ImageData, ImageGeometry 
from operators import CompositeDataContainer
from numbers import Number
from operators import Identity

#%%

class Function(object):
    def __init__(self):
        pass
    def __call__(self,x, out=None):           raise NotImplementedError 
    def gradient(self, x):                    raise NotImplementedError
    def proximal(self, x, tau):               raise NotImplementedError
    def proximal_conjugate(self, x, out=None):      raise NotImplementedError
    def proximal(self, x, tau, out=None): raise NotImplementedError
    def convex_conjugate(self, x, tau, out=None): raise NotImplementedError


class SimpleL2NormSq(Function):
    
    def __init__(self, alpha=1):
        
        super(SimpleL2NormSq, self).__init__()         
        self.alpha = alpha
        
    def __call__(self, x):
        return self.alpha * x.power(2).sum()
    
    def gradient(self,x):
        return 2 * self.alpha * x
    
    def convex_conjugate(self,x):
        return (1/4*self.alpha) * x.power(2).sum()
    
    def proximal(self, x, tau):
        return x.divide(1+2*tau*self.alpha)
    
    def proximal_conjugate(self, x, tau):
        return x.divide(1 + tau/2*self.alpha )
    
        
class L2NormSq(SimpleL2NormSq):
##    
    def __init__(self, A, b = None, alpha=1, **kwargs):
        
        super(L2NormSq, self).__init__()         
        self.alpha = alpha        
        self.A = A
        self.b = b
                
    def __call__(self, x):
        
        if self.b is None:
            return SimpleL2NormSq.__call__(self, self.A.direct(x))
        else:
            return SimpleL2NormSq.__call__(self, self.A.direct(x) - self.b)
        
    def convex_conjugate(self, x):
        
        ''' The convex conjugate corresponds to the simple functional i.e., 
        f(x) = alpha * ||x - b||_{2}^{2}
        '''
        
        if self.b is None:
            return SimpleL2NormSq.convex_conjugate(self, x)
        else:
            return SimpleL2NormSq.convex_conjugate(self, x) + (self.b * x).sum()
                            
    def gradient(self, x):
        
        if self.b is None:
            return 2*self.alpha * self.A.adjoint(self.A.direct(x)) 
        else:
            return 2*self.alpha * self.A.adjoint(self.A.direct(x) - self.b) 
        
    def proximal(self, x, tau):
        
        ''' The proximal operator corresponds to the simple functional i.e., 
        f(x) = alpha * ||x - b||_{2}^{2}
        
        argmin_x { 0.5||x - u||^{2} + tau f(x) }
        '''
        
        if self.b is None:
            return SimpleL2NormSq.proximal(self, x, tau)#x.divide(1+tau)
        else:
            return SimpleL2NormSq.proximal(self, x + 2*tau*self.alpha*self.b, tau) 
        #(x + 2*tau*self.alpha*self.b).divide(1+2*tau*self.alpha)
        
    def proximal_conjugate(self, x, tau):
        
        ''' The proximal operator corresponds to the simple convex conjugate 
        functional i.e., f^{*}(x^{)        
        argmin_x { 0.5||x - u||^{2} + tau f(x) }
        '''
        if self.b is None:
            return SimpleL2NormSq.proximal_conjugate(self, x, tau)
        else:
            return SimpleL2NormSq.proximal_conjugate(self, x - tau * self.b, tau)
              


#x = DataContainer(np.random.randint(10, size=(2,3)))
#f = 

#f = SimpleL2NormSq()
#fx = f(x)
#
#g = OperatorL2NormSq(Identity((2,3)))
#gx = g(x)
#
#print(fx, gx)
#
#grad_fx = f.gradient(x)
#grad_gx = g.gradient(x)
#print(grad_fx.as_array(), grad_gx.as_array())
#
#A = 1
#b = 1
#
#phi = L2NormSq(1)