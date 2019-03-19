# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 13:10:56 2019

@author: evangelos
"""

import numpy as np
from ccpi.optimisation.functions import Function
from ccpi.framework import DataContainer, ImageData, ImageGeometry 
from ccpi.optimisation.functions import ScaledFunction

    
class SimpleL2NormSq(Function):
    
    def __init__(self, alpha=1):
        self.alpha = alpha
        super(SimpleL2NormSq, self).__init__()         
        # Lispchitz constant of gradient
        self.L = 2
        
    def __call__(self, x):
        return self.alpha * x.power(2).sum()
    
    def gradient(self,x, out=None):
        if out is None:
            return 2 * x
        else:
            out.fill(2*x)
    
    def convex_conjugate(self,x):
        return (1/4) * x.squared_norm()
        
    def proximal(self, x, tau, out=None):
        if out is None:
            return x.divide(1+2*tau)
        else:
            x.divide(1+2*tau, out=out)
    
    def proximal_conjugate(self, x, tau, out=None):
        if out is None:
            return x.divide(1 + tau/2)    
        else:
            x.divide(1+tau/2, out=out)



############################   L2NORM FUNCTIONS   #############################
class L2NormSq(SimpleL2NormSq):
    
    def __init__(self, **kwargs):
        super(L2NormSq, self).__init__()
        self.b = kwargs.get('b',None)              

    def __call__(self, x):
        if self.b is None:
            return SimpleL2NormSq.__call__(self, x)
        else:
            return SimpleL2NormSq.__call__(self, x - self.b) 
        
    def gradient(self, x):
        if self.b is None:
            return 2 * x 
        else:
            return 2 * (x - self.b) 
                                                       
    def convex_conjugate(self, x):
        ''' The convex conjugate corresponds to the simple functional i.e., 
        f(x) = alpha * ||x - b||_{2}^{2}
        '''
        if self.b is None:
            return SimpleL2NormSq.convex_conjugate(self, x)
        else:
            return SimpleL2NormSq.convex_conjugate(self, x) + (self.b * x).sum()

    def proximal(self, x, tau):

        ''' The proximal operator corresponds to the simple functional i.e., 
        f(x) = alpha * ||x - b||_{2}^{2}
        
        argmin_x { 0.5||x - u||^{2} + tau f(x) }
        '''        
        if self.b is None:
            return SimpleL2NormSq.proximal(self, x, tau)
        else:
            return self.b + SimpleL2NormSq.proximal(self, x - self.b , tau)
    
    def proximal_conjugate(self, x, tau):
        ''' The proximal operator corresponds to the simple convex conjugate 
        functional i.e., f^{*}(x^{)        
        argmin_x { 0.5||x - u||^{2} + tau f(x) }
        '''
        if self.b is None:
            return SimpleL2NormSq.proximal_conjugate(self, x, tau)
        else:
            return SimpleL2NormSq.proximal_conjugate(self, x - tau * self.b, tau)
        
    def __rmul__(self, scalar):
        return ScaledFunction(self, scalar)        
