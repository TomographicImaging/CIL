#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 19:42:34 2019

@author: evangelos
"""

import numpy as np
from ccpi.optimisation.funcs import Function
from ccpi.framework import DataContainer, ImageData, ImageGeometry 

#from operators import CompositeDataContainer, Identity, CompositeOperator
#from numbers import Number
#from GradientOperator import Gradient

############################   L1NORM FUNCTIONS   #############################
class SimpleL1Norm(Function):
    
    def __init__(self, alpha=1):
        
        super(SimpleL1Norm, self).__init__()         
        self.alpha = alpha
        
    def __call__(self, x):
        return self.alpha * x.abs().sum()
    
    def gradient(self,x):
        return ValueError('Not Differentiable')
            
    def convex_conjugate(self,x):
        return 0
    
    def proximal(self, x, tau):
        ''' Soft Threshold'''
        return x.sign() * (x.abs() - tau * self.alpha).maximum(1.0)
        
    def proximal_conjugate(self, x, tau):
        return x.divide((x.abs()/self.alpha).maximum(1.0))
    
class L1Norm(SimpleL1Norm):
    
    def __init__(self, alpha=1, **kwargs):
        
        super(L1Norm, self).__init__()         
        self.alpha = alpha 
        
        self.A = kwargs.get('A',None)
        self.b = kwargs.get('b',None)
        
    def __call__(self, x):
        
        if self.b is None:
            return SimpleL1Norm.__call__(self, self.A.direct(x))
        else:
            return SimpleL1Norm.__call__(self, self.A.direct(x) - self.b)
        
    def eval_norm(self, x):
        
        return SimpleL1Norm.__call__(self, x)        
    
    def gradient(self, x):
        return ValueError('Not Differentiable')
            
    def convex_conjugate(self,x):
        if self.b is None:
            return SimpleL1Norm.convex_conjugate(self, x)
        else:
            return SimpleL1Norm.convex_conjugate(self, x) + (self.b * x).sum()
    
    def proximal(self, x, tau):
        
        if self.b is None:
            return SimpleL1Norm.proximal(self, x, tau)
        else:
            return self.b + SimpleL1Norm.proximal(self, x + self.b , tau)
        
    def proximal_conjugate(self, x, tau):
        
        if self.b is None:
            return SimpleL1Norm.proximal_conjugate(self, x, tau)
        else:
            return SimpleL1Norm.proximal_conjugate(self, x - tau*self.b, tau)
                        