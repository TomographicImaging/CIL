#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 09:55:36 2019

@author: evangelos
"""

import numpy as np
#from ccpi.optimisation.funcs import Function
from ccpi.optimisation.functions import Function
from ccpi.optimisation.functions import ScaledFunction


class FunctionOperatorComposition(Function):
    
    def __init__(self, operator, function):
        super(FunctionOperatorComposition, self).__init__()
        self.function = function     
        self.operator = operator
        alpha = 1
        if isinstance (function, ScaledFunction):
            alpha = function.scalar
        self.L = 2 * alpha * operator.norm()**2
        
        
    def __call__(self, x):
    
        return self.function(self.operator.direct(x))   

    def call_adjoint(self, x):
    
        return self.function(self.operator.adjoint(x))  

    def convex_conjugate(self, x):
        
        ''' convex_conjugate does not take into account the Operator'''
        return self.function.convex_conjugate(x)

    def proximal(self, x, tau):
        
        ''' proximal does not take into account the Operator'''
        
        return self.function.proximal(x, tau, out=None)

    def proximal_conjugate(self, x, tau, out=None):    

        ''' proximal conjugate does not take into account the Operator'''
        
        return self.function.proximal_conjugate(x, tau) 

    def gradient(self, x, out=None):
        
        ''' Gradient takes into account the Operator'''
        if out is None:
            return self.operator.adjoint(
                self.function.gradient(self.operator.direct(x))
                )
        else:
            self.operator.adjoint(
                self.function.gradient(self.operator.direct(x), 
                out=out)
            )
        
                       