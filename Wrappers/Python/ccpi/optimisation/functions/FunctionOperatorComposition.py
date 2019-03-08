#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 09:55:36 2019

@author: evangelos
"""

import numpy as np
from ccpi.optimisation.funcs import Function


class FunctionOperatorComposition(Function):
    
    def __init__(self, operator, function):
        
        self.function = function     
        self.operator = operator
        self.L = 2*self.function.alpha*operator.norm()**2
        super(FunctionOperatorComposition, self).__init__()
        
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

    def gradient(self, x):
        
        ''' Gradient takes into account the Operator'''
        
        return self.operator.adjoint(self.function.gradient(self.operator.direct(x)))
        
                       