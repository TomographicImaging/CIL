#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 10:01:31 2019

@author: evangelos
"""

import numpy as np
from ccpi.optimisation.funcs import Function
from Operators.CompositeDataContainer import CompositeDataContainer

class BlockFunction(Function):
    
    def __init__(self, operator, *functions):
                
        self.functions = functions      
        self.operator = operator
        self.length = len(self.functions)
        
        super(BlockFunction, self).__init__()
        
    def __call__(self, x):
    
        tmp = self.operator.direct(x)       
                
        t = 0                
        for i in range(tmp.shape[0]):
            t += self.functions[i](tmp.get_item(i))               
        return t
    
    def call_adjoint(self, x):
    
        tmp = operator.adjoint(x)       
                
        t = 0                
        for i in range(tmp.shape[0]):
            t += self.functions[i](tmp.get_item(i))               
        return t 
    
    def convex_conjugate(self, x):
               
        ''' Convex_conjugate does not take into account the BlockOperator'''        
        t = 0                
        for i in range(x.shape[0]):
            t += self.functions[i].convex_conjugate(x.get_item(i))               
        return t  
    
    
    def proximal_conjugate(self, x, tau, out = None):
         
        ''' proximal_conjugate does not take into account the BlockOperator'''
        tmp = [None]*self.length
        for i in range(self.length):
            tmp[i] = self.functions[i].proximal_conjugate(x.get_item(i), tau)
        
        if out is None:
            return CompositeDataContainer(*tmp) 
        else:
            out = tmp
            return CompositeDataContainer(*out)  
    
    def proximal(self, x, tau, out = None):
     
        ''' proximal does not take into account the BlockOperator'''
        out = [None]*self.length
        for i in range(self.length):
            out[i] = self.functions[i].proximal(x.get_item(i), tau)

        return CompositeDataContainer(*out)     
    
    def gradient(self,x, out=None):
        pass