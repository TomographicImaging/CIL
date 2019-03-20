#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 10:01:31 2019

@author: evangelos
"""

import numpy as np
from ccpi.optimisation.functions import Function
from ccpi.framework import BlockDataContainer

class BlockFunction(Function):
    
    def __init__(self, *functions):
                
        self.functions = functions      
        self.length = len(self.functions)
        
        super(BlockFunction, self).__init__()
        
    def get_item(self, index):
        return self.functions[index]       
        
    def __call__(self, x):
                         
        t = 0                
        for i in range(x.shape[0]):
            t += self.functions[i](tmp.get_item(i))               
        return t
    
#    def call_adjoint(self, x):
#    
#        tmp = operator.adjoint(x)       
#                
#        t = 0                
#        for i in range(tmp.shape[0]):
#            t += self.functions[i](tmp.get_item(i))               
#        return t 
    
    def convex_conjugate(self, x):
               
        ''' Convex_conjugate does not take into account the BlockOperator'''        
        t = 0                
        for i in range(x.shape[0]):
            t += self.functions[i].convex_conjugate(x.get_item(i))               
        return t  
    
    
    def proximal_conjugate(self, x, tau, out = None):
         
        ''' proximal_conjugate does not take into account the BlockOperator'''
        out = [None]*self.length
        for i in range(self.length):
            out[i] = self.functions[i].proximal_conjugate(x.get_item(i), tau)

        return BlockDataContainer(*out) 
    
    def proximal(self, x, tau, out = None):
     
        ''' proximal does not take into account the BlockOperator'''
        out = [None]*self.length
        for i in range(self.length):
            out[i] = self.functions[i].proximal(x.get_item(i), tau)

        return BlockDataContainer(*out)     
    
    def gradient(self,x, out=None):
        pass