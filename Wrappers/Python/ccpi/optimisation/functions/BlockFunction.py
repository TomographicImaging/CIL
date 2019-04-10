#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 10:01:31 2019

@author: evangelos
"""

import numpy as np
#from ccpi.optimisation.funcs import Function
from ccpi.optimisation.functions import Function
from ccpi.framework import BlockDataContainer
from numbers import Number

class BlockFunction(Function):
    '''A Block vector of Functions
    
    .. math::

      f = [f_1,f_2,f_3]
      f([x_1,x_2,x_3]) = f_1(x_1) + f_2(x_2) + f_3(x_3)

    '''
    def __init__(self, *functions):
        '''Creator'''
        self.functions = functions      
        self.length = len(self.functions)
        
        super(BlockFunction, self).__init__()
        
    def __call__(self, x):
        '''evaluates the BlockFunction on the BlockDataContainer
        
        :param: x (BlockDataContainer): must have as many rows as self.length

        returns sum(f_i(x_i))
        '''
        if self.length != x.shape[0]:
            raise ValueError('BlockFunction and BlockDataContainer have incompatible size')
        t = 0                
        for i in range(x.shape[0]):
            t += self.functions[i](x.get_item(i))
        return t
    
    def convex_conjugate(self, x):
        '''Convex_conjugate does not take into account the BlockOperator'''        
        t = 0                
        for i in range(x.shape[0]):
            t += self.functions[i].convex_conjugate(x.get_item(i))
        return t  
    
    
    def proximal_conjugate(self, x, tau, out = None):
        '''proximal_conjugate does not take into account the BlockOperator'''
        if out is not None:
            if isinstance(tau, Number):
                for i in range(self.length):
                    self.functions[i].proximal_conjugate(x.get_item(i), tau, out=out.get_item(i))
            else:
                for i in range(self.length):
                    self.functions[i].proximal_conjugate(x.get_item(i), tau.get_item(i),out=out.get_item(i))
            
        else:
                
            out = [None]*self.length
            if isinstance(tau, Number):
                for i in range(self.length):
                    out[i] = self.functions[i].proximal_conjugate(x.get_item(i), tau)
            else:
                for i in range(self.length):
                    out[i] = self.functions[i].proximal_conjugate(x.get_item(i), tau.get_item(i))
            
            return BlockDataContainer(*out) 
    
    def proximal(self, x, tau, out = None):
        '''proximal does not take into account the BlockOperator'''
        out = [None]*self.length
        if isinstance(tau, Number):
            for i in range(self.length):
                out[i] = self.functions[i].proximal(x.get_item(i), tau)
        else:
            for i in range(self.length):
                out[i] = self.functions[i].proximal(x.get_item(i), tau.get_item(i))                        

        return BlockDataContainer(*out)     
    
    def gradient(self,x, out=None):
        '''FIXME: gradient returns pass'''
        pass