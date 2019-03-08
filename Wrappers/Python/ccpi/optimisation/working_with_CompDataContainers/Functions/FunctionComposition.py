#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 19:45:06 2019

@author: evangelos
"""

import numpy as np
from ccpi.optimisation.funcs import Function
from ccpi.framework import DataContainer, ImageData, ImageGeometry 
from Operators.CompositeDataContainer import CompositeDataContainer

class FunctionOperatorComposition(Function):
    
    def __init__(self, function, operator):
        
        self.function = functions      
        self.operator = operator
        self.grad_Lipschitz_cnst = 2*self.function.alpha*operator.norm()**2
        super(FunctionOperatorComposition, self).__init__()
        
    def __call__(self, x):
    
        return self.function(operator.direct(x))   

    def call_adjoint(self, x):
    
        return self.function(operator.adjoint(x))  

    def convex_conjugate(self, x):
        
        return self.function.convex_conjugate(x)

    def proximal(self, x, tau):
        
        ''' proximal does not take into account the Operator'''
        
        return self.function.proximal(x, tau, out=None)

    def proximal_conjugate(self, x, tau):    

        ''' proximal conjugate does not take into account the Operator'''
        
        return self.function.proximal_conjugate(x, tau, out=None) 

    def gradient(self, x):
        
        ''' Gradient takes into account the Operator'''
        
        return self.adjoint(self.function.gradient(self.operator.direct(x)))
        
                       
class BlockFunction(Function):
    
    def __init__(self, operator, *functions):
                
        self.functions = functions      
        self.operator = operator
        self.length = len(self.functions)
        
        super(BlockFunction, self).__init__()
        
    def __call__(self, x):
    
        tmp = operator.direct(x)       
                
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
        out = [None]*self.length
        for i in range(self.length):
            out[i] = self.functions[i].proximal_conjugate(x.get_item(i), tau)

        return CompositeDataContainer(*out) 
    
    def proximal(self, x, tau, out = None):
     
        ''' proximal does not take into account the BlockOperator'''
        out = [None]*self.length
        for i in range(self.length):
            out[i] = self.functions[i].proximal(x.get_item(i), tau)

        return CompositeDataContainer(*out)     
    
    def gradient(self,x, out=None):
        pass
    
    
class FunctionComposition_new(Function):
    
     def __init__(self, operator, *functions):
        
        self.functions = functions      
        self.operator = operator
        self.length = len(self.functions)
          
        super(FunctionComposition_new, self).__init__()
                    
     def __call__(self, x):
         
        if isinstance(x, ImageData):
            x =  CompositeDataContainer(x)         
        
        t = 0                
        for i in range(x.shape[0]):
            t += self.functions[i](x.get_item(i))               
        return t 
    
     def convex_conjugate(self, x):
         
        if isinstance(x, ImageData):
            x =  CompositeDataContainer(x)         
        
        t = 0                
        for i in range(x.shape[0]):
            t += self.functions[i].convex_conjugate(x.get_item(i))               
        return t     
                        
     def proximal_conjugate(self, x, tau, out = None):
     
        if isinstance(x, ImageData):
            x =  CompositeDataContainer(x)
            
        out = [None]*self.length
        for i in range(self.length):
            out[i] = self.functions[i].proximal_conjugate(x.get_item(i), tau)
        
        if self.length==1:
            return ImageData(*out)   
        else:
            return CompositeDataContainer(*out)   

     def proximal(self, x, tau, out = None):
     
        if isinstance(x, ImageData):
            x =  CompositeDataContainer(x)
            
        out = [None]*self.length
        for i in range(self.length):
            out[i] = self.functions[i].proximal(x.get_item(i), tau)
        
        if self.length==1:
            return ImageData(*out)   
        else:
            return CompositeDataContainer(*out) 
        
        
if __name__ == '__main__':

    from  operators import Operator
    from  IdentityOperator import Identity