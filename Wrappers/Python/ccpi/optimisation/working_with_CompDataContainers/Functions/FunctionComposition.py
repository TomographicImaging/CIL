#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 19:45:06 2019

@author: evangelos
"""

import numpy as np
from ccpi.optimisation.funcs import Function
from ccpi.framework import DataContainer, ImageData, ImageGeometry 

import sys
sys.path.insert(0,'/Users/evangelos/Desktop/Projects/CCPi/CCPi-Framework/Wrappers/Python/ccpi/optimisation/working_with_CompDataContainers/Operators')
from CompositeOperator_DataContainer import CompositeOperator, CompositeDataContainer

#from operators import CompositeDataContainer, Identity, CompositeOperator


#from numbers import Number
#from GradientOperator import Gradient

class FunctionComposition_new(Function):
    
     def __init__(self, operator, *functions):
        
        self.functions = functions      
        self.operator = operator
        self.length = len(self.functions)
        
#        if self.length==1:
#            self.L = self.functions[0].alpha*(self.operator.norm()**2)    
                        
        # length == to operator.shape[0]#    
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