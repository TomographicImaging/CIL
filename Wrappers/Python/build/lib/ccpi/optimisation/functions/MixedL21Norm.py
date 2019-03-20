#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 19:43:12 2019

@author: evangelos
"""

import numpy as np
from ccpi.optimisation.functions import Function
from ccpi.framework import DataContainer, ImageData, \
                           ImageGeometry, BlockDataContainer 

############################   mixed_L1,2NORM FUNCTIONS   #####################
class MixedL21Norm(Function):
    
    def __init__(self, **kwargs):

        super(MixedL21Norm, self).__init__()                      
        self.sym_tensor = kwargs.get('sym_tensor',False)
        
    def __call__(self,x):
        
        ''' Evaluates L1,2Norm at point x'''
        
        if self.sym_grad:
            tmp = np.sqrt(tmp1.as_array()[0]**2 +  tmp1.as_array()[1]**2 +  2*tmp1.as_array()[2]**2)
        else:
            tmp = ImageData(tmp1.power(2).sum(axis=0)).sqrt()
            
        return         
                            
    def gradient(self,x, out=None):
        return ValueError('Not Differentiable')
                            
    def convex_conjugate(self,x):
        return 0
    
    def proximal(self, x, tau, out=None):
        pass
    
    def proximal_conjugate(self, x, tau, out=None): 
        
        if self.sym_tensor:
                tmp2 = np.sqrt(x.as_array()[0]**2 +  x.as_array()[1]**2 +  2*x.as_array()[2]**2)/self.alpha
                res = x.divide(ImageData(tmp2).maximum(1.0))                                
        else:
#                res = x.divide((ImageData(x.power(2).sum(axis=0)).sqrt()/self.alpha).maximum(1.0))
                
                a1 = x.get_item(0)
                a2 = x.get_item(1)
                c = ((((a1*a1) + (a2*a2)).sqrt())).maximum(1.0)
                res = BlockDataContainer(x.get_item(0)/c, x.get_item(1)/c)
                
                
#                res = x.divide(x.squared_norm()).sqrt()/self.alpha).maximum(1.0))
                                                   
        return res 


#class mixed_L12Norm(Function):
#    
#    def __init__(self, alpha, **kwargs):
#
#        super(mixed_L12Norm, self).__init__() 
#        
#        self.alpha = alpha 
#        self.b = kwargs.get('b',None)                
#        self.sym_grad = kwargs.get('sym_grad',False)
#        
#    def __call__(self,x):
#        
#        if self.b is None:
#            tmp1 = x
#        else:
#            tmp1 = x - self.b            
##        
#        if self.sym_grad:
#            tmp = np.sqrt(tmp1.as_array()[0]**2 +  tmp1.as_array()[1]**2 +  2*tmp1.as_array()[2]**2)
#        else:
#            tmp = ImageData(tmp1.power(2).sum(axis=0)).sqrt()
#            
#        return self.alpha*tmp.sum()          
#                            
#    def gradient(self,x):
#        return ValueError('Not Differentiable')
#                            
#    def convex_conjugate(self,x):
#        return 0
#    
#    def proximal(self, x, tau):
#        pass
#    
#    def proximal_conjugate(self, x, tau): 
#        
#        if self.sym_grad:
#                tmp2 = np.sqrt(x.as_array()[0]**2 +  x.as_array()[1]**2 +  2*x.as_array()[2]**2)/self.alpha
#                res = x.divide(ImageData(tmp2).maximum(1.0))                                
#        else:
##                res = x.divide((ImageData(x.power(2).sum(axis=0)).sqrt()/self.alpha).maximum(1.0))
#                
#                a1 = x.get_item(0)
#                a2 = x.get_item(1)
#                c = ((((a1*a1) + (a2*a2)).sqrt())/self.alpha).maximum(1.0)
#                res = BlockDataContainer(x.get_item(0)/c, x.get_item(1)/c)
#                
#                
##                res = x.divide(x.squared_norm()).sqrt()/self.alpha).maximum(1.0))
#                                                   
#        return res 
