#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 19:43:12 2019

@author: evangelos
"""

import numpy as np
#from ccpi.optimisation.funcs import Function
from ccpi.optimisation.functions import Function
from ccpi.framework import DataContainer, ImageData, ImageGeometry 

############################   mixed_L1,2NORM FUNCTIONS   #############################
class mixed_L12Norm(Function):
    
    def __init__(self, alpha, **kwargs):

        super(mixed_L12Norm, self).__init__() 
        
        self.alpha = alpha 
        self.b = kwargs.get('b',None)                
        self.sym_grad = kwargs.get('sym_grad',False)
        
    def __call__(self,x):
        
        if self.b is None:
            tmp1 = x
        else:
            tmp1 = x - self.b            
#        
        if self.sym_grad:
            tmp = np.sqrt(tmp1.as_array()[0]**2 +  tmp1.as_array()[1]**2 +  2*tmp1.as_array()[2]**2)
        else:
            tmp = ImageData(tmp1.power(2).sum(axis=0)).sqrt()
            
        return self.alpha*tmp.sum()          
                            
    def gradient(self,x):
        return ValueError('Not Differentiable')
                            
    def convex_conjugate(self,x):
        return 0
    
    def proximal(self, x, tau):
        pass
    
    def proximal_conjugate(self, x, tau): 
        
        if self.sym_grad:
                tmp2 = np.sqrt(x.as_array()[0]**2 +  x.as_array()[1]**2 +  2*x.as_array()[2]**2)/self.alpha
                res = x.divide(ImageData(tmp2).maximum(1.0))                                
        else:
            
                tmp = (x.get_item(0).power(2) + x.get_item(1).power(2)).sqrt()
                res = x.divide((tmp/self.alpha).maximum(1.0))
#                res = x.divide((ImageData(x.power(2).sum(axis=0)).sqrt()/self.alpha).maximum(1.0))

                                                   
        return res 
