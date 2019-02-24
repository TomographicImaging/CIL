#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 13:10:56 2019

@author: evangelos
"""

import numpy as np
from ccpi.optimisation.funcs import Function
from ccpi.framework import DataContainer, ImageData, ImageGeometry
from operators import *


#%%  
    
    
class L1Norm(Function):

    def __init__(self, A, alpha, **kwargs):
        self.A = A    
        self.alpha = alpha
        
        self.b = kwargs.get('b',None)
        self.memopt = kwargs.get('memopt',False)

        super(L1Norm, self).__init__() 
    
    def __call__(self,x):
        y = self.A.direct(x)
        tmp = np.sqrt(np.sum([y.as_array()[i]**2 for i in range(len(y.shape))]))
        eucl_norm = ImageData(tmp)        
        eucl_norm.__isub__(self.b)
        return y.abs().sum() * self.alpha 
                   
    def proximal(self, x, tau, out=None):      
        
        res = x.divide(x.abs()) * (x.abs() - tau).maximum(0)
        if self.b is not None:
            return res + self.b
        else:
            return res
            
    def gradient():
        pass
                
    def proximal_conj(self, x, tau, out = None): 
        
        if self.b==None:  
              
            if self.sym_grad:
                tmp2 = np.sqrt(x.as_array()[0]**2 +  x.as_array()[1]**2 +  2*x.as_array()[2]**2)/self.alpha
                res = x.divide(ImageData(tmp2).maximum(1.0))                                
            else:
                res = x.divide((ImageData(x.power(2).sum(axis=0)).sqrt()/self.alpha).maximum(1.0))  
                                                   
        else:            
            res =  (x - tau*self.b)/ ((x - tau*self.b)).abs().maximum(1.0)

        if self.memopt:    
            out.fill(type(x)(res, geometry = x.geometry))  
        else:
            return type(x)(res, geometry = x.geometry)  
                                                                
        
class L1NormOld(Function):

    def __init__(self,A,b=0,alpha=1.0,memopt = False):
        self.A = A
        self.b = b
        self.alpha = alpha
        self.memopt = memopt
        super(L1NormOld, self).__init__() 
        
    def __call__(self,x):
        y = self.A.direct(x)
        y.__isub__(self.b)
        return y.abs().sum() * self.alpha        
        
    def proximal(self, x, tau, out = None):
        
        res = x.divide((ImageData(x.power(2).sum(axis=0)).sqrt()/self.alpha).maximum(1.0))
                
        if self.memopt:    
            out.fill(type(x)(res, geometry = x.geometry))  
        else:
            return type(x)(res, geometry = x.geometry)
            
    # it is the same as proximal
    def proximal_conj(self, x, tau, out = None):  
                
        res = x.divide((ImageData(x.power(2).sum(axis=0)).sqrt()/self.alpha).maximum(1.0))
        
        if self.memopt:    
            out.fill(type(x)(res, geometry = x.geometry))  
        else:
            return type(x)(res, geometry = x.geometry)   
        
