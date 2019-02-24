#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 16:00:08 2019

@author: evangelos
"""
import numpy as np
from ccpi.optimisation.funcs import Function
from ccpi.framework import DataContainer, ImageData, ImageGeometry
from operators import *

#%%

###############################################################################
##################   REGULARISERS #############################################
###############################################################################

class TGV(Function):
    
    def __init__(self, beta, memopt = False):
        super(TGV,self).__init__()
#        self.alpha = alpha
        self.beta = beta
        self.memopt = memopt
        self.op = gradient(geometry=self)
        
    def __call__(self, u, w):  
        return self.alpha * np.sum(np.sqrt((self.op.direct(u) - w).power(2).sum(axis=0)))
    
    # TODO Almost correct
    def convex_conj(self,x):
        return 0 
    
    def proximal_conj(self, w, tau, out = None):  
        
        tmp2 = np.sqrt(w.as_array()[0]**2 +  w.as_array()[1]**2 +  2*w.as_array()[2]**2)/self.beta
        res = w.divide(ImageData(tmp2).maximum(1.0))
        
        
#        tmp1 = (x.power(2).sum(axis=0)).sqrt()/self.alpha
#        tmp2 = np.sqrt(w.as_array()[0]**2 +  w.as_array()[1]**2 +  2*w.as_array()[2]**2)/self.beta
#        res =  [x.divide(ImageData(tmp1).maximum(1.0)), \
#                w.divide(ImageData(tmp2).maximum(1.0)) ]                
        if self.memopt:    
            out.fill(type(w)(res, geometry = w.geometry))  
        else:
            return type(w)(res, geometry = w.geometry)    
    

class TV(Function):
    
    def __init__(self, alpha, memopt = False):
                
        self.alpha = alpha
        self.memopt = memopt
        self.op = gradient(geometry=self)
        super(TV, self).__init__()
#        self.gradient = gradient
#        op = gradient(geometry=self)
                        
    def __call__(self, x):  
        # TODO isotropic, anisotropic, multiple parameters              
        return self.alpha * np.sum(np.sqrt(self.op.direct(x).power(2).sum(axis=0)))
    
    # TODO Almost correct
    def convex_conj(self,x):
        return 0   
        
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
        
<<<<<<< HEAD
#    def pVariableDim(self):
#        return [ImageData(np.zeros(self.op.domainDim()), ig)]
=======
    def primal_var(self):
        #print(self.op.domainDim())
        return ImageData(np.zeros(self.op.domainDim()), geometry = self.op.geometry)
>>>>>>> 50e992f2a293b7f49fc995400f02a116439c00d2
    
#    def dVariableDim(self):
#        return [ImageData(np.zeros(self.op.rangeDim()))]



<<<<<<< HEAD
            

    

=======
alpha = 10
f = TV(alpha)
f.primal_var()
        
>>>>>>> 50e992f2a293b7f49fc995400f02a116439c00d2
        
        
