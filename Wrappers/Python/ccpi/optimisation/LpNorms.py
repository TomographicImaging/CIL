#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 16:37:05 2019

@author: evangelos
"""

import numpy as np
from operators import Operator
from ccpi.optimisation.funcs import Function

from ccpi.optimisation.ops import PowerMethodNonsquare
from ccpi.framework import ImageData, ImageGeometry

class L1Norm(Function):

    def __init__(self, A , alpha=1, **kwargs):
        self.A = A    
        self.alpha = alpha
        
        self.b = kwargs.get('b',None)
        self.memopt = kwargs.get('memopt',False)
        self.gm_domain = self.A.domain_dim()

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
        
        
if __name__ == '__main__':     
    
    u = ImageData(np.random.randint(10, size = (2,3)))
    bnd_cond = 'Neumann'
    Gxu = FiniteDiff(u.shape, direction = 1, bnd_cond = bnd_cond)
    Gyu = FiniteDiff(u.shape, direction = 0, bnd_cond = bnd_cond)    
    
    Grad2D = CompositeOperator((2,1), Gxu, Gyu)
    
    f = L1Norm(Grad2D)
    
#    x = DataContainer(np.random.randint(10, size = Id.domain_dim()))
    
    
    
    
    
    
    
    
    
    f = L1Norm(Id,alpha=2)
#    print()
                                                                
        