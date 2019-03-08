#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 22:53:55 2019

@author: evangelos
"""

from Operators.FiniteDifferenceOperator import FiniteDiff
from Operators.operators import Operator
#from operators import Operator
from ccpi.optimisation.ops import PowerMethodNonsquare
from ccpi.framework import ImageData, DataContainer
import numpy as np
#fro
#from FiniteDifferenceOperator import FiniteDiff

class SymmetrizedGradient(Operator):
    
    def __init__(self, gm_domain, gm_range, bnd_cond = 'Neumann', **kwargs):
        
        super(SymmetrizedGradient, self).__init__() 
        
        self.gm_domain = gm_domain # Domain of Grad Operator
        self.gm_range = gm_range # Range of Grad Operator
        self.bnd_cond = bnd_cond # Boundary conditions of Finite Differences
    
        # Kwargs Default options            
        self.memopt = kwargs.get('memopt',False)
        self.correlation = kwargs.get('correlation','Space') 
        
        #TODO not tested yet, operator norm???
        self.voxel_size = kwargs.get('voxel_size',[1]*len(gm_domain))  
                                             
        
    def direct(self, x, out=None):
        
        tmp = np.zeros(self.gm_range)
        tmp[0] = FiniteDiff(self.gm_domain[1:], direction = 1, bnd_cond = self.bnd_cond).adjoint(x.as_array()[0])
        tmp[1] = FiniteDiff(self.gm_domain[1:], direction = 0, bnd_cond = self.bnd_cond).adjoint(x.as_array()[1])
        tmp[2] = 0.5 * (FiniteDiff(self.gm_domain[1:], direction = 0, bnd_cond = self.bnd_cond).adjoint(x.as_array()[0]) +
                        FiniteDiff(self.gm_domain[1:], direction = 1, bnd_cond = self.bnd_cond).adjoint(x.as_array()[1]) )
        
        return type(x)(tmp)
    
    
    def adjoint(self, x, out=None):
        
        tmp = np.zeros(self.gm_domain)
        
        tmp[0] = FiniteDiff(self.gm_domain[1:], direction = 1, bnd_cond = self.bnd_cond).direct(x.as_array()[0]) +  \
                 FiniteDiff(self.gm_domain[1:], direction = 0, bnd_cond = self.bnd_cond).direct(x.as_array()[2])
                 
        tmp[1] = FiniteDiff(self.gm_domain[1:], direction = 1, bnd_cond = self.bnd_cond).direct(x.as_array()[2]) +  \
                 FiniteDiff(self.gm_domain[1:], direction = 0, bnd_cond = self.bnd_cond).direct(x.as_array()[1])                 

        return type(x)(tmp)          
            
    def alloc_domain_dim(self):
        return ImageData(np.zeros(self.gm_domain))
    
    def alloc_range_dim(self):
        return ImageData(np.zeros(self.range_dim))
    
    def domain_dim(self):
        return self.gm_domain
    
    def range_dim(self):
        return self.gm_range
                                   
    def norm(self):
#        return np.sqrt(4*len(self.domainDim()))        
        #TODO this takes time for big ImageData
        # for 2D ||grad|| = sqrt(8), 3D ||grad|| = sqrt(12)        
        x0 = ImageData(np.random.random_sample(self.domain_dim()))
        self.s1, sall, svec = PowerMethodNonsquare(self, 25, x0)
        return self.s1 
    


if __name__ == '__main__':   
    
    ###########################################################################  
    ## Symmetrized Gradient
    
    N, M = 2, 3
    ig = (N,M)
    ig2 = (2,) + ig
    ig3 = (3,) + ig
    u1 = DataContainer(np.random.randint(10, size=ig2))
    w1 = DataContainer(np.random.randint(10, size=ig3))
    
    E = SymmetrizedGradient(ig2,ig3)
    
    d1 = E.direct(u1)
    d2 = E.adjoint(w1)
    
    LHS = (d1.as_array()[0]*w1.as_array()[0] + \
           d1.as_array()[1]*w1.as_array()[1] + \
           2*d1.as_array()[2]*w1.as_array()[2]).sum()
    
    RHS = (u1.as_array()[0]*d2.as_array()[0] + \
           u1.as_array()[1]*d2.as_array()[1]).sum()   
    
    
    print(LHS, RHS, E.norm())
    
    
#    
    
    
    
    
    
    
    
    
    
    
    