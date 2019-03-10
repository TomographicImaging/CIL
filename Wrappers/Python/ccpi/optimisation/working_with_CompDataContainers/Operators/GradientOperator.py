#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 22:50:04 2019

@author: evangelos
"""

from Operators.operators import Operator
from ccpi.optimisation.ops import PowerMethodNonsquare
from ccpi.framework import ImageData, DataContainer
import numpy as np
from Operators.FiniteDifferenceOperator import FiniteDiff

#%%

class Gradient(Operator):
    
    def __init__(self, gm_domain, gm_range=None, bnd_cond = 'Neumann', **kwargs):
        
        super(Gradient, self).__init__() 
        
        self.gm_domain = gm_domain # Domain of Grad Operator
        self.gm_range = gm_range # Range of Grad Operator
        self.bnd_cond = bnd_cond # Boundary conditions of Finite Differences

        
        if self.gm_range is None:
           self.gm_range =  ((len(self.gm_domain),)+self.gm_domain)
    
        # Kwargs Default options            
        self.memopt = kwargs.get('memopt',False)
        self.correlation = kwargs.get('correlation','Space') 
        
        #TODO not tested yet, operator norm???
        self.voxel_size = kwargs.get('voxel_size',[1]*len(gm_domain))  
                                             
        
    def direct(self, x, out=None):
        
        tmp = np.zeros(self.gm_range)
        for i in range(len(self.gm_domain)):
            tmp[i] = FiniteDiff(self.gm_domain, direction = i, bnd_cond = self.bnd_cond).direct(x.as_array())/self.voxel_size[i]            
#        return type(x)(tmp)
        if out is None:    
            return type(x)(tmp)
        else:
            return out.fill(tmp)
    
    def adjoint(self, x, out=None):
            
        tmp = np.zeros(self.gm_domain)
        for i in range(len(self.gm_domain)):
            tmp+=FiniteDiff(self.gm_domain, direction = i, bnd_cond = self.bnd_cond).adjoint(x.as_array()[i])/self.voxel_size[i]  
        if out is None:    
            return type(x)(-tmp)
        else:
            return out.fill(tmp)            
  
        
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
    
    N, M = (500,500)
    ig = (N,M)
    G = Gradient(ig)
    u = DataContainer(np.random.randint(10, size=G.domain_dim()))
    w = DataContainer(np.random.randint(10, size=G.range_dim()))
#    w = [DataContainer(np.random.randint(10, size=G.domain_dim())),\
#         DataContainer(np.random.randint(10, size=G.domain_dim()))]

    # domain_dim
    print('Domain {}'.format(G.domain_dim()))
    
    # range_dim
    print('Range {}'.format(G.range_dim()))
    
    # Direct
    z = G.direct(u)
    
    # Adjoint
    z1 = G.adjoint(w)

    print(z)
    print(z1)
    
    LHS = (G.direct(u)*w).sum()
    RHS = (u * G.adjoint(w)).sum()
#    
    print(LHS,RHS)
    print(G.norm())
    
    ##########################################################################
    ## with out
    import time
    
    t = time.time()
    for i in range(5000):
        z = G.direct(u)
        w = G.adjoint(z)
#        print(id(z))
    t1 = time.time()    
    print(t1-t)    
        
    zout = ImageData(np.zeros((2,N,M)))
    zout1 = ImageData(np.zeros((N,M)))
    
    t2 = time.time()
    for i in range(5000):
        G.direct(u, out=zout)
        G.adjoint(zout, out=zout1)
    t3 = time.time()    
    print(t3-t2)     

    
    
    
#    print(G.adjoint(G.direct(u)))
    
      
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    