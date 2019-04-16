#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 22:53:55 2019

@author: evangelos
"""

from ccpi.optimisation.operators import Gradient, Operator, LinearOperator, ScaledOperator
from ccpi.optimisation.ops import PowerMethodNonsquare
from ccpi.framework import ImageData, ImageGeometry, BlockGeometry, BlockDataContainer
import numpy 
from ccpi.optimisation.operators import FiniteDiff, SparseFiniteDiff


class SymmetrizedGradient(Gradient):
    
    def __init__(self, gm_domain, bnd_cond = 'Neumann', **kwargs):
        
        super(SymmetrizedGradient, self).__init__(gm_domain, bnd_cond, **kwargs) 
        
        '''
         Domain of SymGrad is the Range of Gradient
        '''
        self.gm_domain = self.gm_range 
        self.bnd_cond = bnd_cond
        
        self.channels = self.gm_range.get_item(0).channels
        
        if self.correlation=='Space':
            if self.channels>1:
                pass
            else: 
#                # 2D image ---> Dx v1, Dyv2, Dx
                tmp = self.gm_domain.geometries + (self.gm_domain.get_item(0),)
                self.gm_range = BlockGeometry(*tmp )
                self.ind1 = range(self.gm_domain.get_item(0).length)
                self.ind2 = range(self.gm_domain.get_item(0).length-1, -1, -1)
#                self.order = myorder = [0,1,2 3]
                
        elif self.correlation=='SpaceChannels':
            if self.channels>1:
                pass
            else:
                raise ValueError('No channels to correlate')        
                                                     
        
    def direct(self, x, out=None):
        
#        tmp = numpy.zeros(self.gm_range)
#        tmp[0] = FiniteDiff(self.gm_domain[1:], direction = 1, bnd_cond = self.bnd_cond).adjoint(x.as_array()[0])
#        tmp[1] = FiniteDiff(self.gm_domain[1:], direction = 0, bnd_cond = self.bnd_cond).adjoint(x.as_array()[1])
#        tmp[2] = 0.5 * (FiniteDiff(self.gm_domain[1:], direction = 0, bnd_cond = self.bnd_cond).adjoint(x.as_array()[0]) +
#                        FiniteDiff(self.gm_domain[1:], direction = 1, bnd_cond = self.bnd_cond).adjoint(x.as_array()[1]) )
#        
#        return type(x)(tmp)

        tmp = [[None]*2]*2
        for i in range(2):
            for j in range(2):  
                tmp[i][j]=FiniteDiff(self.gm_domain.get_item(0), direction = i, bnd_cond = self.bnd_cond).adjoint(x.get_item(j))
        tmp = numpy.array(tmp)
        z = 0.5 * (tmp.T + tmp)
        z = z.to
        
        return BlockDataContainer(*z.tolist())

    
    def adjoint(self, x, out=None):
        pass
        
        res = []
        for i in range(2):
            tmp = ImageData(np.zeros(x.get_item(0)))
            for j in range(2):                
                tmp += FiniteDiff(self.gm_domain.get_item(0), direction = i, bnd_cond = self.bnd_cond).direct(x.get_item(j))
            res.append(tmp)   
        return res            
                
                
        
#        for 
        
#        tmp = numpy.zeros(self.gm_domain)
#        
#        tmp[0] = FiniteDiff(self.gm_domain[1:], direction = 1, bnd_cond = self.bnd_cond).direct(x.as_array()[0]) +  \
#                 FiniteDiff(self.gm_domain[1:], direction = 0, bnd_cond = self.bnd_cond).direct(x.as_array()[2])
#                 
#        tmp[1] = FiniteDiff(self.gm_domain[1:], direction = 1, bnd_cond = self.bnd_cond).direct(x.as_array()[2]) +  \
#                 FiniteDiff(self.gm_domain[1:], direction = 0, bnd_cond = self.bnd_cond).direct(x.as_array()[1])                 
#
#        return type(x)(tmp)          
            
    def alloc_domain_dim(self):
        return ImageData(numpy.zeros(self.gm_domain))
    
    def alloc_range_dim(self):
        return ImageData(numpy.zeros(self.range_dim))
    
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
    from ccpi.framework import DataContainer
    from ccpi.optimisation.operators import Gradient, BlockOperator, FiniteDiff
    import numpy as np
    
    N, M = 2, 3
    K = 2
    
    ig1 = ImageGeometry(N, M)
    ig2 = ImageGeometry(N, M, channels=K)
    
    E1 = SymmetrizedGradient(ig1, correlation = 'Space', bnd_cond='Neumann')
    E2 = SymmetrizedGradient(ig2, correlation = 'SpaceChannels', bnd_cond='Periodic')
    
    print(E1.domain_geometry().shape)
    print(E2.domain_geometry().shape)
    
    u1 = E1.gm_domain.allocate('random_int')
    u2 = E2.gm_domain.allocate('random_int')
        
    
    res = E1.direct(u1) 
    
    res1 = E1.adjoint(res)
    
#    Dx = FiniteDiff(ig1, direction = 1, bnd_cond = 'Neumann')
#    Dy = FiniteDiff(ig1, direction = 0, bnd_cond = 'Neumann')
#    
#    B = BlockOperator(Dy, Dx)
#    V = BlockDataContainer(u1,u2)
#    
#    res = B.adjoint(V)
    
#    ig = (N,M)
#    ig2 = (2,) + ig
#    ig3 = (3,) + ig
#    u1 = ig.allocate('random_int')
#    w1 = E.gm_range.allocate('random_int')
#    DataContainer(np.random.randint(10, size=ig3))
    
    
    
#    d1 = E.direct(u1)
#    d2 = E.adjoint(w1)
    
#    LHS = (d1.as_array()[0]*w1.as_array()[0] + \
#           d1.as_array()[1]*w1.as_array()[1] + \
#           2*d1.as_array()[2]*w1.as_array()[2]).sum()
#    
#    RHS = (u1.as_array()[0]*d2.as_array()[0] + \
#           u1.as_array()[1]*d2.as_array()[1]).sum()   
#    
#    
#    print(LHS, RHS, E.norm())
#    
    
#    
    
    
    
    
    
    
    
    
    
    
    