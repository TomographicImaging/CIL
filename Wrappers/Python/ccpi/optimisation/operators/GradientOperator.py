#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 22:50:04 2019

@author: evangelos
"""

from ccpi.optimisation.operators import Operator, LinearOperator
from ccpi.optimisation.ops import PowerMethodNonsquare
from ccpi.framework import ImageData
import numpy as np
from ccpi.optimisation.operators import FiniteDiff
from ccpi.framework import ImageGeometry, BlockGeometry
from ccpi.framework import BlockDataContainer
import functools


#%%

class Gradient(LinearOperator):
    
    def __init__(self, gm_domain, bnd_cond = 'Neumann', **kwargs):
        
        super(Gradient, self).__init__() 
                
        self.gm_domain = gm_domain # Domain of Grad Operator
        
        self.correlation = kwargs.get('correlation','Space')
        
        if self.correlation=='Space':
            if self.gm_domain.channels>1:
                self.gm_range = BlockGeometry(*[self.gm_domain]*(len(self.gm_domain)-1))              
            else:    
                self.gm_range = BlockGeometry(*[self.gm_domain]*(len(self.gm_domain)))
        else:
            if self.gm_domain.channels>1:
                self.gm_range = BlockGeometry(*[self.gm_domain]*(len(self.gm_domain)))
            else:
                raise ValueError('No channels to correlate')
         
        self.bnd_cond = bnd_cond    
                                                         
        
    def direct(self, x, out=None):
        
        tmp = self.gm_range.allocate()
        for i in range(tmp.shape[0]):
            tmp.get_item(i).fill(FiniteDiff(self.gm_domain, direction = i, bnd_cond = self.bnd_cond).direct(x))
        return tmp    
        
    def adjoint(self, x, out=None):
        
        tmp = self.gm_domain.allocate()
        for i in range(x.shape[0]):
            tmp+=FiniteDiff(self.gm_domain, direction = i, bnd_cond = self.bnd_cond).adjoint(x.get_item(i))
        return tmp    
            
    
    def domain_geometry(self):
        return self.gm_domain
    
    def range_geometry(self):
        return self.gm_range
                                   
    def norm(self):

        x0 = self.gm_domain.allocate('random')
        self.s1, sall, svec = PowerMethodNonsquare(self, 50, x0)
        return self.s1
    
    
if __name__ == '__main__':
    
    N, M = 200, 300
    
    # check range geometry, examples
    K=2
    ig1 = ImageGeometry(voxel_num_x = M, voxel_num_y = N) 
    ig2 = ImageGeometry(voxel_num_x = M, voxel_num_y = N, voxel_num_z = K) 
    ig3 = ImageGeometry(voxel_num_x = M, voxel_num_y = N, channels = 2) 
    ig4 = ImageGeometry(voxel_num_x = M, voxel_num_y = N, channels = 3, voxel_num_z= K) 
    
    G1 = Gradient(ig1, correlation = 'Space') 
    print(G1.range_geometry().shape, '2D no channels')
        
    G4 = Gradient(ig3, correlation = 'SpaceChannels')
    print(G4.range_geometry().shape, '2D with channels corr')
    G5 = Gradient(ig3, correlation = 'Space')
    print(G5.range_geometry().shape, '2D with channels no corr')
    
    G6 = Gradient(ig4, correlation = 'Space')
    print(G6.range_geometry().shape, '3D with channels no corr')
    G7 = Gradient(ig4, correlation = 'SpaceChannels')
    print(G7.range_geometry().shape, '3D with channels with corr')
    
    
    u = ig1.allocate('random')
    w = G1.range_geometry().allocate('random_int')

    LHS = (G1.direct(u)*w).sum()
    RHS = (u * G1.adjoint(w)).sum()
        
    print(LHS,RHS)
    print(G1.norm())
    
    
    u1 = ig3.allocate('random')
    w1 = G4.range_geometry().allocate('random_int')
    LHS1 = (G4.direct(u1) * w1).sum()
    RHS1 = (u1 * G4.adjoint(w1)).sum() 
    print(G4.norm())
    print(LHS1, RHS1)
    
    u2 = ig4.allocate('random')
    w2 = G7.range_geometry().allocate('random_int')
    LHS2 = (G7.direct(u2) * w2).sum()
    RHS2 = (u2 * G7.adjoint(w2)).sum() 
    print(LHS2, RHS2)    
    print(G7.norm())
    
    
    

      
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    