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
#        self.gm_range = kwargs.get('gm_range',None)
        
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
            
#        if self.correlation=='SpaceChannels':
#            self.gm_range = BlockGeometry(*[self.gm_domain]*(len(self.gm_domain)))
                    
#        self.gm_range = gm_range # Range of Grad Operator
        
         # Boundary conditions of Finite Differences
                
#        if len(self.gm_domain)==2:
#            print('2')
##            print(self.gm_domain)
#            bg = BlockGeometry(self.gm_domain, self.gm_domain)
#            print(bg)
##            self.gm_range = bg
                
#        if self.gm_range is None:
#           self.gm_range =  ((len(self.gm_domain),)+self.gm_domain)
    
        # Kwargs Default options            

                
        #TODO not tested yet, operator norm???
#        self.voxel_size = kwargs.get('voxel_size',[1]*len(gm_domain))  
                                             
        
    def direct(self, x, out=None):
        
        tmp = self.gm_range.allocate()
        for i in range(tmp.shape[0]):
            tmp.get_item(i).fill(FiniteDiff(self.gm_domain, direction = i, bnd_cond = self.bnd_cond).direct(x))
        return tmp    
        
#        tmp = np.zeros(self.gm_range)
#        for i in range(len(self.gm_domain)):
#            tmp[i] = FiniteDiff(self.gm_domain, direction = i, bnd_cond = self.bnd_cond).direct(x.as_array())/self.voxel_size[i]            
#        return type(x)(tmp)
#        return type(x)(tmp)
    
    def adjoint(self, x, out=None):
        
        tmp = self.gm_domain.allocate()
        for i in range(x.shape[0]):
            tmp+=FiniteDiff(self.gm_domain, direction = i, bnd_cond = self.bnd_cond).adjoint(x.get_item(i))
        return tmp    
            
#        tmp = np.zeros(self.gm_domain)
#        for i in range(len(self.gm_domain)):
#            tmp+=FiniteDiff(self.gm_domain, direction = i, bnd_cond = self.bnd_cond).adjoint(x.as_array()[i])/self.voxel_size[i]  
#        return type(x)(-tmp)
        
#    def alloc_domain_dim(self):
#        return ImageData(np.zeros(self.gm_domain))
    
#    def alloc_range_dim(self):
#        return ImageData(np.zeros(self.range_dim))
    
    def domain_geometry(self):
        return self.gm_domain
    
    def range_geometry(self):
        return self.gm_range
                                   
    def norm(self):
#        return np.sqrt(4*len(self.domainDim()))        
        #TODO this takes time for big ImageData
        # for 2D ||grad|| = sqrt(8), 3D ||grad|| = sqrt(12)        
#        x0 = ImageData(np.random.random_sample(self.domain_dim()))
        x0 = self.gm_domain.allocate('random')
        self.s1, sall, svec = PowerMethodNonsquare(self, 50, x0)
        return self.s1
    
    
if __name__ == '__main__':
    
    N, M = 200, 300
    
#    ig = ImageGeometry(voxel_num_x = M, voxel_num_y = N)    
#    u = ig.allocate('random_int')
#    G = Gradient(ig)
    
#    print(G.domain_geometry().allocate().shape)
#    print(G.range_geometry().allocate().shape)
#    
#    print(G.domain_geometry().allocate())
#    print(G.range_geometry().allocate())  
#    
#    print(G.domain_geometry().allocate())
#    print(G.range_geometry().allocate().get_item(0).as_array())
#    print(G.range_geometry().allocate().get_item(1).as_array())
    
    #########################################################################
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
    
    
    u = ig1.allocate('random_int')
    w = G1.range_geometry().allocate('random_int')
#    g1 = G1.range_geometry()
    LHS = (G1.direct(u)*w).sum()
    RHS = (u * G1.adjoint(w)).sum()
    
    print(LHS,RHS)
    print(G1.norm())
    
#    d1 = G1.direct(u)
#    d2 = G1.adjoint(d1)
    
    
    
#    G = FiniteDiff(ig, direction=0, bnd_cond = 'Neumann')
#    print(u.as_array())    
#    print(G.direct(u).as_array())
    
    
    
#    ig = (N,M)
#    G = Gradient(ig)
#    u = DataContainer(np.random.randint(10, size=G.domain_dim()))
#    w = DataContainer(np.random.randint(10, size=G.range_dim()))
##    w = [DataContainer(np.random.randint(10, size=G.domain_dim())),\
##         DataContainer(np.random.randint(10, size=G.domain_dim()))]
#
#    # domain_dim
#    print('Domain {}'.format(G.domain_geometry()))
#    
#    # range_dim
#    print('Range {}'.format(G.range_geometry()))
#    
#    # Direct
#    z = G.direct(u)
#    
#    # Adjoint
#    z1 = G.adjoint(w)
#
#    print(z)
#    print(z1)
#    
#    LHS = (G.direct(u)*w).sum()
#    RHS = (u * G.adjoint(w)).sum()
##    
#    print(LHS,RHS)
#    print(G.norm())
#    
##    print(G.adjoint(G.direct(u)))


      
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    