#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 22:53:55 2019

@author: evangelos
"""

from ccpi.optimisation.operators import Gradient, Operator, LinearOperator, ScaledOperator
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
        
        tmp_gm = len(self.gm_domain.geometries)*self.gm_domain.geometries
        
        self.gm_range = BlockGeometry(*tmp_gm)
        
        self.FD = FiniteDiff(self.gm_domain, direction = 0, bnd_cond = self.bnd_cond)
        
        if self.gm_domain.shape[0]==2:
            self.order_ind = [0,2,1,3]
        else:
            self.order_ind = [0,3,6,1,4,7,2,5,8]            
            
        
#        if self.correlation=='Space':
#            if self.channels>1:
#                pass
#            else: 
##                # 2D image ---> Dx v1, Dyv2, Dx
#                tmp = self.gm_domain.geometries + (self.gm_domain.get_item(0),)
#                self.gm_range = BlockGeometry(*tmp )
#                self.ind1 = range(self.gm_domain.get_item(0).length)
#                self.ind2 = range(self.gm_domain.get_item(0).length-1, -1, -1)
##                self.order = myorder = [0,1,2 3]
#                
#        elif self.correlation=='SpaceChannels':
#            if self.channels>1:
#                pass
#            else:
#                raise ValueError('No channels to correlate')        
                                                     
        
    def direct(self, x, out=None):
        
        if out is None:
            
            tmp = []
            for i in range(self.gm_domain.shape[0]):
                for j in range(x.shape[0]):
                    self.FD.direction = i
                    tmp.append(self.FD.adjoint(x.get_item(j)))
                    
            tmp1 = [tmp[i] for i in self.order_ind]        
                    
            res = [0.5 * sum(x) for x in zip(tmp, tmp1)]   
                    
            return BlockDataContainer(*res)                    
                                     

    
    def adjoint(self, x, out=None):
        
        if out is None:
            
            tmp = [None]*self.gm_domain.shape[0]
            i = 0
            
            for k in range(self.gm_domain.shape[0]):
                tmp1 = 0
                for j in range(self.gm_domain.shape[0]):
                    self.FD.direction = j
                    tmp1 += self.FD.direct(x[i])                    
                    i+=1
                tmp[k] = tmp1  
            return BlockDataContainer(*tmp)
            
#            i = 0
#            
#            tmp = self.gm_domain.allocate()
#            for k in range(tmp.shape[0]):
#                tmp1 = 0
#                for j in range(tmp.shape[0]):
#                    self.FD.direction = j
#                    tmp1 += self.FD.direct(x[i])                    
#                    i+=1
#                tmp.get_item(k).fill(tmp1)  
#            return tmp                
                    
        else:
            pass
         
            
    def domain_geometry(self):
        return self.gm_domain
    
    def range_geometry(self):
        return self.gm_range
                                   
    def norm(self):

        #TODO need dot method for BlockDataContainer
        return numpy.sqrt(4*self.gm_domain.shape[0])
#        x0 = self.gm_domain.allocate('random_int')
#        self.s1, sall, svec = LinearOperator.PowerMethod(self, 10, x0)
#        return self.s1
    


if __name__ == '__main__':   
    
    ###########################################################################  
    ## Symmetrized Gradient
    from ccpi.framework import DataContainer
    from ccpi.optimisation.operators import Gradient, BlockOperator, FiniteDiff
    import numpy as np
    
    N, M = 2, 3
    K = 2
    
    ig1 = ImageGeometry(N, M)
    ig2 = ImageGeometry(N, M, K)
    
    E1 = SymmetrizedGradient(ig1, correlation = 'Space', bnd_cond='Neumann')
    E2 = SymmetrizedGradient(ig2, correlation = 'Space', bnd_cond='Periodic')
    
    print(E1.domain_geometry().shape, E1.range_geometry().shape)
    print(E2.domain_geometry().shape, E2.range_geometry().shape)
    
    u1 = E1.gm_domain.allocate('random_int')    
    a1 = ig1.allocate('random_int')
    a2 = ig1.allocate('random_int')
    a3 = ig1.allocate('random_int')
    w1 = BlockDataContainer(*[a1, a2, a3])
    w11 = BlockDataContainer(*[a1, a2, a2, a3])   
        
    sym_direct = [None]*3
    
    sym_direct[0] = FiniteDiff(ig1, direction = 1, bnd_cond = 'Neumann').adjoint(u1[0])
    sym_direct[1] = FiniteDiff(ig1, direction = 0, bnd_cond = 'Neumann').adjoint(u1[1])
    sym_direct[2] = 0.5 * (FiniteDiff(ig1, direction = 0, bnd_cond = 'Neumann').adjoint(u1[0]) + \
                    FiniteDiff(ig1, direction = 1, bnd_cond = 'Neumann').adjoint(u1[1]))
    
    sym_direct1 = [None]*4
    
    sym_direct1[0] = FiniteDiff(ig1, direction = 0, bnd_cond = 'Neumann').adjoint(u1[0])
    
    sym_direct1[1] = 0.5 * (FiniteDiff(ig1, direction = 0, bnd_cond = 'Neumann').adjoint(u1[1]) + \
                    FiniteDiff(ig1, direction = 1, bnd_cond = 'Neumann').adjoint(u1[0])) 
    
    sym_direct1[2] = 0.5 * (FiniteDiff(ig1, direction = 0, bnd_cond = 'Neumann').adjoint(u1[1]) + \
                    FiniteDiff(ig1, direction = 1, bnd_cond = 'Neumann').adjoint(u1[0]))
       
    sym_direct1[3] = FiniteDiff(ig1, direction = 1, bnd_cond = 'Neumann').adjoint(u1[1])   
    

        
    sym_adjoint = [None]*2
    
    sym_adjoint[0] = FiniteDiff(ig1, direction = 1, bnd_cond = 'Neumann').direct(w1[0]) + \
                     FiniteDiff(ig1, direction = 0, bnd_cond = 'Neumann').direct(w1[2])
    sym_adjoint[1] = FiniteDiff(ig1, direction = 1, bnd_cond = 'Neumann').direct(w1[2]) + \
                     FiniteDiff(ig1, direction = 0, bnd_cond = 'Neumann').direct(w1[1])  
                     
    sym_adjoint1 = [None]*2
    
    sym_adjoint1[0] = FiniteDiff(ig1, direction = 0, bnd_cond = 'Neumann').direct(w11[0]) + \
                     FiniteDiff(ig1, direction = 1, bnd_cond = 'Neumann').direct(w11[1])
    sym_adjoint1[1] = FiniteDiff(ig1, direction = 0, bnd_cond = 'Neumann').direct(w11[2]) + \
                     FiniteDiff(ig1, direction = 1, bnd_cond = 'Neumann').direct(w11[3])  
                 
    
    LHS = (sym_direct[0] * w1[0] + \
          sym_direct[1] * w1[1] + \
          2*sym_direct[2] * w1[2]).sum()
    
    RHS = (u1[0]*sym_adjoint[0] + u1[1]*sym_adjoint[1]).sum()
    
    print(LHS, RHS)
    
    LHS = (sym_direct1[0] * w11[0] + \
          sym_direct1[1] * w11[1] + \
          sym_direct1[2] * w11[2] + \
          sym_direct1[3] * w11[3] ).sum()
    
    RHS = (u1[0]*sym_adjoint1[0] + u1[1]*sym_adjoint1[1]).sum()
    
    print(LHS, RHS)    
    
    
    
    a1 = (E1.direct(u1) * w11).sum()
    b1 = (u1 * E1.adjoint(w11)).sum()
    print(a1, b1)
    
    
    u2 = E2.gm_domain.allocate('random_int')
    
    aa1 = ig2.allocate('random_int')
    aa2 = ig2.allocate('random_int')
    aa3 = ig2.allocate('random_int')
    aa4 = ig2.allocate('random_int')
    aa5 = ig2.allocate('random_int')
    aa6 = ig2.allocate('random_int')  
    
    w2 = BlockDataContainer(*[aa1, aa2, aa3, \
                              aa2, aa4, aa5, \
                              aa3, aa5, aa6])     
    
    tmp1 = E2.direct(u2)
    tmp2 = E2.adjoint(w2)
    c1 = (tmp1 * w2).sum()
    d1 = (u2 * tmp2).sum() 
    
    print(c1, d1)
    
    
    
    
    
    
    
   