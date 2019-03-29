#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 22:50:04 2019

@author: evangelos
"""

from ccpi.optimisation.operators import Operator, LinearOperator, ScaledOperator
from ccpi.optimisation.ops import PowerMethodNonsquare
from ccpi.framework import ImageData, ImageGeometry, BlockGeometry
import numpy 
from ccpi.optimisation.operators import FiniteDiff

#%%

class Gradient(LinearOperator):
    
    def __init__(self, gm_domain, bnd_cond = 'Neumann', **kwargs):
        
        super(Gradient, self).__init__() 
                
        self.gm_domain = gm_domain # Domain of Grad Operator
        
        self.correlation = kwargs.get('correlation','Space')
        
        if self.correlation=='Space':
            if self.gm_domain.channels>1:
                self.gm_range = BlockGeometry(*[self.gm_domain for _ in range(self.gm_domain.length-1)] ) 
                self.ind = numpy.arange(1,self.gm_domain.length)
            else:    
                self.gm_range = BlockGeometry(*[self.gm_domain for _ in range(self.gm_domain.length) ] )
                self.ind = numpy.arange(self.gm_domain.length)
        elif self.correlation=='SpaceChannels':
            if self.gm_domain.channels>1:
                self.gm_range = BlockGeometry(*[self.gm_domain for _ in range(self.gm_domain.length)])
                self.ind = range(self.gm_domain.length)
            else:
                raise ValueError('No channels to correlate')
         
        self.bnd_cond = bnd_cond    
                                                         
        
    def direct(self, x, out=None):
        
        tmp = self.gm_range.allocate()
        
        
        for i in range(tmp.shape[0]):
            tmp.get_item(i).fill(FiniteDiff(self.gm_domain, direction = self.ind[i], bnd_cond = self.bnd_cond).direct(x))
        return tmp    
        
    def adjoint(self, x, out=None):
        
        tmp = self.gm_domain.allocate()
        for i in range(x.shape[0]):
            tmp+=FiniteDiff(self.gm_domain, direction = self.ind[i], bnd_cond = self.bnd_cond).adjoint(x.get_item(i))
        return tmp    
            
    
    def domain_geometry(self):
        return self.gm_domain
    
    def range_geometry(self):
        return self.gm_range
                                   
    def norm(self):

        x0 = self.gm_domain.allocate('random')
        self.s1, sall, svec = PowerMethodNonsquare(self, 10, x0)
        return self.s1
    
    def __rmul__(self, scalar):
        return ScaledOperator(self, scalar) 
    
if __name__ == '__main__':
    
    pass
