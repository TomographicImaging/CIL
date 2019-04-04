#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 19:30:51 2019

@author: evangelos
"""

from ccpi.optimisation.operators import LinearOperator
import scipy.sparse as sp
import numpy as np
from ccpi.framework import ImageData


class Identity(LinearOperator):
    
    def __init__(self, gm_domain, gm_range=None):

        self.gm_domain = gm_domain
        self.gm_range = gm_range  
        if self.gm_range is None:
            self.gm_range = self.gm_domain
        
        super(Identity, self).__init__()
        
    def direct(self,x,out=None):
        if out is None:
            return x.copy()
        else:
            out.fill(x)
    
    def adjoint(self,x, out=None):
        if out is None:
            return x.copy()
        else:
            out.fill(x)
        
    def norm(self):
        return 1.0
        
    def domain_geometry(self):       
        return self.gm_domain
        
    def range_geometry(self):
        return self.gm_range
    
    def matrix(self):
        
        return sp.eye(np.prod(self.gm_domain.shape))
    
    def sum_abs_row(self):
        
        return self.gm_domain.allocate(1)#ImageData(np.array(np.reshape(abs(self.matrix()).sum(axis=0), self.gm_domain.shape, 'F')))
 
    def sum_abs_col(self):
        
        return self.gm_domain.allocate(1)#ImageData(np.array(np.reshape(abs(self.matrix()).sum(axis=1), self.gm_domain.shape, 'F')))
            
    
if __name__ == '__main__':
    
    from ccpi.framework import ImageGeometry

    M, N = 2, 3
    ig = ImageGeometry(M, N)
    arr = ig.allocate('random_int')
    
    Id = Identity(ig)
    d = Id.matrix()
    print(d.toarray())
    
    d1 = Id.sum_abs_col()
    print(d1.as_array())
    
    

            
    
    