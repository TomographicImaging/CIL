#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 19:30:51 2019

@author: evangelos
"""

from ccpi.optimisation.operators import Operator


class Identity(Operator):
    
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