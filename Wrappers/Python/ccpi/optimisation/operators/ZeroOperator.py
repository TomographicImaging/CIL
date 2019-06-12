#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 19:25:53 2019

@author: evangelos
"""

import numpy as np
from ccpi.framework import ImageData
from ccpi.optimisation.operators import LinearOperator

class ZeroOperator(LinearOperator):
    
    def __init__(self, gm_domain, gm_range=None):
        
        super(ZeroOperator, self).__init__()             

        self.gm_domain = gm_domain
        self.gm_range = gm_range  
        if self.gm_range is None:
            self.gm_range = self.gm_domain
                   
        
    def direct(self,x,out=None):
        if out is None:
            return self.gm_range.allocate()
        else:
            out.fill(self.gm_range.allocate())
    
    def adjoint(self,x, out=None):
        if out is None:
            return self.gm_domain.allocate()
        else:
            out.fill(self.gm_domain.allocate())
        
    def calculate_norm(self, **kwargs):
        return 0.
    
    def domain_geometry(self):       
        return self.gm_domain  
        
    def range_geometry(self):
        return self.gm_range