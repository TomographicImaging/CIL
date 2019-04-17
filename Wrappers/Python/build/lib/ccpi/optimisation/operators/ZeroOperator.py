#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 19:25:53 2019

@author: evangelos
"""

import numpy as np
from ccpi.framework import ImageData
from ccpi.optimisation.operators import Operator

class ZeroOp(Operator):
    
    def __init__(self, gm_domain, gm_range):
        self.gm_domain = gm_domain
        self.gm_range = gm_range
        super(ZeroOp, self).__init__()
        
    def direct(self,x,out=None):
        if out is None:
            return ImageData(np.zeros(self.gm_range))
        else:
            return ImageData(np.zeros(self.gm_range))
    
    def adjoint(self,x, out=None):
        if out is None:
            return ImageData(np.zeros(self.gm_domain))
        else:
            return ImageData(np.zeros(self.gm_domain))
        
    def norm(self):
        return 0
    
    def domain_dim(self):       
        return self.gm_domain  
        
    def range_dim(self):
        return self.gm_range