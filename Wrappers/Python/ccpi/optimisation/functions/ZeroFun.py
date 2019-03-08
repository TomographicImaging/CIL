#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 19:44:10 2019

@author: evangelos
"""

import numpy as np
from ccpi.optimisation.funcs import Function
from ccpi.framework import DataContainer, ImageData
from ccpi.framework import BlockDataContainer 

class ZeroFun(Function):
    
    def __init__(self):
        super(ZeroFun, self).__init__()
              
    def __call__(self,x):
        return 0
    
    def convex_conjugate(self, x):
        ''' This is the support function sup <x, x^{*}>  which in fact is the 
        indicator function for the set = {0}
        So 0 if x=0, or inf if x neq 0                
        '''
        
        if x.shape[0]==1:
            return x.maximum(0).sum()
        else:
            if isinstance(x, CompositeDataContainer):
                return x.get_item(0).maximum(0).sum() + x.get_item(1).maximum(0).sum()
            else:
                return x.maximum(0).sum() + x.maximum(0).sum()
    
    def proximal(self,x,tau):
        return x.copy()
        
    def proximal_conjugate(self, x, tau):
        return 0