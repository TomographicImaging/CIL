#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 19:44:10 2019

@author: evangelos
"""

import numpy as np
from ccpi.optimisation.functions import Function
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
        
        return x.maximum(0).sum()
        
    
    def proximal(self,x,tau, out=None):
        if out is None:
            return x.copy()
        else:
            out.fill(x)
        
    def proximal_conjugate(self, x, tau):
        return 0