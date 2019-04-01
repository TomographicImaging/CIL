# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library developed by
#   Visual Analytics and Imaging System Group of the Science Technology
#   Facilities Council, STFC

#   Copyright 2018-2019 Evangelos Papoutsellis and Edoardo Pasca

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""
Created on Wed Mar  6 19:42:34 2019

@author: evangelos
"""

import numpy as np
#from ccpi.optimisation.funcs import Function
from ccpi.optimisation.functions import Function
from ccpi.framework import DataContainer, ImageData, ImageGeometry 


############################   L1NORM FUNCTIONS   #############################
class SimpleL1Norm(Function):
    
    def __init__(self, alpha=1):
        
        super(SimpleL1Norm, self).__init__()         
        self.alpha = alpha
        
    def __call__(self, x):
        return self.alpha * x.abs().sum()
    
    def gradient(self,x):
        return ValueError('Not Differentiable')
            
    def convex_conjugate(self,x):
        return 0
    
    def proximal(self, x, tau):
        ''' Soft Threshold'''
        return x.sign() * (x.abs() - tau * self.alpha).maximum(0)
        
    def proximal_conjugate(self, x, tau):
        return x.divide((x.abs()/self.alpha).maximum(1.0))
    
class L1Norm(SimpleL1Norm):
    
    def __init__(self, alpha=1, **kwargs):
        
        super(L1Norm, self).__init__()         
        self.alpha = alpha 
        self.b = kwargs.get('b',None)
        
    def __call__(self, x):
        
        if self.b is None:
            return SimpleL1Norm.__call__(self, x)
        else:
            return SimpleL1Norm.__call__(self, x - self.b)
            
    def gradient(self, x):
        return ValueError('Not Differentiable')
            
    def convex_conjugate(self,x):
        if self.b is None:
            return SimpleL1Norm.convex_conjugate(self, x)
        else:
            return SimpleL1Norm.convex_conjugate(self, x) + (self.b * x).sum()
    
    def proximal(self, x, tau):
        
        if self.b is None:
            return SimpleL1Norm.proximal(self, x, tau)
        else:
            return self.b + SimpleL1Norm.proximal(self, x - self.b , tau)
        
    def proximal_conjugate(self, x, tau):
        
        if self.b is None:
            return SimpleL1Norm.proximal_conjugate(self, x, tau)
        else:
            return SimpleL1Norm.proximal_conjugate(self, x - tau*self.b, tau)
                        