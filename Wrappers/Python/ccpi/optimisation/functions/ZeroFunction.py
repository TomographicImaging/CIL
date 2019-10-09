# -*- coding: utf-8 -*-
# Copyright 2019 Science Technology Facilities Council
# Copyright 2019 University of Manchester
#
# This work is part of the Core Imaging Library developed by Science Technology
# Facilities Council and University of Manchester
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from ccpi.optimisation.functions import Function

class ZeroFunction(Function):
    
    r'''ZeroFunction: .. math:: f(x) = 0,         
        
        Maps evely element x\in X to zero
    '''
    
    def __init__(self):
        super(ZeroFunction, self).__init__()
              
    def __call__(self,x):
        
        '''Evaluates ZeroFunction at x'''
        return 0
    
    
    def gradient(self, x, out=None):
        
        '''Evaluates gradient of ZeroFunction at x'''        
        
        if out is None:
            return 0
        else:
            out *= 0
    
    def convex_conjugate(self, x):
        
        r''' Convex conjugate of ZeroFunction: support function .. math:: sup <x, x^{*}>  
            
            In fact is the  indicator function for the set = {0}
            So 0 if x=0, or inf if x neq 0  
            
        '''
        return x.maximum(0).sum()
        
    
    def proximal(self, x, tau, out=None):
        
        r'''Proximal operator of ZeroFunction at x
           
            .. math:: prox_{\tau * f}(x)
        '''
        
        if out is None:
            return x.copy()
        else:
            out.fill(x)
        
    def proximal_conjugate(self, x, tau, out = None):
        
        r'''Proximal operator of the convex conjugate of ZeroFunction at x:
           
           .. math:: prox_{\tau * f^{*}}(x)

        '''         
        
        if out is None:
            return 0
        else:
            return 0
