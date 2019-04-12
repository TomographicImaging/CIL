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

import numpy as np
#from ccpi.optimisation.funcs import Function
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
        
        if x.shape[0]==1:
            return x.maximum(0).sum()
        else:
            if isinstance(x, BlockDataContainer):
                return x.get_item(0).maximum(0).sum() + x.get_item(1).maximum(0).sum()
            else:
                return x.maximum(0).sum() + x.maximum(0).sum()
    
    def proximal(self, x, tau, out=None):
        if out is None:
            return x.copy()
        else:
            out.fill(x)
        
    def proximal_conjugate(self, x, tau, out = None):
        if out is None:
            return 0
        else:
            return 0

    def domain_geometry(self):
        pass
    def range_geometry(self):
        pass