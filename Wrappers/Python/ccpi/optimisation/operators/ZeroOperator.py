# -*- coding: utf-8 -*-
#  CCP in Tomographic Imaging (CCPi) Core Imaging Library (CIL).

#   Copyright 2017 UKRI-STFC
#   Copyright 2017 University of Manchester

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from ccpi.framework import ImageData
from ccpi.optimisation.operators import LinearOperator

class ZeroOperator(LinearOperator):
    
    r'''ZeroOperator:  O: X -> Y,  maps any element of :math:`x\in X` into the zero element :math:`\in Y,  O(x) = O_{Y}`
                       
        :param gm_domain: domain of the operator 
        :param gm_range: range of the operator, default: same as domain
        
        
        Note: 
        
        .. math::

                O^{*}: Y^{*} -> X^{*} \text{(Adjoint)}
        
                < O(x), y > = < x, O^{*}(y) >
                       
     '''
    
    def __init__(self, domain_geometry, range_geometry=None):
        if range_geometry is None:
            range_geometry = domain_geometry.clone()
        super(ZeroOperator, self).__init__(domain_geometry=domain_geometry, 
                                           range_geometry=range_geometry)

                   
        
    def direct(self,x,out=None):
        
        '''Returns O(x)'''
        
        
        if out is None:
            return self.range_geometry().allocate()
        else:
            out.fill(self.range_geometry.allocate())
    
    def adjoint(self,x, out=None):
        
        '''Returns O^{*}(y)'''        
        
        if out is None:
            return self.domain_geometry().allocate()
        else:
            out.fill(self.domain_geometry().allocate())
        
    def calculate_norm(self, **kwargs):
        
        '''Evaluates operator norm of ZeroOperator'''
        
        return 0
    
    