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

from ccpi.optimisation.operators import LinearOperator

class ZeroOperator(LinearOperator):
    
    r'''ZeroOperator:  O: X -> Y,  maps any element of x\in X into the zero element in Y
                       O(x) = O_{Y}
                       
                       X : gm_domain
                       Y : gm_range ( Default: Y = X )
                       
                       
                       Note: 
                       .. math::

                              O^{*}: Y^{*} -> X^{*} (Adjoint)
                       
                              < O(x), y > = < x, O^{*}(y) >
                       
     '''
    
    def __init__(self, domain_gm, range_gm=None):
        
        super(ZeroOperator, self).__init__()          
        
        self.domain_gm = domain_gm
        self.range_gm = range_gm  
        
        if self.range_gm is None:
            self.range_gm = self.domain_gm
                                          
        
    def direct(self, x, out=None):
        
        '''Returns O(x)'''
        
        
        if out is None:
            return self.range_gm.allocate()
        else:
            out.fill(self.range_gm.allocate())
    
    def adjoint(self, x, out=None):
        
        '''Returns O^{*}(y)'''        
        
        if out is None:
            return self.domain_gm.allocate()
        else:
            out.fill(self.domain_gm.allocate())
        
    def calculate_norm(self, **kwargs):
        
        '''Evaluates operator norm of ZeroOperator'''
        
        return 0.
    
