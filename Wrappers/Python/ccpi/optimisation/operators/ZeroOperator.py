#========================================================================
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
#
#=========================================================================

from ccpi.optimisation.operators import LinearOperator

class ZeroOperator(LinearOperator):
    
    '''

        ZeroOperator:  O: X -> Y,  maps any element of x\in X into the zero element in Y
                       O(x) = O_{Y}
                       
                       X : gm_domain
                       Y : gm_range ( Default: Y = X )
                       
                       
                       Note: O^{*}: Y^{*} -> X^{*} (Adjoint)
                       
                              < O(x), y > = < x, O^{*}(y) >
                       
                                                      
        
    '''
    
    def __init__(self, gm_domain, gm_range=None):
        
        super(ZeroOperator, self).__init__()             

        self.gm_domain = gm_domain
        self.gm_range = gm_range  
        if self.gm_range is None:
            self.gm_range = self.gm_domain
                   
        
    def direct(self,x,out=None):
        
        '''
        
            Returns O(x)
        
        '''
        
        
        if out is None:
            return self.gm_range.allocate()
        else:
            out.fill(self.gm_range.allocate())
    
    def adjoint(self,x, out=None):
        
        '''
        
            Returns O^{*}(y)
        
        '''        
        
        if out is None:
            return self.gm_domain.allocate()
        else:
            out.fill(self.gm_domain.allocate())
        
    def calculate_norm(self, **kwargs):
        
        '''
        
            Evaluates operator norm of ZeroOperator
        
        '''
        
        return 0
    
    def domain_geometry(self): 
        
        '''
        
            Returns domain_geometry of ZeroOperator
        
        '''
        
        
        return self.gm_domain  
        
    def range_geometry(self):
        
        '''
        
            Returns domain_geometry of ZeroOperator
        
        '''
        
        return self.gm_range