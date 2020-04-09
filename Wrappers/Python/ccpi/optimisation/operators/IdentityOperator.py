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
import scipy.sparse as sp
import numpy as np


class Identity(LinearOperator):
    
    '''Identity:  Id: X -> Y,  Id(x) = x\in Y
                       
                   X : gm_domain
                   Y : gm_range ( Default: Y = X )
                                                                                                                                   
    '''    
    
    
    def __init__(self, domain_geometry, range_geometry=None):

        
        if range_geometry is None:
            range_geometry = domain_geometry
        
        super(Identity, self).__init__(domain_geometry=domain_geometry,
                                       range_geometry=range_geometry)
        
    def direct(self,x,out=None):
        
        '''Returns Id(x)'''
        
        if out is None:
            return x.copy()
        else:
            out.fill(x)
    
    def adjoint(self,x, out=None):
        
        '''Returns Id(x)'''         
        
        
        if out is None:
            return x.copy()
        else:
            out.fill(x)
        
    def calculate_norm(self, **kwargs):
        
        '''Evaluates operator norm of Identity'''        
        
        return 1.0
    
    
    ###########################################################################
    ###############  For preconditioning ######################################
    ###########################################################################                    
    def matrix(self):
        
        return sp.eye(np.prod(self.gm_domain.shape))
    
    def sum_abs_row(self):
        
        return self.gm_range.allocate(1)
    
    def sum_abs_col(self):
        
        return self.gm_domain.allocate(1)
    
    
if __name__ == '__main__':
    
    from ccpi.framework import ImageGeometry

    M, N = 2, 3
    ig = ImageGeometry(M, N)
    arr = ig.allocate('random_int')
    
    Id = Identity(ig)
    d = Id.matrix()
    print(d.toarray())
    
    d1 = Id.sum_abs_col()
    print(d1.as_array())
    
    

            
    
    
