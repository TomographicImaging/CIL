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
from ccpi.optimisation.operators import Operator

class DiagonalOperator(Operator):
    
    r'''DiagonalOperator:  D: X -> X,  takes in a DataContainer or subclass 
    thereof, diag, representing elements on the diagonal of a diagonal 
    operator. Maps an element of :math:`x\in X` onto the element 
    :math:`y \in X,  y = diag*x`, where * denotes elementwise multiplication.
                       
        :param diagonal: DataContainer with diagonal elements
                       
     '''
    
    def __init__(self, diagonal):
        super(DiagonalOperator, self).__init__(domain_geometry=diagonal.geometry, 
                                           range_geometry=diagonal.geometry)
        self.diagonal = diagonal

        
    def direct(self,x,out=None):
        
        '''Returns D(x)'''
        
        if out is None:
            return self.diagonal * x
        else:
            out.fill(self.diagonal * x)
    
    def adjoint(self,x, out=None):
        
        '''Returns D^{*}(y)'''        
        
        if out is None:
            return self.diagonal * x
        else:
            out.fill(self.diagonal * x)
        
    def calculate_norm(self, **kwargs):
        
        '''Evaluates operator norm of DiagonalOperator'''
        
        return self.diagonal.as_array().max()

if __name__ == '__main__':
    
    from ccpi.framework import ImageGeometry

    M = 3
    ig = ImageGeometry(M, M)
    x = ig.allocate('random_int')
    diag = ig.allocate('random_int')
    
    # Print what each ImageData is
    print(x.as_array())
    print(diag.as_array())
    
    # Set up example DiagonalOperator
    D = DiagonalOperator(diag)
    
    # Apply direct and check whether result equals diag*x as expected.
    z = D.direct(x)
    print(z.as_array())
    print((diag*x).as_array())
    
    # Apply adjoint and check whether results equals diag*(diag*x) as expected.
    y = D.adjoint(z)
    print(y.as_array())
    print((diag*(diag*x)).as_array())
    
    
    