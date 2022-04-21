# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library (CIL) developed by CCPi 
#   (Collaborative Computational Project in Tomographic Imaging), with 
#   substantial contributions by UKRI-STFC and University of Manchester.

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy as np
from cil.framework import ImageData
from cil.optimisation.operators import LinearOperator

class DiagonalOperator(LinearOperator):

    r'''DiagonalOperator
    D: X -> X
    Maps an element of :math:`x\in X` onto the element 
    :math:`y \in X,  y = diag*x`, where * denotes elementwise multiplication.

    In matrix-vector interpretation, if x is a vector of length N, then diagonal is 
    also a vector of length N, and D  will be an NxN diagonal matrix with diag 
    on its diagonal and zeros everywhere else.

    Parameters
    ----------
    diagonal : DataContainer
        DataContainer with the same dimensions as the data to be operated on
    domain_geometry : ImageGeometry
        Specifies the geometry of the operator domain. If 'None' will use the diagonal geometry directly.
     '''

    
    def __init__(self, diagonal, domain_geometry=None):

        if domain_geometry is None:
            domain_geometry = diagonal.geometry.copy()

        super(DiagonalOperator, self).__init__(domain_geometry=domain_geometry, 
                                    range_geometry=domain_geometry)
        self.diagonal = diagonal

        
    def direct(self,x,out=None):
        
        '''Returns D(x)'''
        
        if out is None:
            return self.diagonal * x
        else:
            self.diagonal.multiply(x,out=out)
    

    def adjoint(self,x, out=None):
        
        '''Returns D^{*}(y), which is identical to direct, so use direct.'''        
        
        return self.direct(x, out=out)

  
    def calculate_norm(self, **kwargs):
        
        '''Evaluates operator norm of DiagonalOperator'''
        
        return self.diagonal.max()
