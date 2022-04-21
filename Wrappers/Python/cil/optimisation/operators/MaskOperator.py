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

from cil.optimisation.operators import DiagonalOperator

class MaskOperator(DiagonalOperator):
    
    r'''MaskOperator
    D: X -> X
    Maps an element of 
    :math:`x\in X` onto the element :math:`y \in X,  y = mask*x`, 
    where * denotes elementwise multiplication.

    Parameters
    ----------
    mask : DataContainer
        Boolean array with the same dimensions as the data to be operated on
    domain_geometry : ImageGeometry
        Specifies the geometry of the operator domain. If 'None' will use the mask geometry size and spacing as float32
     '''
    
    def __init__(self, mask, domain_geometry=None):

        #if domain_geometry is not specified assume float32 for domain_geometry data type
        if domain_geometry is None:
            domain_geometry = mask.geometry.copy()
            domain_geometry.dtype = np.float32

        super(MaskOperator, self).__init__(mask, domain_geometry)
        self.mask = self.diagonal
        
        

