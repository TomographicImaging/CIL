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
    
    r'''MaskOperator:  D: X -> X,  takes in a DataContainer or subclass 
    thereof, mask, with True or 1.0 representing a value to be
    kept and False or 0.0 a value to be lost/set to zero. Maps an element of 
    :math:`x\in X` onto the element :math:`y \in X,  y = mask*x`, 
    where * denotes elementwise multiplication.
                       
        :param mask: DataContainer of datatype bool or with 1/0 elements
                       
     '''
    
    def __init__(self, mask):
        # Special case of DiagonalOperator which is the superclass of
        # MaskOperator, so simply instanciate a DiagonalOperator with mask.
        super(MaskOperator, self).__init__(mask)
        self.mask = self.diagonal
        
        

