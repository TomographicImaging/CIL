# -*- coding: utf-8 -*-
#  Copyright 2020 United Kingdom Research and Innovation
#  Copyright 2020 The University of Manchester
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Authors:
# CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt

from cil.optimisation.functions import L2NormSquared, L1Norm
from cil.framework import DataContainer
import numpy as np



def mse(dc1, dc2, mask=None):    
    
    ''' Returns the Mean Squared error of two DataContainers
    '''
    if mask is None:
        diff = dc1 - dc2    
        return L2NormSquared().__call__(diff)/dc1.size
    
    else:
        if isinstance(DataContainer):
            mask=mask.array
        indices=np.where(bool(mask))
        return ((dc1.array[indices]-dc2.array[indices])**2).mean()


def mae(dc1, dc2, mask=None):
    
    ''' Returns the Mean Absolute error of two DataContainers
    '''    
    if mask is None:
        diff = dc1 - dc2  
        return L1Norm().__call__(diff)/dc1.size
    else:
        if isinstance(DataContainer):
            mask=mask.array
    indices=np.where(bool(mask))
    return np.abs((dc1.array[indices]-dc2.array[indices])).mean()

def psnr(ground_truth, corrupted, data_range = 255, mask=None):

    ''' Returns the Peak signal to noise ratio
    '''  
    tmp_mse = mse(ground_truth, corrupted, mask=mask) 

    if tmp_mse == 0:
        return 1e5
    return 10 * np.log10((data_range ** 2) / tmp_mse)







