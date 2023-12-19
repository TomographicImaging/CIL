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
    
    Parameters
    ----------
    dc1: `DataContainer`
        One image to be compared
    dc2: `DataContainer`
        Second image to be compared 
    mask: array or `DataContainer` of Boolean values or 0's and 1's with the same dimensions as the `dc1` and `dc2`
        Region of interest for the calculation. 
    '''  

    if mask is None:
        diff = dc1 - dc2    
        return L2NormSquared().__call__(diff)/dc1.size
    else:
        if isinstance(mask, DataContainer):
            mask = mask.as_array()
        return np.mean(((dc1.as_array() - dc2.as_array())**2), where=mask.astype('bool'))


def mae(dc1, dc2, mask=None):
    
    ''' Returns the Mean Absolute error of two DataContainers
    
        
    Parameters
    ----------
    dc1: `DataContainer`
        One image to be compared
    dc2: `DataContainer`
        Second image to be compared 
    mask: array or `DataContainer` of Boolean values or 0's and 1's with the same dimensions as the `dc1` and `dc2`
        Region of interest for the calculation. 
        
        
    '''    
    if mask is None:
        diff = dc1 - dc2  
        return L1Norm().__call__(diff)/dc1.size
    else:
        if isinstance(mask, DataContainer):
            mask=mask.as_array()
    return np.mean(np.abs((dc1.as_array()-dc2.as_array())), where=mask.astype('bool'))

def psnr(ground_truth, corrupted, data_range=None, mask=None):

    ''' Returns the Peak signal to noise ratio
    
    Parameters
    ----------
    ground_truth: `DataContainer`
        The reference image
    corrupted: `DataContainer`
        The image to be evaluated 
    data_range: scalar value, default=None
        PSNR scaling factor, the dynamic range of the images (i.e., the difference between the maximum the and minimum allowed values). To match with scikit-image the default is ground_truth.array.max()
    mask: array or `DataContainer` of Boolean values or 0's and 1's with the same dimensions as the `ground_truth` and `corrupted`
        Region of interest for the calculation. 
    '''  
    if data_range is None:
        if mask is None:
            data_range=ground_truth.array.max()
        else:
            if isinstance(mask, DataContainer):
                mask=mask.as_array()
            data_range=np.amax(ground_truth.as_array()[mask.astype('bool')])
    
    tmp_mse = mse(ground_truth, corrupted, mask=mask) 

    if tmp_mse == 0:
        return 1e5
    return 10 * np.log10((data_range ** 2) / tmp_mse)







