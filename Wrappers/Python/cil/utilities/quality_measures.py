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
    ''' Calculates the mean squared error of two images

    Parameters
    ----------
    dc1: `DataContainer`
        One image to be compared
    dc2: `DataContainer`
        Second image to be compared 
    mask: array or `DataContainer` with the same dimensions as the `dc1` and `dc2`
        The pixelwise operation only considers values where the mask is True or NonZero.

    Returns
    -------
    A number, the mean squared error of the two images
    '''
    dc1 = dc1.as_array()
    dc2 = dc2.as_array()
    
    if mask is not None:
        
        if isinstance(mask, DataContainer):
            mask = mask.as_array()
            
        mask = mask.astype('bool')
        dc1 = np.extract(mask, dc1)
        dc2 = np.extract(mask, dc2)
    return np.mean(((dc1 - dc2)**2))


def mae(dc1, dc2, mask=None):
    ''' Calculates the Mean Absolute error of two images.

    Parameters
    ----------
    dc1: `DataContainer`
        One image to be compared
    dc2: `DataContainer`
        Second image to be compared 
    mask: array or `DataContainer` with the same dimensions as the `dc1` and `dc2`
        The pixelwise operation only considers values where the mask is True or NonZero. 


    Returns
    -------
    A number with the mean absolute error between the two images. 
    '''
    dc1 = dc1.as_array()
    dc2 = dc2.as_array()

    if mask is not None:
        
        if isinstance(mask, DataContainer):
            mask = mask.as_array()
            
        mask = mask.astype('bool')
        dc1 = np.extract(mask, dc1)
        dc2 = np.extract(mask, dc2)
        
    return np.mean(np.abs((dc1-dc2)))


def psnr(ground_truth, corrupted, data_range=None, mask=None):
    ''' Calculates the Peak signal to noise ratio (PSNR) between the two images. 

    Parameters
    ----------
    ground_truth: `DataContainer`
        The reference image
    corrupted: `DataContainer`
        The image to be evaluated 
    data_range: scalar value, default=None
        PSNR scaling factor, the dynamic range of the images (i.e., the difference between the maximum the and minimum allowed values). We take the maximum value in the ground truth array.
    mask: array or `DataContainer` with the same dimensions as the `dc1` and `dc2`
        The pixelwise operation only considers values where the mask is True or NonZero.. 

    Returns
    -------
    A number, the peak signal to noise ration between the two images.
    '''
    if data_range is None:
        
        if mask is None:
            data_range = ground_truth.as_array().max()
            
        else:
            
            if isinstance(mask, DataContainer):
                mask = mask.as_array()
            data_range = np.max(ground_truth.as_array(),
                                 where=mask.astype('bool'), initial=-1e-8)

    tmp_mse = mse(ground_truth, corrupted, mask=mask)

    return 10 * np.log10((data_range ** 2) / tmp_mse)
  