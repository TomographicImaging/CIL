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

from ccpi.optimisation.functions import L2NormSquared, L1Norm
import numpy as np


def mse(dc1, dc2):    
    
    ''' Returns the Mean Squared error of two DataContainers
    '''
    
    diff = dc1 - dc2    
    return L2NormSquared().__call__(diff)/dc1.size


def mae(dc1, dc2):
    
    ''' Returns the Mean Absolute error of two DataContainers
    '''    
    
    diff = dc1 - dc2  
    return L1Norm().__call__(diff)/dc1.size()

def psnr(ground_truth, corrupted, data_range = 255):

    ''' Returns the Peak signal to noise ratio
    '''   
    
    tmp_mse = mse(ground_truth, corrupted)
    if tmp_mse == 0:
        return 1e5
    return 10 * np.log10((data_range ** 2) / tmp_mse)




