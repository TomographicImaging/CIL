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

from ccpi.framework import ImageGeometry
from ccpi.optimisation.functions import L2NormSquared, L1Norm
import numpy as np
import matplotlib.pyplot as plt
from ccpi.framework import TestData
import os
import sys
from skimage import data, io, filters
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import unittest
import warnings
from ccpi.utilities.quality_measures import mse, mae, psnr

#%%
im_coins = data.coins()
im_coins = im_coins/np.max(im_coins)

ig = ImageGeometry(voxel_num_x = im_coins.shape[1], voxel_num_y = im_coins.shape[0])

dc1 = ig.allocate('random')
dc2 = ig.allocate('random')

id_coins = ig.allocate()
id_coins.fill(im_coins)

id_coins_noisy = TestData.random_noise(id_coins, mode='gaussian', var = 0.05, seed=10)

plt.imshow(id_coins.as_array(), cmap='gray')
plt.show()

plt.imshow(id_coins_noisy.as_array(), cmap='gray')
plt.show()


#%%  Check Mean Squared error for random image and images

res1 = mse(dc1, dc2)
res2 = mean_squared_error(dc1.as_array(), dc2.as_array())
print('Check MSE for random ImageData')
np.testing.assert_almost_equal(res1, res2, decimal=5)

res1 = mse(id_coins, id_coins_noisy)
res2 = mean_squared_error(id_coins.as_array(), id_coins_noisy.as_array())
print('Check MSE for Coins image gaussian noise')
np.testing.assert_almost_equal(res1, res2, decimal=5)

#%% check PSNR 
    
res1 = psnr(dc1, dc2, data_range = dc1.as_array().max())
res2 = peak_signal_noise_ratio(dc1.as_array(), dc2.as_array())
print('Check PSNR for random ImageData')
np.testing.assert_almost_equal(res1, res2, decimal=3)

res1 = psnr(id_coins, id_coins_noisy, data_range = dc1.as_array().max())
res2 = peak_signal_noise_ratio(id_coins.as_array(), id_coins_noisy.as_array())
print('Check PSNR for Coins image gaussian noise')
np.testing.assert_almost_equal(res1, res2, decimal=3)


##%% SSIM
#
#print('SSIM is not ready')
#
#def ssim(dc1, dc2, \
#         win_size=None, gradient=False, data_range=None, \
#         multichannel=False, gaussian_weights=True, \
#         full=True, **kwargs):
#    
#    return structural_similarity(dc1.as_array(), dc2.as_array(),\
#                                 win_size=win_size, gradient=gradient, data_range=data_range, \
#                                 multichannel=multichannel, gaussian_weights=gaussian_weights, \
#                                 full=full, **kwargs)
#
#res1 = ssim(id_coins, id_coins_noisy, data_range = 2)
#
#print(res1[0])
#
#res2 = ssim(id_coins, id_coins_noisy)
#
#print(res2[0])
#
#from skimage.util.dtype import dtype_range
#
#z = dtype_range[id_coins.as_array().dtype.type]
#
#
##%%