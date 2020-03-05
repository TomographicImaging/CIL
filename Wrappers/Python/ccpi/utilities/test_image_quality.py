"""
Created on Wed Mar  4 09:57:30 2020

@author: evangelos
"""

from ccpi.framework import ImageGeometry
from ccpi.optimisation.functions import L2NormSquared, L1Norm
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float
from ccpi.framework import TestData
import os
import sys
from skimage import data, io, filters
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import unittest
import warnings


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

def num_elem_geom(dc):
    
    return np.prod(dc.shape)

def mse(dc1, dc2):    
    diff = dc1 - dc2    
    return L2NormSquared().__call__(diff)/num_elem_geom(dc1)

res1 = mse(dc1, dc2)
res2 = mean_squared_error(dc1.as_array(), dc2.as_array())
print('Check MSE for random ImageData')
np.testing.assert_almost_equal(res1, res2, decimal=5)


res1 = mse(id_coins, id_coins_noisy)
res2 = mean_squared_error(id_coins.as_array(), id_coins_noisy.as_array())
print('Check MSE for Coins image gaussian noise')
np.testing.assert_almost_equal(res1, res2, decimal=5)


#%% Mean absolute error

def mae(dc1, dc2):
    diff = dc1 - dc2  
    return L1Norm().__call__(diff)/num_elem_geom(dc1)

#%% check PSNR 
    
def psnr(ground_truth, corrupted, data_range = 255):

    tmp_mse = mse(ground_truth, corrupted)
    if tmp_mse == 0:
        return 1e5
    return 10 * np.log10((data_range ** 2) / tmp_mse)


res1 = psnr(dc1, dc2, data_range = dc1.as_array().max())
res2 = peak_signal_noise_ratio(dc1.as_array(), dc2.as_array())
print('Check PSNR for random ImageData')
np.testing.assert_almost_equal(res1, res2, decimal=3)

res1 = psnr(id_coins, id_coins_noisy, data_range = dc1.as_array().max())
res2 = peak_signal_noise_ratio(id_coins.as_array(), id_coins_noisy.as_array())
print('Check PSNR for Coins image gaussian noise')
np.testing.assert_almost_equal(res1, res2, decimal=3)


#%% SSIM

print('SSIM is not ready')

def ssim(dc1, dc2, \
         win_size=None, gradient=False, data_range=None, \
         multichannel=False, gaussian_weights=True, \
         full=True, **kwargs):
    
    return structural_similarity(dc1.as_array(), dc2.as_array(),\
                                 win_size=win_size, gradient=gradient, data_range=data_range, \
                                 multichannel=multichannel, gaussian_weights=gaussian_weights, \
                                 full=full, **kwargs)

res1 = ssim(id_coins, id_coins_noisy, data_range = 2)

print(res1[0])

res2 = ssim(id_coins, id_coins_noisy)

print(res2[0])

from skimage.util.dtype import dtype_range

z = dtype_range[id_coins.as_array().dtype.type]


#%%