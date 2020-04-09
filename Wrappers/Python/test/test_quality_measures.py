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
import unittest
import warnings
from ccpi.utilities.quality_measures import mse, mae, psnr
from packaging import version
if version.parse(np.version.version) >= version.parse("1.13"):
    try:
        from skimage import data, io, filters
        from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
        has_skimage = True
    except ImportError as ie:
            has_skimage = False
else:
    has_skimage = False

class CCPiTestClass(unittest.TestCase):
        
    def assertNumpyArrayEqual(self, first, second):
        res = True
        try:
            numpy.testing.assert_array_equal(first, second)
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

    def assertNumpyArrayAlmostEqual(self, first, second, decimal=6):
        res = True
        try:
            numpy.testing.assert_array_almost_equal(first, second, decimal)
        except AssertionError as err:
            res = False
            print(err)
            print("expected " , second)
            print("actual " , first)

        self.assertTrue(res)

class TestQualityMeasures(CCPiTestClass):
    
    def setUp(self):
        print ("SETUP", np.version.version)
        if has_skimage:

            id_coins = TestData().load(TestData.CAMERA)

            id_coins_noisy = TestData.random_noise(id_coins, mode='gaussian', var = 0.05, seed=10)

            ig = id_coins.geometry.copy()
            dc1 = ig.allocate('random')
            dc2 = ig.allocate('random')

            self.dc1 = dc1
            self.dc2 = dc2
            self.id_coins = id_coins
            self.id_coins_noisy = id_coins_noisy

    @unittest.skipIf(version.parse(np.version.version) < version.parse("1.13"), "Skip test with numpy < 1.13")
    def test_mse1(self):
        if has_skimage:
            #%%  Check Mean Squared error for random image and images
            res1 = mse(self.id_coins, self.id_coins_noisy)
            res2 = mean_squared_error(self.id_coins.as_array(), self.id_coins_noisy.as_array())
            print('Check MSE for CAMERA image gaussian noise')
            np.testing.assert_almost_equal(res1, res2, decimal=5)
        else:
            self.skipTest("scikit0-image not present ... skipping")
    @unittest.skipIf(version.parse(np.version.version) < version.parse("1.13"), "Skip test with numpy < 1.13")
    def test_mse2(self):
        if has_skimage:
            #%%  Check Mean Squared error for random image and images

            res1 = mse(self.dc1, self.dc2)
            res2 = mean_squared_error(self.dc1.as_array(), self.dc2.as_array())
            print('Check MSE for random ImageData')
            np.testing.assert_almost_equal(res1, res2, decimal=5)

        else:
            self.skipTest("scikit0-image not present ... skipping")
    @unittest.skipIf(version.parse(np.version.version) < version.parse("1.13"), "Skip test with numpy < 1.13")
    def test_psnr1(self):
        if has_skimage:

            res1 = psnr(self.id_coins, self.id_coins_noisy, data_range = self.dc1.max())
            res2 = peak_signal_noise_ratio(self.id_coins.as_array(), self.id_coins_noisy.as_array())
            print('Check PSNR for CAMERA image gaussian noise')
            np.testing.assert_almost_equal(res1, res2, decimal=3)
        else:
            self.skipTest("scikit0-image not present ... skipping")
    @unittest.skipIf(version.parse(np.version.version) < version.parse("1.13"), "Skip test with numpy < 1.13")
    def test_psnr2(self):
        if has_skimage:

            res1 = psnr(self.dc1, self.dc2, data_range = self.dc1.max())
            res2 = peak_signal_noise_ratio(self.dc1.as_array(), self.dc2.as_array())
            print('Check PSNR for random ImageData')
            np.testing.assert_almost_equal(res1, res2, decimal=3)

        else:
            self.skipTest("scikit0-image not present ... skipping")


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
