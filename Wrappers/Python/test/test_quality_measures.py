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

import unittest
from utils import initialise_tests
import numpy as np
from cil.utilities import dataexample
from cil.utilities import noise
from cil.utilities.quality_measures import mse, mae, psnr
from packaging import version
from cil.processors import Slicer
from testclass import CCPiTestClass
if version.parse(np.version.version) >= version.parse("1.13"):
    try:
        from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
        has_skimage = True
    except ImportError as ie:
            has_skimage = False
else:
    has_skimage = False
    
initialise_tests()

class TestQualityMeasures(CCPiTestClass):
    
    def setUp(self):
        if has_skimage:

            id_coins = dataexample.CAMERA.get()

            id_coins_noisy = noise.gaussian(id_coins, var=0.05, seed=10)
            
            ig = id_coins.geometry.copy()
            dc1 = ig.allocate('random')
            dc2 = ig.allocate('random')

            self.dc1 = dc1
            self.dc2 = dc2
            
            self.mask=ig.allocate(0)  
            self.mask.array[:50,:50]=1
            
            self.bool_mask=self.mask.array.astype('bool')
            
            self.id_coins = id_coins
            self.id_coins_noisy = id_coins_noisy

            roi = {'horizontal_x':(0,50,1),'horizontal_y':(0,50,1)}
            processor = Slicer(roi)
            processor.set_input(id_coins)
            self.id_coins_sliced= processor.get_output()
            processor = Slicer(roi)
            processor.set_input(id_coins_noisy )
            self.id_coins_noisy_sliced= processor.get_output()

    @unittest.skipIf((not has_skimage) , "Skip test with has_skimage {}".format( has_skimage))
    def test_mse1(self):
        res1 = mse(self.id_coins, self.id_coins_noisy)
        res2 = mean_squared_error(self.id_coins.as_array(), self.id_coins_noisy.as_array())
        np.testing.assert_almost_equal(res1, res2, decimal=5)
        

    @unittest.skipIf((not has_skimage), "Skip test with  has_skimage {}".format( has_skimage))
    def test_mse2(self):
        res1 = mse(self.dc1, self.dc2)
        res2 = mean_squared_error(self.dc1.as_array(), self.dc2.as_array())
        np.testing.assert_almost_equal(res1, res2, decimal=5)
    
    @unittest.skipIf((not has_skimage), "Skip test with  has_skimage {}".format( has_skimage))
    def test_psnr1(self):
        res1 = psnr(self.id_coins, self.id_coins_noisy, data_range = self.dc1.max())
        res2 = peak_signal_noise_ratio(self.id_coins.as_array(), self.id_coins_noisy.as_array())
        np.testing.assert_almost_equal(res1, res2, decimal=3)
        
    @unittest.skipIf((not has_skimage), "Skip test with  has_skimage {}".format( has_skimage))
    def test_psnr2_default_data_range(self):
        res1 = psnr(self.id_coins, self.id_coins_noisy)
        res2 = peak_signal_noise_ratio(self.id_coins.as_array(), self.id_coins_noisy.as_array())
        np.testing.assert_almost_equal(res1, res2, decimal=3)
        
        


    @unittest.skipIf((not has_skimage), "Skip test with  has_skimage {}".format( has_skimage))
    def test_psnr2(self):
        res1 = psnr(self.dc1, self.dc2, data_range = self.dc1.max())
        res2 = peak_signal_noise_ratio(self.dc1.as_array(), self.dc2.as_array())
        np.testing.assert_almost_equal(res1, res2, decimal=3)
    
    def test_mse_mask(self):
        res1 = mse(self.id_coins_sliced, self.id_coins_noisy_sliced)
        res2 = mse(self.id_coins, self.id_coins_noisy, mask=self.mask.array)
        np.testing.assert_almost_equal(res1, res2, decimal=3)

    def test_mse_bool_mask(self):
        res1 = mse(self.id_coins_sliced, self.id_coins_noisy_sliced)
        res2 = mse(self.id_coins, self.id_coins_noisy, mask=self.bool_mask)
        np.testing.assert_almost_equal(res1, res2, decimal=3)
    
    def test_mse_data_container_mask(self):
        res1 = mse(self.id_coins_sliced, self.id_coins_noisy_sliced)
        res2 = mse(self.id_coins, self.id_coins_noisy, mask=self.mask)
        np.testing.assert_almost_equal(res1, res2, decimal=3)
        
    def test_psnr_mask(self):
        res1 = psnr(self.id_coins_sliced, self.id_coins_noisy_sliced)
        res2 = psnr(self.id_coins, self.id_coins_noisy, mask=self.mask)
        np.testing.assert_almost_equal(res1, res2, decimal=3)
        
    def test_mae_mask(self):
        res1 = mae(self.id_coins_sliced, self.id_coins_noisy_sliced)
        res2 = mae(self.id_coins, self.id_coins_noisy, mask=self.mask)
        np.testing.assert_almost_equal(res1, res2, decimal=3)
        
    def test_infinite_psnr(self):
        self.assertEqual(psnr(self.id_coins_sliced, self.id_coins_sliced), np.inf)
            
        
        
