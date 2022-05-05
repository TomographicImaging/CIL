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
from cil.utilities import dataexample
from cil.utilities import noise
import unittest
from cil.utilities.quality_measures import mse, mae, psnr
from packaging import version
if version.parse(np.version.version) >= version.parse("1.13"):
    try:
        from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
        has_skimage = True
    except ImportError as ie:
            has_skimage = False
else:
    has_skimage = False
    

class TestQualityMeasures(unittest.TestCase):
    
    def setUp(self):
        if has_skimage:

            id_coins = dataexample.CAMERA.get()

            id_coins_noisy = noise.gaussian(id_coins, var=0.05, seed=10)
            
            ig = id_coins.geometry.copy()
            dc1 = ig.allocate('random')
            dc2 = ig.allocate('random')

            self.dc1 = dc1
            self.dc2 = dc2
            self.id_coins = id_coins
            self.id_coins_noisy = id_coins_noisy


    @unittest.skipIf((not has_skimage) or version.parse(np.version.version) < version.parse("1.13"), "Skip test with numpy {} < 1.13 or has_skimage {}".format(np.version.version, has_skimage))
    def test_mse1(self):
        res1 = mse(self.id_coins, self.id_coins_noisy)
        res2 = mean_squared_error(self.id_coins.as_array(), self.id_coins_noisy.as_array())
        np.testing.assert_almost_equal(res1, res2, decimal=5)
        

    @unittest.skipIf((not has_skimage) or version.parse(np.version.version) < version.parse("1.13"), "Skip test with numpy {} < 1.13 or has_skimage {}".format(np.version.version, has_skimage))
    def test_mse2(self):
        res1 = mse(self.dc1, self.dc2)
        res2 = mean_squared_error(self.dc1.as_array(), self.dc2.as_array())
        np.testing.assert_almost_equal(res1, res2, decimal=5)
    

    @unittest.skipIf((not has_skimage) or version.parse(np.version.version) < version.parse("1.13"), "Skip test with numpy {} < 1.13 or has_skimage {}".format(np.version.version, has_skimage))
    def test_psnr1(self):
        res1 = psnr(self.id_coins, self.id_coins_noisy, data_range = self.dc1.max())
        res2 = peak_signal_noise_ratio(self.id_coins.as_array(), self.id_coins_noisy.as_array())
        np.testing.assert_almost_equal(res1, res2, decimal=3)
        


    @unittest.skipIf((not has_skimage) or version.parse(np.version.version) < version.parse("1.13"), "Skip test with numpy {} < 1.13 or has_skimage {}".format(np.version.version, has_skimage))
    def test_psnr2(self):
        res1 = psnr(self.dc1, self.dc2, data_range = self.dc1.max())
        res2 = peak_signal_noise_ratio(self.dc1.as_array(), self.dc2.as_array())
        np.testing.assert_almost_equal(res1, res2, decimal=3)
        

