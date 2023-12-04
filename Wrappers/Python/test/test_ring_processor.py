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
from cil.processors import RingRemover, TransmissionAbsorptionConverter, Slicer
from cil.framework import ImageData, ImageGeometry, AcquisitionGeometry
from cil.utilities import dataexample

import os
import numpy as np
from utils import has_tomophantom, initialise_tests
from testclass import CCPiTestClass

initialise_tests()

if has_tomophantom:
    import tomophantom
    from tomophantom import TomoP2D


class TestL1NormRR(unittest.TestCase):
    def setUp(self):
        pass


    def tearDown(self):
        pass


    @unittest.skipUnless(has_tomophantom, "Tomophantom not installed")
    def test_L1Norm_2D(self):
        model = 12 # select a model number from the library
        N = 400 # set dimension of the phantom
        path = os.path.dirname(tomophantom.__file__)
        path_library2D = os.path.join(path, "Phantom2DLibrary.dat")

        phantom_2D = TomoP2D.Model(model, N, path_library2D)    
        # data = ImageData(phantom_2D)
        ig = ImageGeometry(voxel_num_x=N, voxel_num_y=N)
        data = ig.allocate(None)
        data.fill(phantom_2D)

        # Create acquisition data and geometry
        detectors = N
        angles = np.linspace(0, 180, 120, dtype=np.float32)

        ag = AcquisitionGeometry.create_Parallel2D()\
            .set_angles(angles, angle_unit=AcquisitionGeometry.DEGREE)\
            .set_panel(detectors)
        sin = ag.allocate(None)
        sino = TomoP2D.ModelSino(model, detectors, detectors, angles, path_library2D)
        sin.fill(sino)
        sin_stripe = sin.copy()
        #        sin_stripe = sin
        tmp = sin_stripe.as_array()
        tmp[:,::27]=tmp[:,::28]
        sin_stripe.fill(tmp)

        ring_recon = RingRemover(20, "db15", 21, info = True)(sin_stripe)

        error = (ring_recon - sin).abs().as_array().mean()
        np.testing.assert_almost_equal(error, 83.20592, 4)
        
class TestRingRemover(CCPiTestClass):
    def add_bad_pixels(self, data, number_of_columns, number_of_hot_pix, seed):

        data_corrupted = data.copy()

        # get intensity range
        low = np.amin(data.as_array())
        high = np.amax(data.as_array())

        # we seed random number generator for repeatability
        rng = np.random.RandomState(seed=seed) 
        # indices of bad columns
        columns = rng.randint(0, data.shape[1], size=number_of_columns)
        # indices of hot pixels
        pix_row = rng.randint(0, data.shape[0], size=number_of_hot_pix)
        pix_col = rng.randint(0, data.shape[1], size=number_of_hot_pix)
        # values in hot pixels
        pixel_values = rng.uniform(low=low, high=high, size=number_of_hot_pix)

        for i in range(number_of_columns):
            col_pattern = rng.uniform(low=low, high=high, size=data.shape[0])
            data_corrupted.as_array()[:, columns[i]] = data.as_array()[:, columns[i]]+col_pattern

        for i in range(number_of_hot_pix):
            data_corrupted.as_array()[pix_row[i], pix_col[i]] = pixel_values[i]

        return data_corrupted
    
    def test_ring_remover(self):
        
        data = dataexample.SIMULATED_PARALLEL_BEAM_DATA.get().get_slice(vertical=80)

        data_corrupted = self.add_bad_pixels(data, 1, 1, 1312)
        data_corrupted = TransmissionAbsorptionConverter()(data_corrupted)
        r = RingRemover(8,'db20', 1.5)
        r.set_input(data_corrupted)
        data_corrected = r.get_output()

        self.assertEqual(data_corrupted.shape, data_corrected.shape)

        data_corrupted_odd = Slicer(roi={'horizontal': (1, None)})(data_corrupted)
        data_corrupted_odd = Slicer(roi={'angle': (1,None)})(data_corrupted_odd)
        r = RingRemover(8,'db20', 1.5)
        r.set_input(data_corrupted_odd)
        data_corrected_odd = r.get_output()

        self.assertEqual(data_corrupted_odd.shape, data_corrected_odd.shape)

