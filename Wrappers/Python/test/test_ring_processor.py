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
        

from cil.processors import RingRemover
from cil.framework import ImageData, ImageGeometry, AcquisitionGeometry
# from ccpi.astra.operators import AstraProjectorSimple

import tomophantom
from tomophantom import TomoP2D
import os
import numpy as np
import unittest

class TestL1NormRR(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
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
        print ("L1Norm ", error)
        np.testing.assert_almost_equal(error, 83.20592, 4)
