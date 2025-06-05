#  Copyright 2025 United Kingdom Research and Innovation
#  Copyright 2025 The University of Manchester
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
from utils import has_tigre, has_nvidia, initialise_tests
from cil.utilities import dataexample
from unittest_parametrize import parametrize
from unittest_parametrize import ParametrizedTestCase
import numpy as np
from cil.processors import Slicer, TransmissionAbsorptionConverter

initialise_tests()

if has_tigre:
    from cil.plugins.tigre import ART, SART, SIRT, OSSART

from testclass import CCPiTestClass


class TestTigreReconstructionAlgorithms(ParametrizedTestCase,  unittest.TestCase):

    def setUp(self):
        self.ground_truth_3D = dataexample.SIMULATED_SPHERE_VOLUME.get()

        self.data_cone = dataexample.SIMULATED_CONE_BEAM_DATA.get()
        self.data_cone = TransmissionAbsorptionConverter()(self.data_cone)
        self.data_cone = Slicer(roi={'angle': (0, -1, 5)})(self.data_cone)

        self.data_fan_beam = self.data_cone.get_slice(vertical='centre')
        self.ground_truth_2D = self.ground_truth_3D.get_slice(vertical='centre')

        self.data_parallel = dataexample.SIMULATED_CONE_BEAM_DATA.get()
        self.data_parallel = TransmissionAbsorptionConverter()(self.data_parallel)
        self.data_parallel = Slicer(roi={'angle': (0, -1, 5)})(self.data_parallel)

        self.data_parallel_2D = self.data_parallel.get_slice(vertical='centre')

        self.ig3D = self.ground_truth_3D.geometry
        self.ig2D = self.ground_truth_2D.geometry

    @parametrize(
        argnames="alg",
        argvalues=[(SART,), (SIRT,), (OSSART,)],
        ids=["SART", "SIRT", "OSSART"]
    )
    @unittest.skipUnless(has_tigre and has_nvidia, "Requires TIGRE GPU")
    def test_missing_parameters_raises_error(self, alg):
        with self.assertRaises(ValueError) as context:
            alg()
        self.assertIn("You must pass", str(context.exception))

    @unittest.skipUnless(has_tigre and has_nvidia, "Requires TIGRE GPU")
    def test_sirt_initialization_success(self):
        alg = SIRT(image_geometry=self.ig2D, data=self.data_parallel_2D)
        self.assertTrue(alg.configured)
        self.assertEqual(alg.tigre_alg.blocksize, len(self.data_parallel_2D.geometry.angles))
        self.assertEqual(alg.tigre_alg.niter, 0)
        self.assertTrue(alg.tigre_alg.__dict__['noneg'])

    @unittest.skipUnless(has_tigre and has_nvidia, "Requires TIGRE GPU")
    def test_sart_initialization_success(self):
        alg = SART(image_geometry=self.ig2D, data=self.data_parallel_2D, noneg=False)
        self.assertTrue(alg.configured)
        self.assertEqual(alg.tigre_alg.blocksize, 1)
        self.assertEqual(alg.tigre_alg.niter, 0)
        self.assertFalse(alg.tigre_alg.__dict__['noneg'])
        self.assertEqual(np.sum(np.abs(alg.get_output().as_array())),0)
        self.assertEqual(np.sum(np.abs(alg.tigre_alg.__dict__['init'][0, :, :])), 0)
        
    @unittest.skipUnless(has_tigre and has_nvidia, "Requires TIGRE GPU")
    def test_ossart_initialization_success(self):
        alg = OSSART(initial=self.ig2D.allocate(1), image_geometry=self.ig2D, data=self.data_parallel_2D, blocksize=2, OrderStrategy='random')
        self.assertTrue(alg.configured)
        self.assertEqual(alg.tigre_alg.blocksize, 2)
        self.assertEqual(alg.tigre_alg.niter, 0)
        self.assertTrue(alg.tigre_alg.__dict__['noneg'])
        self.assertEqual(alg.tigre_alg.__dict__['OrderStrategy'], 'random')
        self.assertEqual(np.sum(np.abs(alg.tigre_alg.__dict__['init'][0, :, :])), np.sum(np.abs(self.ig2D.allocate(1).as_array())))

    @parametrize(
        argnames="algorithm,image_geometry,data",
        argvalues=[
            (SART, 'ig2D', 'data_parallel_2D'),
            (SIRT, 'ig2D', 'data_parallel_2D'),
            (OSSART, 'ig2D', 'data_parallel_2D'),
            (SART, 'ig2D', 'data_fan_beam'),
            (SIRT, 'ig2D', 'data_fan_beam'),
            (OSSART, 'ig2D', 'data_fan_beam'),
            (SART, 'ig3D', 'data_parallel'),
            (SIRT, 'ig3D', 'data_parallel'),
            (OSSART, 'ig3D', 'data_parallel'),
            (SART, 'ig3D', 'data_cone'),
            (SIRT, 'ig3D', 'data_cone'),
            (OSSART, 'ig3D', 'data_cone')
        ],
        ids=[
            'SART_2D_parallel', 'SIRT_2D_parallel', 'OSSART_2D_parallel',
            'SART_fan_beam', 'SIRT_fan_beam', 'OSSART_fan_beam',
            'SART_3D_parallel', 'SIRT_3D_parallel', 'OSSART_3D_parallel',
            'SART_cone', 'SIRT_cone', 'OSSART_cone'
        ]
    )
    @unittest.skipUnless(has_tigre and has_nvidia, "Requires TIGRE GPU")
    def test_update(self, algorithm, image_geometry, data):
        ig = getattr(self, image_geometry)
        gt = self.ground_truth_2D if image_geometry == 'ig2D' else self.ground_truth_3D
        dat = getattr(self, data)

        try:
            alg = algorithm(image_geometry=ig, data=dat)
        except ValueError:
            alg = algorithm(image_geometry=ig, data=dat, blocksize=3)

        x = alg.get_output()
        self.assertEqual(np.sum(x.as_array() ** 2), 0)
        l2_error = np.sum((gt.as_array() - x.as_array()) ** 2)
        alg.run(1)
        y = alg.get_output()
        self.assertGreater(np.sum(y.as_array() ** 2), 0)
        l2_error_2 = np.sum((gt.as_array() - y.as_array()) ** 2)
        self.assertLess(l2_error_2, l2_error)
