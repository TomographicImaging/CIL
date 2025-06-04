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
from parameterized import parameterized
import numpy as np

initialise_tests()

if has_tigre:
    from cil.plugins.tigre import ART, SART, SIRT, OSSART
    


class TestTigreReconstructionAlgorithms(unittest.TestCase):

    def setUp(self):
        self.ground_truth_3D = dataexample.SIMULATED_SPHERE_VOLUME.get()

        self.data_cone = dataexample.SIMULATED_CONE_BEAM_DATA.get()

        self.data_fan_beam = self.data_cone.get_slice(vertical='centre')
        self.ground_truth_2D = self.ground_truth_3D.get_slice(vertical='centre')

        self.data_parallel = dataexample.SIMULATED_PARALLEL_BEAM_DATA.get()

        self.data_parallel_2D= self.data_parallel.get_slice(vertical='centre')

        
        self.ig3D = self.ground_truth_3D.geometry
        self.ig2D = self.ground_truth_2D.geometry 
        
        ag_fan = self.data_fan_beam.geometry
        ag_cone = self.data_cone.geometry
        
        ag_parallel_2D = self.data_parallel_2D.geometry
        ag_parallel = self.data_parallel.geometry
        
    @unittest.skipUnless(has_tigre , "Requires TIGRE")    
    @parameterized.expand([('SART', SART), ('SIRT', SIRT), ('OSSART', OSSART)])
    def test_missing_parameters_raises_error(self, name, alg):
        with self.assertRaises(ValueError) as context:
            alg()
        self.assertIn("You must pass", str(context.exception))

    @unittest.skipUnless(has_tigre , "Requires TIGRE")    
    def test_sirt_initialization_success(self):
        alg = SIRT(image_geometry=self.ig2D, data=self.data_parallel_2D)
        self.assertTrue(alg.configured)
        self.assertTrue(alg.tigre_alg.blocksize ==  len(self.data_parallel_2D.geometry.angles))
        self.assertTrue(alg.tigre_alg.niter == 0)
        self.assertTrue(alg.tigre_alg.__dict__['noneg'])
        
    @unittest.skipUnless(has_tigre , "Requires TIGRE")            
    def test_sart_initialization_success(self):
        alg = SART(image_geometry=self.ig2D, data=self.data_parallel_2D, noneg=False)
        self.assertTrue(alg.configured)
        self.assertTrue(alg.tigre_alg.blocksize ==  1)
        self.assertTrue(alg.tigre_alg.niter == 0)
        self.assertFalse(alg.tigre_alg.__dict__['noneg'])
        self.assertNumpyArrayEqual(self.ig2D.allocate(0).as_array(), alg.get_output().as_array())

    @unittest.skipUnless(has_tigre , "Requires TIGRE")            
    def test_ossart_initialization_success(self):
        alg = OSSART(initial = self.ig2D.allocate(1), image_geometry=self.ig2D, data=self.data_parallel_2D, blocksize=2,  OrderStrategy='random')
        self.assertTrue(alg.configured)
        self.assertTrue(alg.tigre_alg.blocksize ==  2)
        self.assertTrue(alg.tigre_alg.niter == 0)
        self.assertTrue(alg.tigre_alg.__dict__['noneg'])
        self.assertTrue(alg.tigre_alg.__dict__['OrderStrategy']=='random')
        self.assertNumpyArrayEqual(alg.tigre_alg.__dict__['init'],self.ig2D.allocate(1).as_array() )
        self.assertNumpyArrayEqual(alg.get_output().as_array(), self.ig2D.allocate(1).as_array() )
        
    

    @parameterized.expand([('SART_2D_parallel', SART, 'ig2D', 'data_parallel_2D'),
    ('SIRT_2D_parallel', SIRT, 'ig2D', 'data_parallel_2D'),
    ('OSSART_2D_parallel', OSSART, 'ig2D', 'data_parallel_2D'),
    ('SART_cone', SART, 'ig2D', 'data_cone'),
    ('SIRT_cone', SIRT, 'ig2D', 'data_cone'),
    ('OSSART_cone', OSSART, 'ig2D', 'data_cone'),
    ('SART_3D_parallel', SART, 'ig3D', 'data_parallel_3D'),
    ('SIRT_3D_parallel', SIRT, 'ig3D', 'data_parallel_3D'),
    ('OSSART_3D_parallel', OSSART, 'ig3D', 'data_parallel_3D'),
    ('SART_fan_beam', SART, 'ig3D', 'data_fan_beam'),
    ('SIRT_fan_beam', SIRT, 'ig3D', 'data_fan_beam'),
    ('OSSART_fan_beam', OSSART, 'ig3D', 'data_fan_beam'),])
    @unittest.skipUnless(has_tigre and has_nvidia, "Requires TIGRE GPU")
    def test_update(self, name, algorithm, image_geometry, data):

        if image_geometry =='ig2D':
            ig = self.ig2D
            gt = self.ground_truth_2D
        else:
            ig = self.ig3D
            gt = self.ground_truth_3D
        if data == 'data_cone':
            dat = self.data_cone
        elif data == 'data_fan_beam':
            dat = self.data_fan_beam
        elif data == 'data_parallel_2D':
            dat = self.data_parallel_2D
        else:
            dat = self.data_parallel_3D
            
        try:
            alg = algorithm(image_geometry=ig, data = dat)
        except ValueError: 
            alg = algorithm(image_geometry=ig, data = dat, blocksize=3)
        
        
        x = alg.get_output()
        l2_error = np.sum((gt.as_array()-x.as_array())**2)
        alg.run(1)
        y = alg.get_output()
        self.assertTrue( np.sum( (x.as_array() - y.as_array())**2)>0)
        l2_error_2 = np.sum((gt.as_array()-y.as_array())**2)
        self.assertTrue(ls_error2 < l2_error)
        
        
        
        
            
            
