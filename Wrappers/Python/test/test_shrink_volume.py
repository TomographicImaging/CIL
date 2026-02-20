#  Copyright 2026 United Kingdom Research and Innovation
#  Copyright 2026 The University of Manchester
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
from unittest.mock import patch
import numpy as np

from cil.framework import ImageGeometry, ImageData
from cil.framework.labels import AcquisitionType

from cil.utilities.shrink_volume import VolumeShrinker
from cil.utilities import dataexample

from utils import has_astra, has_tigre, has_nvidia, initialise_tests

initialise_tests()


class TestVolumeShrinker(unittest.TestCase):

    def setUp(self):
        self.data_cone = dataexample.SIMULATED_CONE_BEAM_DATA.get()
        self.data_parallel = dataexample.SYNCHROTRON_PARALLEL_BEAM_DATA.get()
        arr = np.zeros((10, 10, 10), dtype=np.float32)
        arr[2:8, 2:8, 2:8] = 0.2
        arr[3:7, 3:7, 3:7] = 1
        
        self.test_recon = ImageData(arr, geometry=ImageGeometry(10,10,10))
        

    @unittest.skipUnless(has_tigre and has_nvidia, "TIGRE GPU not installed")
    def test_init_tigre(self):
        for data in [self.data_cone, self.data_parallel]:
            data.reorder('tigre')
            vs = VolumeShrinker(data, recon_backend='tigre')
            self.assertEqual(vs.recon_backend, 'tigre')
            self.assertEqual(vs.data, data)

            data.reorder('astra')
            with self.assertRaises(ValueError):
                vs = VolumeShrinker(data, recon_backend='tigre')
                vs.run()

    @unittest.skipUnless(has_astra and has_nvidia, "Astra GPU not installed")
    def test_init_astra(self):
        for data in [self.data_cone, self.data_parallel]:
            data.reorder('astra')
            vs = VolumeShrinker(data, recon_backend='astra')
            self.assertEqual(vs.recon_backend, 'astra')
            self.assertEqual(vs.data, data)

            data.reorder('tigre')
            with self.assertRaises(ValueError):
                vs = VolumeShrinker(data, recon_backend='astra')
                vs.run()

    @unittest.skipUnless(has_astra and has_nvidia, "Astra GPU not installed")
    def test_run_manual_astra(self):

        limits = {
                'horizontal_x': (10, 50),
                'horizontal_y': (20, 90),
                'vertical': (2, 25)
            }
        
        for data in [self.data_cone, self.data_parallel]:
            data.reorder('astra')
            vs = VolumeShrinker(data, recon_backend='astra')
            new_ig = vs.run(limits=limits, preview=False, method='manual')

            # expected sizes are stop - start
            for dim in ['horizontal_x', 'horizontal_y', 'vertical']:
                self.assertEqual(new_ig.shape[new_ig.dimension_labels.index(dim)], limits[dim][1] - limits[dim][0])

    @unittest.skipUnless(has_tigre and has_nvidia, "TIGRE GPU not installed")
    def test_run_manual_astra(self):

        limits = {
                'horizontal_x': (10, 50),
                'horizontal_y': (20, 90),
                'vertical': (2, 25)
            }
        
        for data in [self.data_cone, self.data_parallel]:
            data.reorder('tigre')
            vs = VolumeShrinker(data, recon_backend='tigre')
            new_ig = vs.run(limits=limits, preview=False, method='manual')

            # expected sizes are stop - start
            for dim in ['horizontal_x', 'horizontal_y', 'vertical']:
                self.assertEqual(new_ig.shape[new_ig.dimension_labels.index(dim)], limits[dim][1] - limits[dim][0])

    unittest.skipUnless(has_astra and has_nvidia, "Astra GPU not installed")
    def test_reduce_reconstruction_volume_astra(self):
        vs = VolumeShrinker(self.data_cone, recon_backend='astra')
        bounds = vs.reduce_reconstruction_volume(self.test_recon, binning=1, method='threshold', kwargs={'threshold':0.5})
        for dim in ['horizontal_x', 'horizontal_y', 'vertical']:
            self.assertEqual(bounds[dim], (3,6))

        bounds = vs.reduce_reconstruction_volume(self.test_recon, binning=1, method='threshold', kwargs={'threshold':0.1})
        for dim in ['horizontal_x', 'horizontal_y', 'vertical']:
            self.assertEqual(bounds[dim], (2,7))
            
        bounds = vs.reduce_reconstruction_volume(self.test_recon, binning=1, method='otsu', kwargs={})
        for dim in ['horizontal_x', 'horizontal_y', 'vertical']:
            self.assertEqual(bounds[dim], (3,6))

        bounds = vs.reduce_reconstruction_volume(self.test_recon, binning=1, method='otsu', kwargs={'buffer':1})
        for dim in ['horizontal_x', 'horizontal_y', 'vertical']:
            self.assertEqual(bounds[dim], (2,7))

