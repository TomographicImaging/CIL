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

        arr = np.zeros((10, 10, 10), dtype=np.float32)
        arr[2:8, 2:8, 3:7] = 0.2
        arr[3:7, 3:7, 4:6] = 1
        self.test_recon_asymmetrical = ImageData(arr, geometry=ImageGeometry(10,10,10))
        

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
            old_ig = data.geometry.get_ImageGeometry()
            vs = VolumeShrinker(data, recon_backend='astra')
            new_ig = vs.run(limits=limits, preview=False, method='manual')
            # get the voxel size and centers that correspond to each dimension
            voxel_sizes = {'horizontal_x': old_ig.voxel_size_x,'horizontal_y': old_ig.voxel_size_y,'vertical': old_ig.voxel_size_z}
            centers = {'horizontal_x': new_ig.center_x, 'horizontal_y': new_ig.center_y, 'vertical': new_ig.center_z}
            
            for dim in ['horizontal_x', 'horizontal_y', 'vertical']:
                # expected sizes are stop - start
                new_size = limits[dim][1] - limits[dim][0]
                ind = new_ig.dimension_labels.index(dim)
                self.assertEqual(new_ig.shape[ind], new_size)

                old_size = old_ig.shape[ind]

                # expected center is (new center-old center)*voxel size
                center = ((limits[dim][0]+(new_size/2))-(old_size/2))*voxel_sizes[dim]
                self.assertEqual(center, centers[dim])


        # test some non-sensical limits
        limits = {
                'horizontal_x': (10, 5),
                'horizontal_y': (20, 9),
                'vertical': (12, 2)
            }

        for data in [self.data_cone, self.data_parallel]:
            data.reorder('astra')
            vs = VolumeShrinker(data, recon_backend='astra')
            with self.assertRaises(ValueError):
                new_ig = vs.run(limits=limits, preview=False, method='manual')

    @unittest.skipUnless(has_tigre and has_nvidia, "TIGRE GPU not installed")
    def test_run_manual_tigre(self):

        limits = {
                'horizontal_x': (10, 50),
                'horizontal_y': (20, 90),
                'vertical': (2, 25)
            }
        
        for data in [self.data_cone, self.data_parallel]:
            data.reorder('tigre')
            old_ig = data.geometry.get_ImageGeometry()
            vs = VolumeShrinker(data, recon_backend='tigre')
            new_ig = vs.run(limits=limits, preview=False, method='manual')

            # get the voxel size and centers that correspond to each dimension
            voxel_sizes = {'horizontal_x': old_ig.voxel_size_x,'horizontal_y': old_ig.voxel_size_y,'vertical': old_ig.voxel_size_z}
            centers = {'horizontal_x': new_ig.center_x, 'horizontal_y': new_ig.center_y, 'vertical': new_ig.center_z}
            
            for dim in ['horizontal_x', 'horizontal_y', 'vertical']:
                # expected sizes are stop - start
                new_size = limits[dim][1] - limits[dim][0]
                ind = new_ig.dimension_labels.index(dim)
                self.assertEqual(new_ig.shape[ind], new_size)

                old_size = old_ig.shape[ind]

                # expected center is (new center-old center)*voxel size
                center = ((limits[dim][0]+(new_size/2))-(old_size/2))*voxel_sizes[dim]
                self.assertEqual(center, centers[dim])

        # test some non-sensical limits
        limits = {
                'horizontal_x': (10, 5),
                'horizontal_y': (20, 9),
                'vertical': (12, 2)
            }
        
        for data in [self.data_cone, self.data_parallel]:
            data.reorder('tigre')
            vs = VolumeShrinker(data, recon_backend='tigre')
            with self.assertRaises(ValueError):
                new_ig = vs.run(limits=limits, preview=False, method='manual')

    def test_reduce_reconstruction_volume(self):
        vs = VolumeShrinker(self.data_cone, recon_backend='astra')

        # test error with threshold higher than max value in volume
        with self.assertRaises(ValueError):
            bounds = vs._reduce_reconstruction_volume(self.test_recon, binning=1, method='threshold', threshold=500)

        # test expected boundaries are found
        bounds = vs._reduce_reconstruction_volume(self.test_recon, binning=1, method='threshold', threshold=0.5)
        for dim in ['horizontal_x', 'horizontal_y', 'vertical']:
            self.assertEqual(bounds[dim], (3,6))

        bounds = vs._reduce_reconstruction_volume(self.test_recon, binning=1, method='threshold', threshold=0.1)
        for dim in ['horizontal_x', 'horizontal_y', 'vertical']:
            self.assertEqual(bounds[dim], (2,7))

        bounds = vs._reduce_reconstruction_volume(self.test_recon, binning=1, method='otsu')
        for dim in ['horizontal_x', 'horizontal_y', 'vertical']:
            self.assertEqual(bounds[dim], (3,6))

        bounds = vs._reduce_reconstruction_volume(self.test_recon, binning=1, method='otsu', buffer=1)
        for dim in ['horizontal_x', 'horizontal_y', 'vertical']:
            self.assertEqual(bounds[dim], (2,7))

        # test the asymmetrical volume
        bounds = vs._reduce_reconstruction_volume(self.test_recon_asymmetrical, binning=1, method='threshold', threshold=0.5)
        self.assertEqual(bounds['horizontal_x'], (4,5))
        self.assertEqual(bounds['horizontal_y'], (3,6))
        self.assertEqual(bounds['vertical'], (3,6))

        bounds = vs._reduce_reconstruction_volume(self.test_recon_asymmetrical, binning=1, method='threshold', threshold=0.1)
        self.assertEqual(bounds['horizontal_x'], (3,6))
        self.assertEqual(bounds['horizontal_y'], (2,7))
        self.assertEqual(bounds['vertical'], (2,7))

        bounds = vs._reduce_reconstruction_volume(self.test_recon_asymmetrical, binning=1, method='otsu')
        self.assertEqual(bounds['horizontal_x'], (4,5))
        self.assertEqual(bounds['horizontal_y'], (3,6))
        self.assertEqual(bounds['vertical'], (3,6))

        bounds = vs._reduce_reconstruction_volume(self.test_recon_asymmetrical, binning=1, method='otsu', buffer=1)
        self.assertEqual(bounds['horizontal_x'], (3,6))
        self.assertEqual(bounds['horizontal_y'], (2,7))
        self.assertEqual(bounds['vertical'], (2,7))

