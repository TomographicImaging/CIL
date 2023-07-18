# -*- coding: utf-8 -*-
#  Copyright 2019 United Kingdom Research and Innovation
#  Copyright 2019 The University of Manchester
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
import os
from cil.io import NEXUSDataReader
from cil.io import NEXUSDataWriter
from cil.framework import AcquisitionGeometry, ImageGeometry
import numpy
import shutil

initialise_tests()

class TestNexusReaderWriter(unittest.TestCase):
    
    def setUp(self):
        self.data_dir = os.path.join(os.getcwd(), 'test_nxs')
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)

        self.ag2d = AcquisitionGeometry.create_Parallel2D()\
                                    .set_angles([0, 90, 180],-3.0, 'radian')\
                                    .set_panel(5, 0.2, origin='top-right')\
                                    .set_channels(6)\
                                    .set_labels(['horizontal', 'angle'])

        self.ad2d = self.ag2d.allocate('random_int')

        self.ag3d = AcquisitionGeometry.create_Cone3D([0.1,-500,2], [3,600,-1], [0,1,0],[0,0,-1],[0.2,-0.1,0.5],[-0.1,0.2,0.9])\
                                    .set_angles([0, 90, 180])\
                                    .set_panel([5,10],[0.1,0.3])\

        self.ad3d = self.ag3d.allocate('random_int')


    def tearDown(self):
        shutil.rmtree(self.data_dir)


    def test_writeImageData(self):
        im_size = 5
        ig = ImageGeometry(voxel_num_x = im_size,
        		           voxel_num_y = im_size)
        im = ig.allocate('random',seed=9)
        writer = NEXUSDataWriter()
        writer.set_up(file_name = os.path.join(self.data_dir, 'test_nexus_im.nxs'),
                      data = im)
        writer.write()
        self.readImageDataAndTest()


    def test_writeAcquisitionData(self):
        writer = NEXUSDataWriter()
        writer.set_up(file_name = os.path.join(self.data_dir, 'test_nexus_ad2d.nxs'),
                      data = self.ad2d)
        writer.write()
        
        writer = NEXUSDataWriter()
        writer.set_up(file_name = os.path.join(self.data_dir, 'test_nexus_ad3d.nxs'),
                      data = self.ad3d)
        writer.write()

        self.readAcquisitionDataAndTest()


    def test_writeImageData_compressed(self):
        im_size = 5
        ig = ImageGeometry(voxel_num_x = im_size,
        		           voxel_num_y = im_size)
        im = ig.allocate('random',seed=9)

        writer = NEXUSDataWriter()
        writer.set_up(file_name = os.path.join(self.data_dir, 'test_nexus_im'),
                      data = im, compression=16)
        writer.write()

        self.assertTrue(writer.dtype == numpy.uint16)
        self.assertTrue(writer.compression == 16)

        self.readImageDataAndTest(atol=1e-4)

    def test_write_throws_when_data_is_none(self):
        with self.assertRaises(TypeError) as cm:
            writer = NEXUSDataWriter(file_name='test_file_name.nxs')
            writer.write()
        self.assertEqual(str(cm.exception), 'Data to write out must be set.')

    def test_write_throws_when_file_is_none(self):
        with self.assertRaises(TypeError) as cm:
            writer = NEXUSDataWriter(data=self.ad2d)
            writer.write()
        self.assertEqual(str(cm.exception), 'Path to nexus file to write to is required.')

    def test_write_throws_when_file_is_not_path_like(self):
        with self.assertRaises(TypeError) as cm:
            writer = NEXUSDataWriter(file_name=self.ad2d , data=self.ad2d)
            writer.write()

    def test_write_throws_when_file_path_not_possible(self):
        with self.assertRaises(OSError):
            writer = NEXUSDataWriter(file_name="_/_/_" , data=self.ad2d)
            writer.write()


    def readImageDataAndTest(self,atol=0):        
        im_size = 5
        ig_test = ImageGeometry(voxel_num_x = im_size,
                                voxel_num_y = im_size)
        im_test = ig_test.allocate('random',seed=9)
        
        reader = NEXUSDataReader()
        reader.set_up(file_name = os.path.join(self.data_dir, 'test_nexus_im.nxs'))
        im = reader.read()
        ig = reader.get_geometry()

        assert ig == ig_test
        numpy.testing.assert_allclose(im.as_array(), im_test.as_array(),atol=atol, err_msg='Loaded image is not correct')
        self.assertEqual(ig.voxel_num_x, ig_test.voxel_num_x, 'ImageGeometry is not correct')
        self.assertEqual(ig.voxel_num_y, ig_test.voxel_num_y, 'ImageGeometry is not correct')
        

    def readAcquisitionDataAndTest(self):
        reader2d = NEXUSDataReader()
        reader2d.set_up(file_name = os.path.join(self.data_dir, 'test_nexus_ad2d.nxs'))
        ad2d = reader2d.read()
        ag2d = reader2d.get_geometry()

        numpy.testing.assert_array_equal(ad2d.as_array(), self.ad2d.as_array(), 'Loaded image is not correct')
        self.assertEqual(ag2d.geom_type, self.ag2d.geom_type, 'ImageGeometry.geom_type is not correct')
        numpy.testing.assert_array_equal(ag2d.angles, self.ag2d.angles, 'ImageGeometry.angles is not correct')
        self.assertEqual(ag2d.pixel_num_h, self.ag2d.pixel_num_h, 'ImageGeometry.pixel_num_h is not correct')
        self.assertEqual(ag2d.pixel_size_h, self.ag2d.pixel_size_h, 'ImageGeometry.pixel_size_h is not correct')
        self.assertEqual(ag2d.pixel_num_v, self.ag2d.pixel_num_v, 'ImageGeometry.pixel_num_v is not correct')
        self.assertEqual(ag2d.pixel_size_v, self.ag2d.pixel_size_v, 'ImageGeometry.pixel_size_v is not correct')

        assert ag2d == self.ag2d

        reader3d = NEXUSDataReader()
        reader3d.set_up(file_name = os.path.join(self.data_dir, 'test_nexus_ad3d.nxs'))
        ad3d = reader3d.read()
        ag3d = reader3d.get_geometry()
        
        numpy.testing.assert_array_equal(ad3d.as_array(), self.ad3d.as_array(), 'Loaded image is not correct')
        numpy.testing.assert_array_equal(ag3d.angles, self.ag3d.angles, 'AcquisitionGeometry.angles is not correct')
        self.assertEqual(ag3d.geom_type, self.ag3d.geom_type, 'AcquisitionGeometry.geom_type is not correct')
        self.assertEqual(ag3d.dimension, self.ag3d.dimension, 'AcquisitionGeometry.dimension is not correct')
        self.assertEqual(ag3d.pixel_num_h, self.ag3d.pixel_num_h, 'AcquisitionGeometry.pixel_num_h is not correct')
        self.assertEqual(ag3d.pixel_size_h, self.ag3d.pixel_size_h, 'AcquisitionGeometry.pixel_size_h is not correct')
        self.assertEqual(ag3d.pixel_num_v, self.ag3d.pixel_num_v, 'AcquisitionGeometry.pixel_num_v is not correct')
        self.assertEqual(ag3d.pixel_size_v, self.ag3d.pixel_size_v, 'AcquisitionGeometry.pixel_size_v is not correct')
        self.assertEqual(ag3d.channels, self.ag3d.channels, 'AcquisitionGeometry.channels is not correct')

        assert ag3d == self.ag3d

