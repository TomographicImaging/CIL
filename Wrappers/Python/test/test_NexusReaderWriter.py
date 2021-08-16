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

import unittest
import os
from cil.io import NEXUSDataReader
from cil.io import NEXUSDataWriter
from cil.framework import AcquisitionData, AcquisitionGeometry, ImageData, ImageGeometry
import numpy
import shutil
    

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

        self.flat_field_2d = None
        self.dark_field_2d = None


        self.ag3d = AcquisitionGeometry.create_Cone3D([0.1,-500,2], [3,600,-1], [0,1,0],[0,0,-1],[0.2,-0.1,0.5],[-0.1,0.2,0.9])\
                                    .set_angles([0, 90, 180])\
                                    .set_panel([5,10],[0.1,0.3])\

        self.ad3d = self.ag3d.allocate('random_int')

        self.flat_field_3d = None
        self.dark_field_3d = None

    def tearDown(self):
        shutil.rmtree(self.data_dir)
    
    def test_writeImageData(self):
        im_size = 5
        ig = ImageGeometry(voxel_num_x = im_size,
        		           voxel_num_y = im_size)
        im = ig.allocate()
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
	
    def readImageDataAndTest(self):
        
        im_size = 5
        ig_test = ImageGeometry(voxel_num_x = im_size,
                                voxel_num_y = im_size)
        im_test = ig_test.allocate()
        
        reader = NEXUSDataReader()
        reader.set_up(file_name = os.path.join(self.data_dir, 'test_nexus_im.nxs'))
        im = reader.read()
        ig = reader.get_geometry()

        assert ig == ig_test
        numpy.testing.assert_array_equal(im.as_array(), im_test.as_array(), 'Loaded image is not correct')
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
        
        if self.dark_field_2d is not None:
            dark_field_2d = reader2d.load_dark()
            numpy.testing.assert_array_equal(dark_field_2d, self.dark_field_2d, 'Dark Field Data is not correct')
        
        if self.flat_field_2d is not None:
            flat_field_2d = reader2d.load_flat()
            numpy.testing.assert_array_equal(flat_field_2d, self.flat_field_2d, 'Flat Field Data is not correct')

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
        self.assertEqual(ag3d.dist_source_center, self.ag3d.dist_source_center, 'AcquisitionGeometry.dist_source_center is not correct')
        self.assertEqual(ag3d.dist_center_detector, self.ag3d.dist_center_detector, 'AcquisitionGeometry.dist_center_detector is not correct')
        self.assertEqual(ag3d.channels, self.ag3d.channels, 'AcquisitionGeometry.channels is not correct')

        if self.dark_field_3d is not None:
            dark_field_3d = reader3d.load_dark()
            numpy.testing.assert_array_equal(dark_field_3d, self.dark_field_3d)

        if self.flat_field_3d is not None:
            flat_field_3d = reader3d.load_flat()
            numpy.testing.assert_array_equal(flat_field_3d, self.flat_field_3d)

        assert ag3d == self.ag3d

    def test_writeAcquisitionData_with_dark_and_flat_fields(self):
        im_size = 5
        angles = 2
        self.flat_field_2d = numpy.ones((im_size, angles))
        self.dark_field_2d = numpy.zeros((im_size, angles))
        self.dark_field_position_key = numpy.random.randint(0, 2, angles)
        self.flat_field_position_key = numpy.random.randint(0, 2, angles)

        writer = NEXUSDataWriter()
        writer.set_up(file_name=os.path.join(self.data_dir, 'test_nexus_ad2d.nxs'),
                      data=self.ad2d, flat_field=self.flat_field_2d, flat_field_key = self.flat_field_position_key,
                      dark_field=self.dark_field_2d, dark_field_key=self.dark_field_position_key)
        writer.write()

        self.flat_field_3d = numpy.ones((1, 10, 5))
        self.dark_field_3d = numpy.zeros((1, 10, 5))

        self.flat_field_position_key = numpy.random.randint(0, 2, 1)
        # below, we will not set the dark field position key.
        # in this case, it should automatically be set to an
        # array of zeros with the same shape as the dark_field array
        self.dark_field_position_key = numpy.zeros((1))

        writer = NEXUSDataWriter()
        writer.set_up(file_name=os.path.join(self.data_dir, 'test_nexus_ad3d.nxs'),
                      data=self.ad3d, flat_field=self.flat_field_3d,
                      flat_field_key=self.flat_field_position_key,
                      dark_field=self.dark_field_3d)
        writer.write()

        self.readAcquisitionDataAndTest()
    
        
if __name__ == '__main__':
    unittest.main()