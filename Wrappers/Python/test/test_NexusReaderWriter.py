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
from __future__ import division

import unittest
import os
from cil.io import NEXUSDataReader
from cil.io import NEXUSDataWriter
from cil.framework import AcquisitionData, AcquisitionGeometry, ImageData, ImageGeometry
import numpy
    

class TestNexusReaderWriter(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def testwriteImageData(self):
        im_size = 5
        ig = ImageGeometry(voxel_num_x = im_size,
        		           voxel_num_y = im_size)
        im = ig.allocate()
        writer = NEXUSDataWriter()
        writer.set_up(file_name = os.path.join(os.getcwd(), 'test_nexus_im.nxs'),
                      data_container = im)
        writer.write_file()
        self.stestreadImageData()
        
    def testwriteAcquisitionData(self):
        im_size = 5
        ag2d = AcquisitionGeometry(geom_type = 'parallel', 
                                   dimension = '2D', 
                                   angles = numpy.array([0, 1]), 
                                   pixel_num_h = im_size, 
                                   pixel_size_h = 1, 
                                   pixel_num_v = im_size, 
                                   pixel_size_v = 1)
        ad2d = ag2d.allocate()
        writer = NEXUSDataWriter()
        writer.set_up(file_name = os.path.join(os.getcwd(), 'test_nexus_ad2d.nxs'),
                      data_container = ad2d)
        writer.write_file()
        
        ag3d = AcquisitionGeometry(geom_type = 'cone', 
                                   dimension = '3D', 
                                   angles = numpy.array([0, 1]), 
                                   pixel_num_h = im_size, 
                                   pixel_size_h = 1, 
                                   pixel_num_v = im_size, 
                                   pixel_size_v = 1,
                                   dist_source_center = 1,
                                   dist_center_detector = 1, 
                                   channels = im_size)
        ad3d = ag3d.allocate()
        writer = NEXUSDataWriter()
        writer.set_up(file_name = os.path.join(os.getcwd(), 'test_nexus_ad3d.nxs'),
                      data_container = ad3d)
        writer.write_file()

        self.stestreadAcquisitionData()
	
    def stestreadImageData(self):
        
        im_size = 5
        ig_test = ImageGeometry(voxel_num_x = im_size,
                                voxel_num_y = im_size)
        im_test = ig_test.allocate()
        
        reader = NEXUSDataReader()
        reader.set_up(nexus_file = os.path.join(os.getcwd(), 'test_nexus_im.nxs'))
        im = reader.load_data()
        ig = reader.get_geometry()
        numpy.testing.assert_array_equal(im.as_array(), im_test.as_array(), 'Loaded image is not correct')
        self.assertEqual(ig.voxel_num_x, ig_test.voxel_num_x, 'ImageGeometry is not correct')
        self.assertEqual(ig.voxel_num_y, ig_test.voxel_num_y, 'ImageGeometry is not correct')
        
    def stestreadAcquisitionData(self):
        im_size = 5
        ag2d_test = AcquisitionGeometry(geom_type = 'parallel', 
                                        dimension = '2D', 
                                        angles = numpy.array([0, 1]), 
                                        pixel_num_h = im_size, 
                                        pixel_size_h = 1, 
                                        pixel_num_v = im_size, 
                                        pixel_size_v = 1)
        ad2d_test = ag2d_test.allocate()
        
        reader2d = NEXUSDataReader()
        reader2d.set_up(nexus_file = os.path.join(os.getcwd(), 'test_nexus_ad2d.nxs'))
        ad2d = reader2d.load_data()
        ag2d = reader2d.get_geometry()
        numpy.testing.assert_array_equal(ad2d.as_array(), ad2d_test.as_array(), 'Loaded image is not correct')
        self.assertEqual(ag2d.geom_type, ag2d_test.geom_type, 'ImageGeometry.geom_type is not correct')
        numpy.testing.assert_array_equal(ag2d.angles, ag2d_test.angles, 'ImageGeometry.angles is not correct')
        self.assertEqual(ag2d.pixel_num_h, ag2d_test.pixel_num_h, 'ImageGeometry.pixel_num_h is not correct')
        self.assertEqual(ag2d.pixel_size_h, ag2d_test.pixel_size_h, 'ImageGeometry.pixel_size_h is not correct')
        self.assertEqual(ag2d.pixel_num_v, ag2d_test.pixel_num_v, 'ImageGeometry.pixel_num_v is not correct')
        self.assertEqual(ag2d.pixel_size_v, ag2d_test.pixel_size_v, 'ImageGeometry.pixel_size_v is not correct')
        
        ag3d_test = AcquisitionGeometry(geom_type = 'cone', 
                                        dimension = '3D', 
                                        angles = numpy.array([0, 1]), 
                                        pixel_num_h = im_size, 
                                        pixel_size_h = 1, 
                                        pixel_num_v = im_size, 
                                        pixel_size_v = 1,
                                        dist_source_center = 1,
                                        dist_center_detector = 1, 
                                        channels = im_size)
        ad3d_test = ag3d_test.allocate()
        
        reader3d = NEXUSDataReader()
        reader3d.set_up(nexus_file = os.path.join(os.getcwd(), 'test_nexus_ad3d.nxs'))
        ad3d = reader3d.load_data()
        ag3d = reader3d.get_geometry()
        
        numpy.testing.assert_array_equal(ad3d.as_array(), ad3d_test.as_array(), 'Loaded image is not correct')
        numpy.testing.assert_array_equal(ag3d.angles, ag3d_test.angles, 'AcquisitionGeometry.angles is not correct')
        self.assertEqual(ag3d.geom_type, ag3d_test.geom_type, 'AcquisitionGeometry.geom_type is not correct')
        self.assertEqual(ag3d.dimension, ag3d_test.dimension, 'AcquisitionGeometry.dimension is not correct')
        self.assertEqual(ag3d.pixel_num_h, ag3d_test.pixel_num_h, 'AcquisitionGeometry.pixel_num_h is not correct')
        self.assertEqual(ag3d.pixel_size_h, ag3d_test.pixel_size_h, 'AcquisitionGeometry.pixel_size_h is not correct')
        self.assertEqual(ag3d.pixel_num_v, ag3d_test.pixel_num_v, 'AcquisitionGeometry.pixel_num_v is not correct')
        self.assertEqual(ag3d.pixel_size_v, ag3d_test.pixel_size_v, 'AcquisitionGeometry.pixel_size_v is not correct')
        self.assertEqual(ag3d.dist_source_center, ag3d_test.dist_source_center, 'AcquisitionGeometry.dist_source_center is not correct')
        self.assertEqual(ag3d.dist_center_detector, ag3d_test.dist_center_detector, 'AcquisitionGeometry.dist_center_detector is not correct')
        self.assertEqual(ag3d.channels, ag3d_test.channels, 'AcquisitionGeometry.channels is not correct')
                
        def tearDown(self):
            os.remove(os.path.join(os.getcwd(), 'test_nexus_im.nxs'))
            os.remove(os.path.join(os.getcwd(), 'test_nexus_ad2d.nxs'))
            os.remove(os.path.join(os.getcwd(), 'test_nexus_ad3d.nxs'))

if __name__ == '__main__':
    unittest.main()