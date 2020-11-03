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

import sys
import unittest
import numpy
from ccpi.framework import DataContainer
from ccpi.framework import ImageData
from ccpi.framework import AcquisitionData
from ccpi.framework import ImageGeometry
from ccpi.framework import AcquisitionGeometry
from timeit import default_timer as timer

class TestSubset(unittest.TestCase):
    def setUp(self):
        self.ig = ImageGeometry(2,3,4,channels=5)
        angles = numpy.asarray([90.,0.,-90.], dtype=numpy.float32)

        self.ag = AcquisitionGeometry('parallel', 'edo', pixel_num_h=20, pixel_num_v=2, angles=angles, 
                         dist_source_center = 312.2, 
                         dist_center_detector = 123.,
                         channels=4 )
        
        self.ag_cone = AcquisitionGeometry.create_Cone3D([0,-500,0],[0,500,0])\
                                     .set_panel((2,20))\
                                     .set_angles(self.ag.angles)\
                                     .set_channels(4)


    def test_ImageDataAllocate1a(self):
        data = self.ig.allocate()
        default_dimension_labels = [ImageGeometry.CHANNEL, ImageGeometry.VERTICAL,
                ImageGeometry.HORIZONTAL_Y, ImageGeometry.HORIZONTAL_X]
        self.assertTrue( default_dimension_labels == list(data.dimension_labels.values()) )
    def test_ImageDataAllocate1b(self):
        data = self.ig.allocate()
        self.assertTrue( data.shape == (5,4,3,2))
        
    def test_ImageDataAllocate2a(self):
        non_default_dimension_labels = [ ImageGeometry.HORIZONTAL_X, ImageGeometry.VERTICAL,
                ImageGeometry.HORIZONTAL_Y, ImageGeometry.CHANNEL]
        data = self.ig.allocate(dimension_labels=non_default_dimension_labels)
        self.assertTrue( non_default_dimension_labels == list(data.dimension_labels.values()) )
        
    def test_ImageDataAllocate2b(self):
        non_default_dimension_labels = [ ImageGeometry.HORIZONTAL_X, ImageGeometry.VERTICAL,
                ImageGeometry.HORIZONTAL_Y, ImageGeometry.CHANNEL]
        data = self.ig.allocate(dimension_labels=non_default_dimension_labels)
        self.assertTrue( data.shape == (2,4,3,5))

    def test_ImageDataSubset1a(self):
        non_default_dimension_labels = [ImageGeometry.HORIZONTAL_X, ImageGeometry.CHANNEL, ImageGeometry.HORIZONTAL_Y,
        ImageGeometry.VERTICAL]
        data = self.ig.allocate(dimension_labels=non_default_dimension_labels)
        sub = data.subset(horizontal_y = 1)
        self.assertTrue( sub.shape == (2,5,4))

    def test_ImageDataSubset2a(self):
        non_default_dimension_labels = [ImageGeometry.HORIZONTAL_X, ImageGeometry.CHANNEL, ImageGeometry.HORIZONTAL_Y,
        ImageGeometry.VERTICAL]
        data = self.ig.allocate(dimension_labels=non_default_dimension_labels)
        sub = data.subset(horizontal_x = 1)
        self.assertTrue( sub.shape == (5,3,4))

    def test_ImageDataSubset3a(self):
        non_default_dimension_labels = [ImageGeometry.HORIZONTAL_X, ImageGeometry.CHANNEL, ImageGeometry.HORIZONTAL_Y,
        ImageGeometry.VERTICAL]
        data = self.ig.allocate(dimension_labels=non_default_dimension_labels)
        sub = data.subset(channel = 1)
        self.assertTrue( sub.shape == (2,3,4))

    def test_ImageDataSubset4a(self):
        non_default_dimension_labels = [ImageGeometry.HORIZONTAL_X, ImageGeometry.CHANNEL, ImageGeometry.HORIZONTAL_Y,
        ImageGeometry.VERTICAL]
        data = self.ig.allocate(dimension_labels=non_default_dimension_labels)
        sub = data.subset(vertical = 1)
        self.assertTrue( sub.shape == (2,5,3))

    def test_ImageDataSubset5a(self):
        non_default_dimension_labels = [ImageGeometry.HORIZONTAL_X, ImageGeometry.HORIZONTAL_Y]
        data = self.ig.allocate(dimension_labels=non_default_dimension_labels)
        sub = data.subset(horizontal_y = 1)
        self.assertTrue( sub.shape == (2,))

    def test_ImageDataSubset1b(self):
        non_default_dimension_labels = [ImageGeometry.HORIZONTAL_X, ImageGeometry.CHANNEL, ImageGeometry.HORIZONTAL_Y,
        ImageGeometry.VERTICAL]
        data = self.ig.allocate(dimension_labels=non_default_dimension_labels)
        new_dimension_labels = [ImageGeometry.HORIZONTAL_Y, ImageGeometry.CHANNEL, ImageGeometry.VERTICAL, ImageGeometry.HORIZONTAL_X]
        sub = data.subset(dimensions=new_dimension_labels)
        self.assertTrue( sub.shape == (3,5,4,2))


    def test_AcquisitionDataAllocate1a(self):
        data = self.ag.allocate()
        default_dimension_labels = [AcquisitionGeometry.CHANNEL ,
                 AcquisitionGeometry.ANGLE , AcquisitionGeometry.VERTICAL ,
                 AcquisitionGeometry.HORIZONTAL]
        self.assertTrue(  default_dimension_labels == list(data.dimension_labels.values()) )

    def test_AcquisitionDataAllocate1b(self):
        data = self.ag.allocate()
        self.assertTrue( data.shape == (4,3,2,20))

    def test_AcquisitionDataAllocate2a(self):
        non_default_dimension_labels = [AcquisitionGeometry.CHANNEL, AcquisitionGeometry.HORIZONTAL,
         AcquisitionGeometry.VERTICAL, AcquisitionGeometry.ANGLE]
        data = self.ag.allocate(dimension_labels=non_default_dimension_labels)


        self.assertTrue(  non_default_dimension_labels == list(data.dimension_labels.values()) )
        
    def test_AcquisitionDataAllocate2b(self):
        non_default_dimension_labels = [AcquisitionGeometry.CHANNEL, AcquisitionGeometry.HORIZONTAL,
         AcquisitionGeometry.VERTICAL, AcquisitionGeometry.ANGLE]
        data = self.ag.allocate(dimension_labels=non_default_dimension_labels)
        self.assertTrue( data.shape == (4,20,2,3))

    def test_AcquisitionDataSubset1a(self):
        non_default_dimension_labels = [AcquisitionGeometry.CHANNEL, AcquisitionGeometry.HORIZONTAL,
         AcquisitionGeometry.VERTICAL, AcquisitionGeometry.ANGLE]
        data = self.ag.allocate(dimension_labels=non_default_dimension_labels)
        #self.assertTrue( data.shape == (4,20,2,3))
        sub = data.subset(vertical = 0)
        self.assertTrue( sub.shape == (4,20,3))
    
    def test_AcquisitionDataSubset1b(self):
        non_default_dimension_labels = [AcquisitionGeometry.CHANNEL, AcquisitionGeometry.HORIZONTAL,
         AcquisitionGeometry.VERTICAL, AcquisitionGeometry.ANGLE]
        data = self.ag.allocate(dimension_labels=non_default_dimension_labels)
        #self.assertTrue( data.shape == (4,20,2,3))
        sub = data.subset(channel = 0)
        self.assertTrue( sub.shape == (20,2,3))
    def test_AcquisitionDataSubset1c(self):
        non_default_dimension_labels = [AcquisitionGeometry.CHANNEL, AcquisitionGeometry.HORIZONTAL,
         AcquisitionGeometry.VERTICAL, AcquisitionGeometry.ANGLE]
        data = self.ag.allocate(dimension_labels=non_default_dimension_labels)
        #self.assertTrue( data.shape == (4,20,2,3))
        sub = data.subset(horizontal = 0)
        self.assertTrue( sub.shape == (4,2,3))
    def test_AcquisitionDataSubset1d(self):
        non_default_dimension_labels = [AcquisitionGeometry.CHANNEL, AcquisitionGeometry.HORIZONTAL,
         AcquisitionGeometry.VERTICAL, AcquisitionGeometry.ANGLE]
        data = self.ag.allocate(dimension_labels=non_default_dimension_labels)
        #self.assertTrue( data.shape == (4,20,2,3))
        sliceme = 1
        sub = data.subset(angle = sliceme)
        #print (sub.shape  , sub.dimension_labels)
        self.assertTrue( sub.shape == (4,20,2) )
        self.assertTrue( sub.geometry.angles[0] == data.geometry.angles[sliceme])
    def test_AcquisitionDataSubset1e(self):
        non_default_dimension_labels = [AcquisitionGeometry.CHANNEL, AcquisitionGeometry.HORIZONTAL,
         AcquisitionGeometry.VERTICAL, AcquisitionGeometry.ANGLE]
        data = self.ag.allocate(dimension_labels=non_default_dimension_labels)
        #self.assertTrue( data.shape == (4,20,2,3))
        sliceme = 1
        sub = data.subset(angle = sliceme)
        self.assertTrue( sub.geometry.angles[0] == data.geometry.angles[sliceme])
    def test_AcquisitionDataSubset1f(self):
        
        data = self.ag.allocate()
        #self.assertTrue( data.shape == (4,20,2,3))
        sliceme = 1
        sub = data.subset(angle = sliceme)
        self.assertTrue( sub.geometry.angles[0] == data.geometry.angles[sliceme])
        
    def test_AcquisitionDataSubset1g(self):
        
        data = self.ag_cone.allocate()
        sliceme = 1
        sub = data.subset(angle = sliceme)
        self.assertTrue( sub.geometry.angles[0] == data.geometry.angles[sliceme])       

    def test_AcquisitionDataSubset1h(self):
        
        data = self.ag_cone.allocate()
        sliceme = 1
        sub = data.subset(vertical = sliceme, force=True)
        self.assertTrue( sub.shape == (4, 3, 2))
