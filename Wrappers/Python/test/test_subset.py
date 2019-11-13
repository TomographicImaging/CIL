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
        self.ig = ImageGeometry(1,2,3,channels=4)
        angles = numpy.asarray([90.,0.,-90.], dtype=numpy.float32)

        self.ag = AcquisitionGeometry('cone', 'edo', pixel_num_h=20, pixel_num_v=2, angles=angles, 
                         dist_source_center = 312.2, 
                         dist_center_detector = 123.,
                         channels=4 )

    def test_ImageDataAllocate1a(self):
        data = self.ig.allocate()
        default_dimension_labels = [ImageGeometry.CHANNEL, ImageGeometry.VERTICAL,
                ImageGeometry.HORIZONTAL_Y, ImageGeometry.HORIZONTAL_X]
        self.assertTrue( default_dimension_labels == list(data.dimension_labels.values()) )
    def test_ImageDataAllocate1b(self):
        data = self.ig.allocate()
        default_dimension_labels = [ImageGeometry.CHANNEL, ImageGeometry.VERTICAL,
                ImageGeometry.HORIZONTAL_Y, ImageGeometry.HORIZONTAL_X]
        self.assertTrue( data.shape == (4,3,2,1))
        
    def test_ImageDataAllocate2a(self):
        non_default_dimension_labels = [ ImageGeometry.HORIZONTAL_X, ImageGeometry.VERTICAL,
                ImageGeometry.HORIZONTAL_Y, ImageGeometry.CHANNEL]
        data = self.ig.allocate(dimension_labels=non_default_dimension_labels)
        self.assertTrue( non_default_dimension_labels == list(data.dimension_labels.values()) )
        
    def test_ImageDataAllocate2b(self):
        non_default_dimension_labels = [ ImageGeometry.HORIZONTAL_X, ImageGeometry.VERTICAL,
                ImageGeometry.HORIZONTAL_Y, ImageGeometry.CHANNEL]
        data = self.ig.allocate(dimension_labels=non_default_dimension_labels)
        self.assertTrue( data.shape == (1,3,2,4))

    def test_AcquisitionDataAllocate1a(self):
        data = self.ag.allocate()
        default_dimension_labels = [AcquisitionGeometry.CHANNEL ,
                 AcquisitionGeometry.ANGLE , AcquisitionGeometry.VERTICAL ,
                 AcquisitionGeometry.HORIZONTAL]
        self.assertTrue(  default_dimension_labels == list(data.dimension_labels.values()) )

    def test_AcquisitionDataAllocate1b(self):
        data = self.ag.allocate()
        default_dimension_labels = [AcquisitionGeometry.CHANNEL ,
                 AcquisitionGeometry.ANGLE , AcquisitionGeometry.VERTICAL ,
                 AcquisitionGeometry.HORIZONTAL]

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
        
        
            

        