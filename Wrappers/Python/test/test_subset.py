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
from cil.framework import DataContainer
from cil.framework import ImageData
from cil.framework import AcquisitionData
from cil.framework import ImageGeometry
from cil.framework import AcquisitionGeometry
from timeit import default_timer as timer

class Test_reorder(unittest.TestCase):
    def test_DataContainer(self):
        arr = numpy.arange(0,120).reshape(2,3,4,5)
        data = DataContainer(arr, True,dimension_labels=['c','z','y','x'])
        data.reorder(['x','y','z','c'])
        self.assertEquals(data.shape,(5,4,3,2))
        numpy.testing.assert_array_equal(data.array, arr.transpose(3,2,1,0))

    def test_ImageData(self):
        ig = ImageGeometry(voxel_num_x=5, voxel_num_y=4, voxel_num_z=3, channels=2,  dimension_labels=['channel','vertical','horizontal_y','horizontal_x'])
        data = ig.allocate(None)
        new_order = ['horizontal_x', 'horizontal_y','vertical', 'channel']
        data.reorder(new_order)
        self.assertEquals(data.shape,(5,4,3,2))
        self.assertEquals(data.geometry.dimension_labels,tuple(new_order))

    def test_AcquisitionData(self):
        ag = AcquisitionGeometry.create_Parallel3D().set_panel([5,4]).set_angles([0,1,2]).set_channels(2).set_labels(['channel','angle','vertical','horizontal'])
        data = ag.allocate(None)
        new_order = ['horizontal', 'vertical','angle', 'channel']
        data.reorder(new_order)
        self.assertEquals(data.shape,(5,4,3,2))
        self.assertEquals(data.geometry.dimension_labels,tuple(new_order))

class Test_get_slice(unittest.TestCase):
    def test_DataContainer(self):
        arr = numpy.arange(0,120).reshape(2,3,4,5)
        data = DataContainer(arr, True,dimension_labels=['c','z','y','x'])

        data_new = data.get_slice(c=1)
        self.assertEquals(data_new.shape,(3,4,5))
        numpy.testing.assert_array_equal(data_new.array, arr[1])

        data_new = data.get_slice(c=1,y=3)
        self.assertEquals(data_new.shape,(3,5))
        numpy.testing.assert_array_equal(data_new.array, arr[1,:,3,:])

        data_new = data.get_slice(c=1,y=3,z=1)
        self.assertEquals(data_new.shape,(5,))
        numpy.testing.assert_array_equal(data_new.array, arr[1,1,3,:])

    def test_ImageData(self):
        ig = ImageGeometry(voxel_num_x=5, voxel_num_y=4, voxel_num_z=3, channels=2,  dimension_labels=['channel','vertical','horizontal_y','horizontal_x'])
        data = ig.allocate(None)
        data_new = data.get_slice(vertical=1)
        self.assertEquals(data_new.shape,(2,4,5))
        self.assertEquals(data_new.geometry.dimension_labels,('channel','horizontal_y','horizontal_x'))

    def test_AcquisitionData(self):
        ag = AcquisitionGeometry.create_Parallel3D().set_panel([5,4]).set_angles([0,1,2]).set_channels(2).set_labels(['channel','angle','vertical','horizontal'])
        data = ag.allocate(None)
        data_new = data.get_slice(angle=2)
        self.assertEquals(data_new.shape,(2,4,5))
        self.assertEquals(data_new.geometry.dimension_labels,('channel','vertical','horizontal'))

        #won't return a geometry for un-reconstructable slice
        ag = AcquisitionGeometry.create_Cone3D([0,-200,0],[0,200,0]).set_panel([5,4]).set_angles([0,1,2]).set_channels(2).set_labels(['channel','angle','vertical','horizontal'])
        data = ag.allocate('random')
        data_new = data.get_slice(vertical=1,force=True)
        self.assertEquals(data_new.shape,(2,3,5))
        self.assertTrue(isinstance(data_new,(DataContainer)))
        self.assertIsNone(data_new.geometry)
        self.assertEquals(data_new.dimension_labels,('channel','angle','horizontal'))

        #if 'centre' is between pixels interpolates
        data_new = data.get_slice(vertical='centre')
        self.assertEquals(data_new.shape,(2,3,5))
        self.assertEquals(data_new.geometry.dimension_labels,('channel','angle','horizontal'))
        numpy.testing.assert_allclose(data_new.array, (data.array[:,:,1,:] +data.array[:,:,2,:])/2 )

class TestSubset(unittest.TestCase):
    def setUp(self):
        self.ig = ImageGeometry(2,3,4,channels=5)
        angles = numpy.asarray([90.,0.,-90.], dtype=numpy.float32)
        
        self.ag_cone = AcquisitionGeometry.create_Cone3D([0,-500,0],[0,500,0])\
                                    .set_panel((20,2))\
                                    .set_angles(angles)\
                                    .set_channels(4)

        self.ag = AcquisitionGeometry.create_Parallel3D()\
                                    .set_angles(angles)\
                                    .set_channels(4)\
                                    .set_panel((20,2))


    def test_ImageDataAllocate1a(self):
        data = self.ig.allocate()
        default_dimension_labels = [ImageGeometry.CHANNEL, ImageGeometry.VERTICAL,
                ImageGeometry.HORIZONTAL_Y, ImageGeometry.HORIZONTAL_X]
        self.assertTrue( default_dimension_labels == list(data.dimension_labels) )

    def test_ImageDataAllocate1b(self):
        data = self.ig.allocate()
        self.assertTrue( data.shape == (5,4,3,2))
        
    def test_ImageDataAllocate2a(self):
        non_default_dimension_labels = [ ImageGeometry.HORIZONTAL_X, ImageGeometry.VERTICAL,
                ImageGeometry.HORIZONTAL_Y, ImageGeometry.CHANNEL]
        self.ig.set_labels(non_default_dimension_labels)
        data = self.ig.allocate()
        self.assertTrue( non_default_dimension_labels == list(data.dimension_labels) )
        
    def test_ImageDataAllocate2b(self):
        non_default_dimension_labels = [ ImageGeometry.HORIZONTAL_X, ImageGeometry.VERTICAL,
                ImageGeometry.HORIZONTAL_Y, ImageGeometry.CHANNEL]
        self.ig.set_labels(non_default_dimension_labels)
        data = self.ig.allocate()
        self.assertTrue( data.shape == (2,4,3,5))

    def test_ImageDataSubset1a(self):
        non_default_dimension_labels = [ImageGeometry.HORIZONTAL_X, ImageGeometry.CHANNEL, ImageGeometry.HORIZONTAL_Y,
        ImageGeometry.VERTICAL]
        self.ig.set_labels(non_default_dimension_labels)
        data = self.ig.allocate()
        sub = data.subset(horizontal_y = 1)
        self.assertTrue( sub.shape == (2,5,4))

    def test_ImageDataSubset2a(self):
        non_default_dimension_labels = [ImageGeometry.HORIZONTAL_X, ImageGeometry.CHANNEL, ImageGeometry.HORIZONTAL_Y,
        ImageGeometry.VERTICAL]
        self.ig.set_labels(non_default_dimension_labels)
        data = self.ig.allocate()
        sub = data.subset(horizontal_x = 1)
        self.assertTrue( sub.shape == (5,3,4))

    def test_ImageDataSubset3a(self):
        non_default_dimension_labels = [ImageGeometry.HORIZONTAL_X, ImageGeometry.CHANNEL, ImageGeometry.HORIZONTAL_Y,
        ImageGeometry.VERTICAL]
        self.ig.set_labels(non_default_dimension_labels)
        data = self.ig.allocate()
        sub = data.subset(channel = 1)
        self.assertTrue( sub.shape == (2,3,4))

    def test_ImageDataSubset4a(self):
        non_default_dimension_labels = [ImageGeometry.HORIZONTAL_X, ImageGeometry.CHANNEL, ImageGeometry.HORIZONTAL_Y,
        ImageGeometry.VERTICAL]
        self.ig.set_labels(non_default_dimension_labels)
        data = self.ig.allocate()
        sub = data.subset(vertical = 1)
        self.assertTrue( sub.shape == (2,5,3))

    def test_ImageDataSubset5a(self):
        non_default_dimension_labels = [ImageGeometry.HORIZONTAL_X, ImageGeometry.HORIZONTAL_Y]
        self.ig.set_labels(non_default_dimension_labels)
        data = self.ig.allocate()
        sub = data.subset(horizontal_y = 1)
        self.assertTrue( sub.shape == (2,))

    def test_ImageDataSubset1b(self):
        non_default_dimension_labels = [ImageGeometry.HORIZONTAL_X, ImageGeometry.CHANNEL, ImageGeometry.HORIZONTAL_Y,
        ImageGeometry.VERTICAL]
        self.ig.set_labels(non_default_dimension_labels)
        data = self.ig.allocate()
        new_dimension_labels = [ImageGeometry.HORIZONTAL_Y, ImageGeometry.CHANNEL, ImageGeometry.VERTICAL, ImageGeometry.HORIZONTAL_X]
        sub = data.subset(dimensions=new_dimension_labels)
        self.assertTrue( sub.shape == (3,5,4,2))

    def test_ImageDataSubset1c(self):
        data = self.ig.allocate()
        sub = data.subset(channel=0,horizontal_x=0,horizontal_y=0)
        self.assertTrue( sub.shape == (4,))


    def test_AcquisitionDataAllocate1a(self):
        data = self.ag.allocate()
        default_dimension_labels = [AcquisitionGeometry.CHANNEL ,
                 AcquisitionGeometry.ANGLE , AcquisitionGeometry.VERTICAL ,
                 AcquisitionGeometry.HORIZONTAL]
        self.assertTrue(  default_dimension_labels == list(data.dimension_labels) )

    def test_AcquisitionDataAllocate1b(self):
        data = self.ag.allocate()
        self.assertTrue( data.shape == (4,3,2,20))

    def test_AcquisitionDataAllocate2a(self):
        non_default_dimension_labels = [AcquisitionGeometry.CHANNEL, AcquisitionGeometry.HORIZONTAL,
         AcquisitionGeometry.VERTICAL, AcquisitionGeometry.ANGLE]
        self.ag.set_labels(non_default_dimension_labels)
        data = self.ag.allocate()


        self.assertTrue(  non_default_dimension_labels == list(data.dimension_labels) )
        
    def test_AcquisitionDataAllocate2b(self):
        non_default_dimension_labels = [AcquisitionGeometry.CHANNEL, AcquisitionGeometry.HORIZONTAL,
         AcquisitionGeometry.VERTICAL, AcquisitionGeometry.ANGLE]
        self.ag.set_labels(non_default_dimension_labels)
        data = self.ag.allocate()
        self.assertTrue( data.shape == (4,20,2,3))

    def test_AcquisitionDataSubset1a(self):
        non_default_dimension_labels = [AcquisitionGeometry.CHANNEL, AcquisitionGeometry.HORIZONTAL,
         AcquisitionGeometry.VERTICAL, AcquisitionGeometry.ANGLE]
        self.ag.set_labels(non_default_dimension_labels)
        data = self.ag.allocate()
        #self.assertTrue( data.shape == (4,20,2,3))
        sub = data.subset(vertical = 0)
        self.assertTrue( sub.shape == (4,20,3))
    
    def test_AcquisitionDataSubset1b(self):
        non_default_dimension_labels = [AcquisitionGeometry.CHANNEL, AcquisitionGeometry.HORIZONTAL,
         AcquisitionGeometry.VERTICAL, AcquisitionGeometry.ANGLE]
        self.ag.set_labels(non_default_dimension_labels)
        data = self.ag.allocate()
        #self.assertTrue( data.shape == (4,20,2,3))
        sub = data.subset(channel = 0)
        self.assertTrue( sub.shape == (20,2,3))
    def test_AcquisitionDataSubset1c(self):
        non_default_dimension_labels = [AcquisitionGeometry.CHANNEL, AcquisitionGeometry.HORIZONTAL,
         AcquisitionGeometry.VERTICAL, AcquisitionGeometry.ANGLE]
        self.ag.set_labels(non_default_dimension_labels)
        data = self.ag.allocate()
        #self.assertTrue( data.shape == (4,20,2,3))
        sub = data.subset(horizontal = 0, force=True)
        self.assertTrue( sub.shape == (4,2,3))
    def test_AcquisitionDataSubset1d(self):
        non_default_dimension_labels = [AcquisitionGeometry.CHANNEL, AcquisitionGeometry.HORIZONTAL,
         AcquisitionGeometry.VERTICAL, AcquisitionGeometry.ANGLE]
        self.ag.set_labels(non_default_dimension_labels)
        data = self.ag.allocate()
        #self.assertTrue( data.shape == (4,20,2,3))
        sliceme = 1
        sub = data.subset(angle = sliceme)
        #print (sub.shape  , sub.dimension_labels)
        self.assertTrue( sub.shape == (4,20,2) )
        self.assertTrue( sub.geometry.angles[0] == data.geometry.angles[sliceme])
    def test_AcquisitionDataSubset1e(self):
        non_default_dimension_labels = [AcquisitionGeometry.CHANNEL, AcquisitionGeometry.HORIZONTAL,
         AcquisitionGeometry.VERTICAL, AcquisitionGeometry.ANGLE]
        self.ag.set_labels(non_default_dimension_labels)
        data = self.ag.allocate()
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
        sub = data.subset(vertical = 'centre')
        self.assertTrue( sub.geometry.shape == (4,3,20))       

    def test_AcquisitionDataSubset1i(self):
        
        data = self.ag_cone.allocate()
        sliceme = 1
        sub = data.subset(vertical = sliceme, force=True)
        self.assertTrue( sub.shape == (4, 3, 20))

    def test_AcquisitionDataSubset1j(self):

        data = self.ag.allocate()
        sub = data.subset(angle = 0)
        sub = sub.subset(vertical = 0)
        sub = sub.subset(horizontal = 0, force=True)

        self.assertTrue( sub.shape == (4,))

    def test_AcquisitionDataSubset_forastra(self):

        self.ag.set_labels(['horizontal','vertical', 'angle', 'channel'])
        new_ag = self.ag.subset('astra')
        self.assertTrue(  list(new_ag.dimension_labels) == ['channel','vertical', 'angle', 'horizontal'] )

        ad = self.ag.allocate()
        new_ad = self.ag.subset('astra')
        self.assertTrue(new_ad.shape == (4, 2, 3, 20) )

    def test_AcquisitionDataSubset_fortigre(self):

        self.ag.set_labels(['horizontal','vertical', 'angle', 'channel'])
        new_ag = self.ag.subset('tigre')
        self.assertTrue(  list(new_ag.dimension_labels) == ['channel','angle', 'vertical', 'horizontal'] )

        ad = self.ag.allocate()
        new_ad = self.ag.subset('tigre')
        self.assertTrue(new_ad.shape == (4, 3, 2, 20) )

    def test_ImageDataSubset_forastra(self):

        self.ig.set_labels(['horizontal_x','horizontal_y', 'vertical', 'channel'])
        new_ig = self.ig.subset('astra')
        self.assertTrue(list(new_ig.dimension_labels) == ['channel','vertical', 'horizontal_y', 'horizontal_x'] )

        id = self.ig.allocate()
        new_id = self.ig.subset('astra')
        self.assertTrue(new_id.shape == (5,4,3,2) )

    def test_ImageDataSubset_fortigre(self):

        self.ig.set_labels(['horizontal_x','horizontal_y', 'vertical', 'channel'])
        new_ig = self.ig.subset('tigre')
        self.assertTrue(list(new_ig.dimension_labels) == ['channel','vertical', 'horizontal_y', 'horizontal_x'] )

        id = self.ig.allocate()
        new_id = self.ig.subset('tigre')
        self.assertTrue(new_id.shape == (5,4,3,2) )