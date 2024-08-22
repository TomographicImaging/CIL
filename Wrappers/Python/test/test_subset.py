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
import numpy
from cil.framework import DataContainer, ImageGeometry, AcquisitionGeometry
from cil.framework.labels import ImageDimension, AcquisitionDimension
from timeit import default_timer as timer

initialise_tests()

class Test_reorder(unittest.TestCase):
    def test_DataContainer(self):
        arr = numpy.arange(0,120).reshape(2,3,4,5)
        data = DataContainer(arr, True,dimension_labels=['c','z','y','x'])
        data.reorder(['x','y','z','c'])
        self.assertEqual(data.shape,(5,4,3,2))
        numpy.testing.assert_array_equal(data.array, arr.transpose(3,2,1,0))

    def test_ImageData(self):
        ig = ImageGeometry(voxel_num_x=5, voxel_num_y=4, voxel_num_z=3, channels=2,  dimension_labels=['channel','vertical','horizontal_y','horizontal_x'])
        data = ig.allocate(None)
        new_order = ['horizontal_x', 'horizontal_y','vertical', 'channel']
        data.reorder(new_order)
        self.assertEqual(data.shape,(5,4,3,2))
        self.assertEqual(data.geometry.dimension_labels,tuple(new_order))

    def test_AcquisitionData(self):
        ag = AcquisitionGeometry.create_Parallel3D().set_panel([5,4]).set_angles([0,1,2]).set_channels(2).set_labels(['channel','angle','vertical','horizontal'])
        data = ag.allocate(None)
        new_order = ['horizontal', 'vertical','angle', 'channel']
        data.reorder(new_order)
        self.assertEqual(data.shape,(5,4,3,2))
        self.assertEqual(data.geometry.dimension_labels,tuple(new_order))

    def test_AcquisitionData_forastra(self):
        ag = AcquisitionGeometry.create_Parallel3D().set_panel([5,4]).set_angles([0,1,2]).set_channels(2).set_labels(['horizontal','vertical', 'angle', 'channel'])
        data = ag.allocate(None)

        data.reorder('astra')
        self.assertTrue(  list(data.dimension_labels) == ['channel','vertical', 'angle', 'horizontal'] )
        self.assertTrue(data.shape == (2,4,3,5) )

    def test_AcquisitionData_fortigre(self):
        ag = AcquisitionGeometry.create_Parallel3D().set_panel([5,4]).set_angles([0,1,2]).set_channels(2).set_labels(['horizontal','vertical', 'angle', 'channel'])
        data = ag.allocate(None)

        data.reorder('tigre')
        self.assertTrue(  list(data.dimension_labels) == ['channel', 'angle','vertical', 'horizontal'] )
        self.assertTrue(data.shape == (2,3,4,5))

    def test_ImageData_forastra(self):
        ig = ImageGeometry(voxel_num_x=5, voxel_num_y=4, voxel_num_z=3, channels=2,  dimension_labels=['horizontal_x','horizontal_y', 'vertical', 'channel'])
        data = ig.allocate(None)

        data.reorder('astra')
        self.assertTrue(list(data.dimension_labels) == ['channel','vertical', 'horizontal_y', 'horizontal_x'] )
        self.assertTrue(data.shape == (2,3,4,5))

    def test_ImageData_fortigre(self):
        ig = ImageGeometry(voxel_num_x=5, voxel_num_y=4, voxel_num_z=3, channels=2,  dimension_labels=['horizontal_x','horizontal_y', 'vertical', 'channel'])
        data = ig.allocate(None)

        data.reorder('tigre')
        self.assertTrue(list(data.dimension_labels) == ['channel','vertical', 'horizontal_y', 'horizontal_x'] )
        self.assertTrue(data.shape == (2,3,4,5))

    def test_reorder_with_tuple(self):
        vgeometry = ImageGeometry(voxel_num_x=4, voxel_num_y=3, channels=2)
        data = vgeometry.allocate(0)
        new_order = ('horizontal_y','horizontal_x', 'channel')
        data.reorder(new_order)
        self.assertListEqual(list(new_order), list(data.geometry.dimension_labels))
        self.assertListEqual(list(new_order), list(data.dimension_labels))

    def test_reorder_with_list(self):
        vgeometry = ImageGeometry(voxel_num_x=4, voxel_num_y=3, channels=2)
        data = vgeometry.allocate(0)
        new_order = ['horizontal_y','horizontal_x', 'channel']
        data.reorder(new_order)
        self.assertListEqual(list(new_order), list(data.geometry.dimension_labels))
        self.assertListEqual(list(new_order), list(data.dimension_labels))

    def test_reorder_with_tuple_wrong_len(self):
        vgeometry = ImageGeometry(voxel_num_x=4, voxel_num_y=3, channels=2)
        data = vgeometry.allocate(0)
        new_order = ('horizontal_y','channel')
        try:
            data.reorder(new_order)
            assert False
        except ValueError:
            assert True

    def test_reorder_with_tuple_wrong_label(self):
        vgeometry = ImageGeometry(voxel_num_x=4, voxel_num_y=3, channels=2)
        data = vgeometry.allocate(0)
        new_order = ('horizontal_y','channel','temperature')
        try:
            data.reorder(new_order)
            assert False, "Unit test should have failed! Expecting labels in {}, got {}".format(vgeometry.dimension_labels, new_order)
        except ValueError:
            assert True

    def test_reorder_with_iterable_no_len(self):
        vgeometry = ImageGeometry(voxel_num_x=4, voxel_num_y=3, channels=2)
        data = vgeometry.allocate(0)
        class Label(object):
            def __init__(self, labels):
                self.labels = labels[:]
            def __next__(self):
                return self.labels.__next__()
            def __iter__(self):
                return self
        new_order = Label(['horizontal_y','channel','horizontal_x'])
        try:
            data.reorder(new_order)
            assert False, "Unit test should have failed! Expecting len to be implemented"
        except ValueError as ve:
            assert True, ve

    def test_reorder_with_repeated_label(self):
        vgeometry = ImageGeometry(voxel_num_x=4, voxel_num_y=3, channels=2)
        data = vgeometry.allocate(0)
        new_order = ['horizontal_y','channel','horizontal_y']
        # print (len(new_order))
        try:
            data.reorder(new_order)
            assert False, "should have found a repeated label"
        except ValueError as ve:
            assert True, ve

class Test_get_slice(unittest.TestCase):
    def test_DataContainer(self):
        arr = numpy.arange(0,120).reshape(2,3,4,5)
        data = DataContainer(arr, True,dimension_labels=['c','z','y','x'])

        data_new = data.get_slice(c=1)
        self.assertEqual(data_new.shape,(3,4,5))
        numpy.testing.assert_array_equal(data_new.array, arr[1])

        data_new = data.get_slice(c=1,y=3)
        self.assertEqual(data_new.shape,(3,5))
        numpy.testing.assert_array_equal(data_new.array, arr[1,:,3,:])

        data_new = data.get_slice(c=1,y=3,z=1)
        self.assertEqual(data_new.shape,(5,))
        numpy.testing.assert_array_equal(data_new.array, arr[1,1,3,:])

    def test_ImageData(self):
        ig = ImageGeometry(voxel_num_x=5, voxel_num_y=4, voxel_num_z=3, channels=2,  dimension_labels=['channel','vertical','horizontal_y','horizontal_x'])
        data = ig.allocate(None)
        data_new = data.get_slice(vertical=1)
        self.assertEqual(data_new.shape,(2,4,5))
        self.assertEqual(data_new.geometry.dimension_labels,('channel','horizontal_y','horizontal_x'))

    def test_AcquisitionData(self):
        ag = AcquisitionGeometry.create_Parallel3D().set_panel([5,4]).set_angles([0,1,2]).set_channels(2).set_labels(['channel','angle','vertical','horizontal'])
        data = ag.allocate(None)
        data_new = data.get_slice(angle=2)
        self.assertEqual(data_new.shape,(2,4,5))
        self.assertEqual(data_new.geometry.dimension_labels,('channel','vertical','horizontal'))

        #won't return a geometry for un-reconstructable slice
        ag = AcquisitionGeometry.create_Cone3D([0,-200,0],[0,200,0]).set_panel([5,4]).set_angles([0,1,2]).set_channels(2).set_labels(['channel','angle','vertical','horizontal'])
        data = ag.allocate('random')
        data_new = data.get_slice(vertical=1,force=True)
        self.assertEqual(data_new.shape,(2,3,5))
        self.assertTrue(isinstance(data_new,(DataContainer)))
        self.assertIsNone(data_new.geometry)
        self.assertEqual(data_new.dimension_labels,('channel','angle','horizontal'))

        #if 'centre' is between pixels interpolates
        data_new = data.get_slice(vertical='centre')
        self.assertEqual(data_new.shape,(2,3,5))
        self.assertEqual(data_new.geometry.dimension_labels,('channel','angle','horizontal'))
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
        default_dimension_labels = [ImageDimension["CHANNEL"], ImageDimension["VERTICAL"],
                ImageDimension["HORIZONTAL_Y"], ImageDimension["HORIZONTAL_X"]]
        self.assertTrue( default_dimension_labels == list(data.dimension_labels) )

    def test_ImageDataAllocate1b(self):
        data = self.ig.allocate()
        self.assertTrue( data.shape == (5,4,3,2))

    def test_ImageDataAllocate2a(self):
        non_default_dimension_labels = [ ImageDimension["HORIZONTAL_X"], ImageDimension["VERTICAL"],
                ImageDimension["HORIZONTAL_Y"], ImageDimension["CHANNEL"]]
        self.ig.set_labels(non_default_dimension_labels)
        data = self.ig.allocate()
        self.assertTrue( non_default_dimension_labels == list(data.dimension_labels) )

    def test_ImageDataAllocate2b(self):
        non_default_dimension_labels = [ ImageDimension["HORIZONTAL_X"], ImageDimension["VERTICAL"],
                ImageDimension["HORIZONTAL_Y"], ImageDimension["CHANNEL"]]
        self.ig.set_labels(non_default_dimension_labels)
        data = self.ig.allocate()
        self.assertTrue( data.shape == (2,4,3,5))

    def test_ImageDataSubset1a(self):
        non_default_dimension_labels = [ImageDimension["HORIZONTAL_X"], ImageDimension["CHANNEL"], ImageDimension["HORIZONTAL_Y"],
        ImageDimension["VERTICAL"]]
        self.ig.set_labels(non_default_dimension_labels)
        data = self.ig.allocate()
        sub = data.get_slice(horizontal_y = 1)
        self.assertTrue( sub.shape == (2,5,4))

    def test_ImageDataSubset2a(self):
        non_default_dimension_labels = [ImageDimension["HORIZONTAL_X"], ImageDimension["CHANNEL"], ImageDimension["HORIZONTAL_Y"],
        ImageDimension["VERTICAL"]]
        self.ig.set_labels(non_default_dimension_labels)
        data = self.ig.allocate()
        sub = data.get_slice(horizontal_x = 1)
        self.assertTrue( sub.shape == (5,3,4))

    def test_ImageDataSubset3a(self):
        non_default_dimension_labels = [ImageDimension["HORIZONTAL_X"], ImageDimension["CHANNEL"], ImageDimension["HORIZONTAL_Y"],
        ImageDimension["VERTICAL"]]
        self.ig.set_labels(non_default_dimension_labels)
        data = self.ig.allocate()
        sub = data.get_slice(channel = 1)
        self.assertTrue( sub.shape == (2,3,4))

    def test_ImageDataSubset4a(self):
        non_default_dimension_labels = [ImageDimension["HORIZONTAL_X"], ImageDimension["CHANNEL"], ImageDimension["HORIZONTAL_Y"],
        ImageDimension["VERTICAL"]]
        self.ig.set_labels(non_default_dimension_labels)
        data = self.ig.allocate()
        sub = data.get_slice(vertical = 1)
        self.assertTrue( sub.shape == (2,5,3))

    def test_ImageDataSubset5a(self):
        non_default_dimension_labels = [ImageDimension["HORIZONTAL_X"], ImageDimension["HORIZONTAL_Y"]]
        self.ig.set_labels(non_default_dimension_labels)
        data = self.ig.allocate()
        sub = data.get_slice(horizontal_y = 1)
        self.assertTrue( sub.shape == (2,))

    def test_ImageDataSubset1b(self):
        non_default_dimension_labels = [ImageDimension["HORIZONTAL_X"], ImageDimension["CHANNEL"], ImageDimension["HORIZONTAL_Y"],
        ImageDimension["VERTICAL"]]
        self.ig.set_labels(non_default_dimension_labels)
        data = self.ig.allocate()
        new_dimension_labels = [ImageDimension["HORIZONTAL_Y"], ImageDimension["CHANNEL"], ImageDimension["VERTICAL"], ImageDimension["HORIZONTAL_X"]]
        data.reorder(new_dimension_labels)
        self.assertTrue( data.shape == (3,5,4,2))

    def test_ImageDataSubset1c(self):
        data = self.ig.allocate()
        sub = data.get_slice(channel=0,horizontal_x=0,horizontal_y=0)
        self.assertTrue( sub.shape == (4,))


    def test_AcquisitionDataAllocate1a(self):
        data = self.ag.allocate()
        default_dimension_labels = [AcquisitionDimension["CHANNEL"] ,
                 AcquisitionDimension["ANGLE"] , AcquisitionDimension["VERTICAL"] ,
                 AcquisitionDimension["HORIZONTAL"]]
        self.assertTrue(  default_dimension_labels == list(data.dimension_labels) )

    def test_AcquisitionDataAllocate1b(self):
        data = self.ag.allocate()
        self.assertTrue( data.shape == (4,3,2,20))

    def test_AcquisitionDataAllocate2a(self):
        non_default_dimension_labels = [AcquisitionDimension["CHANNEL"], AcquisitionDimension["HORIZONTAL"],
         AcquisitionDimension["VERTICAL"], AcquisitionDimension["ANGLE"]]
        self.ag.set_labels(non_default_dimension_labels)
        data = self.ag.allocate()


        self.assertTrue(  non_default_dimension_labels == list(data.dimension_labels) )

    def test_AcquisitionDataAllocate2b(self):
        non_default_dimension_labels = [AcquisitionDimension["CHANNEL"], AcquisitionDimension["HORIZONTAL"],
         AcquisitionDimension["VERTICAL"], AcquisitionDimension["ANGLE"]]
        self.ag.set_labels(non_default_dimension_labels)
        data = self.ag.allocate()
        self.assertTrue( data.shape == (4,20,2,3))

    def test_AcquisitionDataSubset1a(self):
        non_default_dimension_labels = [AcquisitionDimension["CHANNEL"], AcquisitionDimension["HORIZONTAL"],
         AcquisitionDimension["VERTICAL"], AcquisitionDimension["ANGLE"]]
        self.ag.set_labels(non_default_dimension_labels)
        data = self.ag.allocate()
        #self.assertTrue( data.shape == (4,20,2,3))
        sub = data.get_slice(vertical = 0)
        self.assertTrue( sub.shape == (4,20,3))

    def test_AcquisitionDataSubset1b(self):
        non_default_dimension_labels = [AcquisitionDimension["CHANNEL"], AcquisitionDimension["HORIZONTAL"],
         AcquisitionDimension["VERTICAL"], AcquisitionDimension["ANGLE"]]
        self.ag.set_labels(non_default_dimension_labels)
        data = self.ag.allocate()
        #self.assertTrue( data.shape == (4,20,2,3))
        sub = data.get_slice(channel = 0)
        self.assertTrue( sub.shape == (20,2,3))
    def test_AcquisitionDataSubset1c(self):
        non_default_dimension_labels = [AcquisitionDimension["CHANNEL"], AcquisitionDimension["HORIZONTAL"],
         AcquisitionDimension["VERTICAL"], AcquisitionDimension["ANGLE"]]
        self.ag.set_labels(non_default_dimension_labels)
        data = self.ag.allocate()
        #self.assertTrue( data.shape == (4,20,2,3))
        sub = data.get_slice(horizontal = 0, force=True)
        self.assertTrue( sub.shape == (4,2,3))
    def test_AcquisitionDataSubset1d(self):
        non_default_dimension_labels = [AcquisitionDimension["CHANNEL"], AcquisitionDimension["HORIZONTAL"],
         AcquisitionDimension["VERTICAL"], AcquisitionDimension["ANGLE"]]
        self.ag.set_labels(non_default_dimension_labels)
        data = self.ag.allocate()
        #self.assertTrue( data.shape == (4,20,2,3))
        sliceme = 1
        sub = data.get_slice(angle = sliceme)
        #print (sub.shape  , sub.dimension_labels)
        self.assertTrue( sub.shape == (4,20,2) )
        self.assertTrue( sub.geometry.angles[0] == data.geometry.angles[sliceme])
    def test_AcquisitionDataSubset1e(self):
        non_default_dimension_labels = [AcquisitionDimension["CHANNEL"], AcquisitionDimension["HORIZONTAL"],
         AcquisitionDimension["VERTICAL"], AcquisitionDimension["ANGLE"]]
        self.ag.set_labels(non_default_dimension_labels)
        data = self.ag.allocate()
        #self.assertTrue( data.shape == (4,20,2,3))
        sliceme = 1
        sub = data.get_slice(angle = sliceme)
        self.assertTrue( sub.geometry.angles[0] == data.geometry.angles[sliceme])
    def test_AcquisitionDataSubset1f(self):

        data = self.ag.allocate()
        #self.assertTrue( data.shape == (4,20,2,3))
        sliceme = 1
        sub = data.get_slice(angle = sliceme)
        self.assertTrue( sub.geometry.angles[0] == data.geometry.angles[sliceme])

    def test_AcquisitionDataSubset1g(self):

        data = self.ag_cone.allocate()
        sliceme = 1
        sub = data.get_slice(angle = sliceme)
        self.assertTrue( sub.geometry.angles[0] == data.geometry.angles[sliceme])

    def test_AcquisitionDataSubset1h(self):

        data = self.ag_cone.allocate()
        sub = data.get_slice(vertical = 'centre')
        self.assertTrue( sub.geometry.shape == (4,3,20))

    def test_AcquisitionDataSubset1i(self):

        data = self.ag_cone.allocate()
        sliceme = 1
        sub = data.get_slice(vertical = sliceme, force=True)
        self.assertTrue( sub.shape == (4, 3, 20))

    def test_AcquisitionDataSubset1j(self):

        data = self.ag.allocate()
        sub = data.get_slice(angle = 0)
        sub = sub.get_slice(vertical = 0)
        sub = sub.get_slice(horizontal = 0, force=True)

        self.assertTrue( sub.shape == (4,))
