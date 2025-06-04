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
import functools
import logging
import sys

import array_api_compat.numpy
import numpy as np

from cil.framework import (DataContainer, ImageGeometry, ImageData, VectorGeometry, AcquisitionData,
                           AcquisitionGeometry, BlockGeometry, VectorData)
from cil.framework.labels import ImageDimension, AcquisitionDimension

from testclass import CCPiTestClass
from utils import initialise_tests

import array_api_compat
from unittest_parametrize import param, parametrize, ParametrizedTestCase
import numpy
try:
    import torch
except ImportError:
    torch = None
import unittest

log = logging.getLogger(__name__)
initialise_tests()


def aid(x):
    """returns the memory block address of an array"""
    return x.as_array().__array_interface__['data'][0]


class TestDataContainer(ParametrizedTestCase, CCPiTestClass):
    def create_DataContainer(self, X,Y,Z, value=1, backend='numpy'):
        a = value * np.ones((X, Y, Z), dtype='float32')
        #print("a refcount " , sys.getrefcount(a))
        ds = DataContainer(a, False, ['X', 'Y', 'Z'])
        return ds
    
    @parametrize("xp, raise_error, err_type", 
        [param(numpy, None, None, id="numpy"), 
         param(torch, None, None, id="torch"),
         ]) 
    def test_creation_nocopy(self, xp, raise_error, err_type):
        if xp is None:
            self.skipTest("torch not available")
        shape = 2, 3, 4, 5
        size = xp.prod(xp.asarray(shape))
        a = xp.arange(size)
        #print("a refcount " , sys.getrefcount(a))
        a = xp.reshape(a, shape)
        #print("a refcount " , sys.getrefcount(a))
        ds = DataContainer(a, False, ['X', 'Y', 'Z', 'W'])
        #print("a refcount " , sys.getrefcount(a))
        self.assertEqual(id(a), id(ds.array))
        self.assertEqual(ds.dimension_labels, ('X', 'Y', 'Z', 'W'))

    def testGb_creation_nocopy(self):
        X, Y, Z = 8, 64, 64
        a = np.ones((X, Y, Z), dtype='float32')
        #print("a refcount " , sys.getrefcount(a))
        ds = DataContainer(a, False, ['X', 'Y', 'Z'])
        #print("a refcount " , sys.getrefcount(a))
        self.assertEqual(sys.getrefcount(a), 3)
        ds1 = ds.copy()
        self.assertNotEqual(aid(ds), aid(ds1))
        ds1 = ds.clone()
        self.assertNotEqual(aid(ds), aid(ds1))

    @parametrize("xp, raise_error, err_type", 
        [param(numpy, None, None, id="numpy"), 
         param(torch, None, None, id="torch"),
         ]) 
    def test_ndim(self, xp, raise_error, err_type):
        if xp is None:
            self.skipTest("torch not available")
        x_np = xp.arange(0, 60).reshape(3,4,5)
        x_cil = DataContainer(x_np)
        self.assertEqual(x_np.ndim, x_cil.ndim)
        self.assertEqual(3, x_cil.ndim)

    @parametrize("xp, raise_error, err_type", 
        [param(numpy, None, None, id="numpy"), 
         param(torch, None, None, id="torch"),
         ]) 
    def test_DataContainer_equal(self, xp, raise_error, err_type):
        if xp is None:
            self.skipTest("torch not available")
        array = xp.linspace(-1, 1, 32, dtype=xp.float32).reshape(4, 8)
        data = DataContainer(array)
        data1 = data.copy()

        # Check two identical DataContainers are equal
        self.assertTrue((data == data1).all())
        # Check it works for comparing DataContainer with numpy array
        self.assertTrue((data == data1.as_array()).all())

        # # Check two different DataContainers are not equal
        data1 += 1
        self.assertFalse((data ==  data1).all())

    @parametrize("xp, raise_error, err_type", 
        [param(numpy, None, None, id="numpy"), 
         param(torch, None, None, id="torch"),
         ]) 
    def test_AcquisitionData_equal(self, xp, raise_error, err_type):
        if xp is None:
            self.skipTest("torch not available")
        array = xp.linspace(-1, 1, 32, dtype=xp.float32).reshape(4, 8)
        geom = AcquisitionGeometry.create_Parallel3D().set_panel((8, 4)).set_channels(1).set_angles([1])
        data = AcquisitionData(array, geometry=geom)

        data1 = data.copy()

        # Check two identical AcquisitionData are equal
        self.assertTrue(data == data1)
        # Check it works for comparing AcquisitionData with numpy array
        self.assertTrue(data == data1.as_array())

        # # Check two different AcquisitionData are not equal
        data1 += 1
        self.assertFalse(data ==  data1)

        # Check the equality of two AcquisitionData with different shapes
        data_different_shape = data.copy()
        data_different_shape.array = data_different_shape.array.reshape(8, 4)

        self.assertFalse(data == data_different_shape)

        # Check the equality of two AcquisitionData with different dtypes
        data_different_dtype = data.copy()
        # data_different_dtype.array = data_different_dtype.array.astype(np.float64)
        arr = xp.asarray(data_different_dtype.array, dtype=xp.float64)
        self.assertFalse(data == arr)


        # Check the equality of two AcquisitionData with different labels
        data_different_labels = data.copy()
        print(data_different_labels.geometry.dimension_labels)
        data_different_labels.geometry.set_labels([AcquisitionDimension("ANGLE"), AcquisitionDimension("CHANNEL") ])
        self.assertFalse(data == data_different_labels)

    @parametrize("xp, device, raise_error, err_type", 
        [param(numpy, None, None, None, id="numpy"), 
         param(torch, None, None, None, id="torch_cpu"),
         ]) 
    def test_ImageData_equal(self, xp, device, raise_error, err_type):
        if xp is None:
            self.skipTest("torch not available")
        array = xp.linspace(-1, 1, 32, dtype=xp.float32).reshape(4, 8)
        geom = ImageGeometry(voxel_num_x=8, voxel_num_y=4)
        data = ImageData(array, geometry=geom)

        data1 = data.copy()

        # Check two identical ImageData are equal
        self.assertTrue(data == data1)
        # Check it works for comparing ImageData with numpy array
        self.assertTrue(data == data1.as_array())

        # # Check two different ImageData are not equal
        data1 += 1
        self.assertFalse(data ==  data1)

        # Check the equality of two ImageData with different shapes
        data_different_shape = data.copy()
        data_different_shape.array = data_different_shape.array.reshape(8, 4)

        self.assertFalse(data == data_different_shape)

        # Check the equality of two ImageData with different dtypes
        data_different_dtype = data.geometry.allocate(0, dtype=np.float64)
        self.assertFalse(data == data_different_dtype)


        # Check the equality of two ImageData with different labels
        data_different_labels = data.copy()
        data_different_labels.geometry.set_labels([ImageDimension("VERTICAL"), ImageDimension("HORIZONTAL_X")])
        self.assertFalse(data == data_different_labels)

    @parametrize("xp, device, raise_error, err_type", 
        [param(numpy, None, None, None, id="numpy"), 
         param(torch, None, None, None, id="torch_cpu"),
         ]) 
    def testInlineAlgebra(self, xp, device, raise_error, err_type):
        if xp is None:
            self.skipTest("torch not available")
        X, Y, Z = 8, 16, 32
        a = np.ones((X, Y, Z), dtype='float32')
        b = np.ones((X, Y, Z), dtype='float32')
        ds = DataContainer(a, False, ['X', 'Y', 'Z'])
        ds += 2
        # self.assertEqual(ds.as_array()[0][0][0], 3.)
        np.testing.assert_array_almost_equal(ds.as_array(), 3 * b)
        ds -= 2
        # self.assertEqual(ds.as_array()[0][0][0], 1.)
        np.testing.assert_array_almost_equal(ds.as_array(), b)

        ds *= 2
        # self.assertEqual(ds.as_array()[0][0][0], 2.)
        np.testing.assert_array_almost_equal(ds.as_array(), b * 2)

        ds /= 2
        # self.assertEqual(ds.as_array()[0][0][0], 1.)
        np.testing.assert_array_almost_equal(ds.as_array(), b)

        ds1 = ds.copy()
        ds1 += 1
        ds += ds1
        np.testing.assert_array_almost_equal(ds.as_array(), 3 * b)
        # self.assertEqual(ds.as_array()[0][0][0], 3.)
        ds -= ds1
        np.testing.assert_array_almost_equal(ds.as_array(), b)
        # self.assertEqual(ds.as_array()[0][0][0], 1.)
        ds *= ds1
        np.testing.assert_array_almost_equal(ds.as_array(), 2 * b)
        # self.assertEqual(ds.as_array()[0][0][0], 2.)
        ds /= ds1
        np.testing.assert_array_almost_equal(ds.as_array(), b)
        # self.assertEqual(ds.as_array()[0][0][0], 1.)

    @parametrize("xp, device, raise_error, err_type", 
        [param(numpy, None, None, None, id="numpy"), 
         param(torch, None, None, None, id="torch_cpu"),
         ])
    def test_unary_operations(self, xp, device, raise_error, err_type):
        if xp is None:
            self.skipTest("torch not available")
        X, Y, Z = 8, 16, 32
        a = -xp.ones((X, Y, Z), dtype=xp.float32)
        b = xp.ones((X, Y, Z), dtype=xp.float32)
        ds = DataContainer(a, False, ['X', 'Y', 'Z'])

        ds.sign(out=ds)
        # self.assertEqual(ds.as_array()[0][0][0], -1.)
        np.testing.assert_array_almost_equal(ds.as_array(), -b)


        ds.abs(out=ds)
        np.testing.assert_array_almost_equal(ds.as_array(), b)
        # self.assertEqual(ds.as_array()[0][0][0], 1.)

        ds *= 2
        ds.sqrt(out=ds)
        np.testing.assert_array_almost_equal(ds.as_array(),np.sqrt(2) * b)

        # self.assertEqual(ds.as_array()[0][0][0],
        #                  np.sqrt(2., dtype='float32'))

    def test_binary_add(self):
        X, Y, Z = 8, 16, 32
        a = np.ones((X, Y, Z), dtype='float32')
        b = np.ones((X, Y, Z), dtype='float32')

        ds = DataContainer(a, False, ['X', 'Y', 'Z'])
        ds1 = ds.copy()
        out = ds.copy()

        ds.add(ds1, out=out)
        ds2 = ds.add(ds1)

        self.assertNumpyArrayEqual(out.as_array(), ds2.as_array())

        ds0 = ds
        ds0.add(2, out=out)
        # self.assertEqual(3., out.as_array()[0][0][0])
        np.testing.assert_array_almost_equal(out.as_array(), 3* b)


        ds3 = ds0.add(2)
        self.assertNumpyArrayEqual(out.as_array(), ds3.as_array())

    def test_binary_subtract(self):
        X, Y, Z = 8, 16, 32
        a = np.ones((X, Y, Z), dtype='float32')
        b = np.ones((X, Y, Z), dtype='float32')

        ds = DataContainer(a, False, ['X', 'Y', 'Z'])
        ds1 = ds.copy()
        out = ds.copy()


        ds.subtract(ds1, out=out)
        ds2 = out.subtract(ds1)
        self.assertNumpyArrayEqual(out.as_array(), b * 0)
        # self.assertEqual(0., out.as_array()[0][0][0])
        np.testing.assert_array_almost_equal(ds2.as_array(), -b)
        # self.assertEqual(-1., ds2.as_array()[0][0][0])
        # self.assertLess(t1 - t0, t2 - t1)

        del ds1
        ds0 = ds.copy()
        ds0.subtract(2, out=ds0)
        # self.assertEqual(-1., ds0.as_array()[0][0][0])
        np.testing.assert_array_almost_equal(ds0.as_array(), - b)


        ds3 = ds0.subtract(2)
        # self.assertEqual(-1., ds0.as_array()[0][0][0])
        np.testing.assert_array_almost_equal(ds0.as_array(), - b)

        # self.assertEqual(-3., ds3.as_array()[0][0][0])
        np.testing.assert_array_almost_equal(ds3.as_array(), -3 * b)

    def test_binary_multiply(self):
        X, Y, Z = 8, 16, 32
        a = np.ones((X, Y, Z), dtype='float32')
        b = np.ones((X, Y, Z), dtype='float32')
        ds = DataContainer(a, False, ['X', 'Y', 'Z'])
        ds1 = ds.copy()

        ds.multiply(ds1, out=ds)
        ds2 = ds.multiply(ds1)
        np.testing.assert_array_almost_equal(ds.as_array(), ds2.as_array())

        ds0 = ds
        ds0.multiply(2, out=ds0)
        # self.assertEqual(2., ds0.as_array()[0][0][0])
        np.testing.assert_array_almost_equal(ds0.as_array(), b * 2)

        ds3 = ds0.multiply(2)
        # self.assertEqual(4., ds3.as_array()[0][0][0])
        np.testing.assert_array_almost_equal(ds3.as_array(), 4 * b)

        # self.assertEqual(2., ds.as_array()[0][0][0])
        np.testing.assert_array_almost_equal(ds.as_array(), 2 * b)

        ds.multiply(2.5, out=ds0)
        # self.assertEqual(2.5*2., ds0.as_array()[0][0][0])
        np.testing.assert_array_almost_equal(ds.as_array(), 5 * b)

    def test_binary_divide(self):
        X, Y, Z = 8, 16, 32
        a = np.ones((X, Y, Z), dtype='float32')
        b = np.ones((X, Y, Z), dtype='float32')

        ds = DataContainer(a, False, ['X', 'Y', 'Z'])
        ds1 = ds.copy()

        ds.divide(ds1, out=ds)
        # self.assertEqual(ds.as_array()[0][0][0], 1.)
        np.testing.assert_array_almost_equal(ds.as_array(), b)

        ds0 = ds
        ds0.divide(2, out=ds0)
        # self.assertEqual(0.5, ds0.as_array()[0][0][0])
        np.testing.assert_array_almost_equal(ds0.as_array(), 0.5 * b)


        ds3 = ds0.divide(2)
        # self.assertEqual(.25, ds3.as_array()[0][0][0])
        np.testing.assert_array_almost_equal(ds3.as_array(), 0.25 * b)

        # self.assertEqual(.5, ds.as_array()[0][0][0])
        np.testing.assert_array_almost_equal(ds.as_array(), 0.5 * b)


    def test_reverse_operand_algebra(self):
        number = 3/2

        X, Y, Z = 32, 64, 128
        a = np.ones((X, Y, Z), dtype='float32')
        ds = DataContainer(a * 3, False, ['X', 'Y', 'Z'])

        # rdiv
        b = number / ds
        np.testing.assert_array_almost_equal(a * 0.5, b.as_array())
        # radd
        number = 1
        b = number + ds
        np.testing.assert_array_almost_equal(a * 4, b.as_array())
        # rsub
        number = 3
        b = number - ds
        np.testing.assert_array_almost_equal(np.zeros_like(a), b.as_array())
        # rmul
        number = 1/3
        b = number * ds
        np.testing.assert_array_almost_equal(a, b.as_array())
        # rpow
        number = 2
        b = number ** ds
        np.testing.assert_array_almost_equal(a * 8, b.as_array())

    def test_creation_copy(self):
        shape = 2, 3, 4, 5
        size = np.prod(shape)
        a = np.arange(size)
        #print("a refcount " , sys.getrefcount(a))
        a = np.reshape(a, shape)
        #print("a refcount " , sys.getrefcount(a))
        ds = DataContainer(a, True, ['X', 'Y', 'Z', 'W'])
        #print("a refcount " , sys.getrefcount(a))
        self.assertEqual(sys.getrefcount(a), 2)

    def test_dot(self):
        a0 = np.arange(2*3*4)
        a1 = a0 * 2

        ds0 = DataContainer(np.reshape(a0,(2,3,4)))
        ds1 = DataContainer(np.reshape(a1,(2,3,4)))

        np.testing.assert_equal(ds0.dot(ds1), a0.dot(a1))

        a2 = np.arange(0, 2*3*5*2, 2)
        ds2 = DataContainer(np.reshape(a2,(2,3,5)))

        # it should fail if the shape is wrong
        try:
            ds2.dot(ds0)
            self.assertTrue(False)
        except ValueError as ve:
            self.assertTrue(True)

        n0 = (ds0 * ds1).sum()
        n1 = ds0.as_array().ravel().dot(ds1.as_array().ravel())
        self.assertEqual(n0, n1)

    def test_exp_log(self):
        a0 = np.ones(2*3*4)

        ds0 = DataContainer(np.reshape(a0,(2,3,4)), suppress_warning=True)
        # ds1 = DataContainer(np.reshape(a1,(2,3,4)), suppress_warning=True)
        b = ds0.exp().log()
        np.testing.assert_allclose(ds0.as_array(), b.as_array())

        self.assertEqual(ds0.exp().as_array()[0][0][0], np.exp(1))
        self.assertEqual(ds0.log().as_array()[0][0][0], 0.)

    def test_ImageData(self):
        # create ImageData from geometry
        vgeometry = ImageGeometry(voxel_num_x=4, voxel_num_y=3, channels=2)
        #vol = ImageData(geometry=vgeometry)
        vol = vgeometry.allocate()
        self.assertEqual(vol.shape, (2, 3, 4))

        vol1 = vol + 1
        self.assertNumpyArrayEqual(vol1.as_array(), np.ones(vol.shape))

        vol1 = vol - 1
        self.assertNumpyArrayEqual(vol1.as_array(), -np.ones(vol.shape))

        vol1 = 2 * (vol + 1)
        self.assertNumpyArrayEqual(vol1.as_array(), 2 * np.ones(vol.shape))

        vol1 = (vol + 1) / 2
        self.assertNumpyArrayEqual(vol1.as_array(), np.ones(vol.shape) / 2)

        vol1 = vol + 1
        self.assertEqual(vol1.sum(), 2*3*4)
        vol1 = (vol + 2) ** 2
        self.assertNumpyArrayEqual(vol1.as_array(), np.ones(vol.shape) * 4)

        self.assertEqual(vol.number_of_dimensions, 3)

        ig2 = ImageGeometry (voxel_num_x=2,voxel_num_y=3,voxel_num_z=4,
                     dimension_labels=[ImageDimension["HORIZONTAL_X"], ImageDimension["HORIZONTAL_Y"],
                 ImageDimension["VERTICAL"]])
        data = ig2.allocate()
        self.assertNumpyArrayEqual(np.asarray(data.shape), np.asarray(ig2.shape))
        self.assertNumpyArrayEqual(np.asarray(data.shape), data.as_array().shape)

    def test_ImageData_from_numpy(self):
        """
        Test the creation and manipulation of ImageData from a numpy array.
        """
        # Initialize array and ImageGeometry
        arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        ig = ImageGeometry(voxel_num_x=4, voxel_num_y=3, voxel_num_z=2)
        data = ImageData(arr, deep_copy=True, geometry=ig)

        # Check initial shape, geometry, and deep copy
        self.assertEqual(data.shape, (2, 3, 4))
        self.assertEqual(data.geometry, ig)
        self.assertNotEqual(id(data.array), id(arr))

        # Fill with zeros and check
        data.fill(0)
        np.testing.assert_array_equal(data.as_array(), np.zeros(ig.shape))

        # Fill with original array and check
        data.fill(arr)
        self.assertEqual(data.shape, (2, 3, 4))
        self.assertEqual(data.geometry, ig)

        # Reshape array with singleton dimension and create new ImageData
        arr = arr.reshape(1, 2, 3, 4)
        data = ImageData(arr, geometry=ig)
        self.assertEqual(data.shape, (2, 3, 4))
        self.assertEqual(data.geometry, ig)

        # Fill with zeros and reshaped array, then check
        data.fill(0)
        data.fill(arr)
        self.assertEqual(data.shape, (2, 3, 4))
        self.assertEqual(data.geometry, ig)

    def test_ImageData_apply_circular_mask(self):
        ig = ImageGeometry(voxel_num_x=6, voxel_num_y=3, voxel_size_x=0.5, voxel_size_y = 1)
        data_orig = ig.allocate(1)

        data_masked1 = data_orig.copy()
        data_masked1.apply_circular_mask(0.8)

        self.assertEqual(data_orig.geometry, data_masked1.geometry)
        self.assertEqual(np.count_nonzero(data_masked1.array), 14)

        data_masked1 = data_orig.copy()
        data_masked1.apply_circular_mask(0.5)

        self.assertEqual(data_orig.geometry, data_masked1.geometry)
        self.assertEqual(np.count_nonzero(data_masked1.array), 8)

        data2 = data_orig.copy()
        data_masked2 = data2.apply_circular_mask(0.5, False)

        np.testing.assert_allclose(data_orig.array, data2.array)
        self.assertEqual(np.count_nonzero(data_masked2.array), 8)



    def test_AcquisitionData(self):
        sgeometry = AcquisitionGeometry.create_Parallel3D().set_angles(np.linspace(0, 180, num=10)).set_panel((5,3)).set_channels(2)

        #sino = AcquisitionData(geometry=sgeometry)
        sino = sgeometry.allocate()
        self.assertEqual(sino.shape, (2, 10, 3, 5))

        ag = AcquisitionGeometry.create_Parallel3D().set_angles(np.linspace(0, 180, num=10)).set_panel((2,3)).set_channels(4)
        data = ag.allocate()
        self.assertNumpyArrayEqual(np.asarray(data.shape), np.asarray(ag.shape))
        self.assertNumpyArrayEqual(np.asarray(data.shape), data.as_array().shape)

        ag2 = AcquisitionGeometry.create_Parallel3D().set_angles(np.linspace(0, 180, num=10)).set_panel((2,3)).set_channels(4)\
                                 .set_labels([AcquisitionDimension["VERTICAL"] ,
                         AcquisitionDimension["ANGLE"], AcquisitionDimension["HORIZONTAL"], AcquisitionDimension["CHANNEL"]])

        data = ag2.allocate()
        self.assertNumpyArrayEqual(np.asarray(data.shape), np.asarray(ag2.shape))
        self.assertNumpyArrayEqual(np.asarray(data.shape), data.as_array().shape)

    def test_AcquisitionData_from_numpy(self):
        """
        Test the creation and manipulation of AcquisitionData from a numpy array.
        """
        arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        ag = AcquisitionGeometry.create_Parallel3D().set_angles([0, 1]).set_panel((4, 3))
        data = AcquisitionData(arr, deep_copy=True, geometry=ag)

        # Check initial shape, geometry, and deep copy
        self.assertEqual(data.shape, (2, 3, 4))
        self.assertEqual(data.geometry, ag)
        self.assertNotEqual(id(data.array), id(arr))

        # Fill with zeros and check
        data.fill(0)
        np.testing.assert_array_equal(data.as_array(), np.zeros(ag.shape))

        # Fill with original array and check
        data.fill(arr)
        self.assertEqual(data.shape, (2, 3, 4))
        self.assertEqual(data.geometry, ag)
        np.testing.assert_array_equal(data.as_array(), arr)

        # Reshape array to add empty dimensions and test create AcquisitionData
        arr = arr.reshape(1, 2, 3, 4)
        data = AcquisitionData(arr, geometry=ag)
        self.assertEqual(data.shape, (2, 3, 4))
        self.assertEqual(data.geometry, ag)

        # Fill with reshaped array
        data.fill(0)
        data.fill(arr)
        self.assertEqual(data.shape, (2, 3, 4))
        self.assertEqual(data.geometry, ag)

        # Change dtype to float64, fill, and check
        arr = arr.astype(np.float64)
        data.fill(arr)
        self.assertEqual(data.shape, (2, 3, 4))
        self.assertEqual(data.geometry, ag)
        self.assertEqual(data.dtype, np.float32)


    def test_ImageGeometry_allocate(self):
        vgeometry = ImageGeometry(voxel_num_x=4, voxel_num_y=3, channels=2)
        image = vgeometry.allocate()
        self.assertEqual(0,image.as_array()[0][0][0])
        image = vgeometry.allocate(1)
        self.assertEqual(1,image.as_array()[0][0][0])
        default_order = ['channel' , 'horizontal_y' , 'horizontal_x']
        self.assertEqual(default_order[0], image.dimension_labels[0])
        self.assertEqual(default_order[1], image.dimension_labels[1])
        self.assertEqual(default_order[2], image.dimension_labels[2])
        order = [ 'horizontal_x' , 'horizontal_y', 'channel' ]
        vgeometry.set_labels(order)
        image = vgeometry.allocate(0)
        self.assertEqual(order[0], image.dimension_labels[0])
        self.assertEqual(order[1], image.dimension_labels[1])
        self.assertEqual(order[2], image.dimension_labels[2])

        ig = ImageGeometry(2,3,2)
        try:
            z = ImageData(np.random.randint(10, size=(2,3)), geometry=ig)
            self.assertTrue(False)
        except ValueError as ve:
            log.info(str(ve))
            self.assertTrue(True)

        #vgeometry.allocate('')
    def test_AcquisitionGeometry_allocate(self):
        ageometry = AcquisitionGeometry.create_Parallel3D().set_angles(np.linspace(0, 180, num=10)).set_panel((5,3)).set_channels(2)
        sino = ageometry.allocate(0)
        shape = sino.shape
        self.assertAlmostEqual(0.,sino.as_array()[0][0][0][0])
        self.assertAlmostEqual(0.,sino.as_array()[shape[0]-1][shape[1]-1][shape[2]-1][shape[3]-1])

        sino = ageometry.allocate(1)
        self.assertEqual(1,sino.as_array()[0][0][0][0])
        self.assertEqual(1,sino.as_array()[shape[0]-1][shape[1]-1][shape[2]-1][shape[3]-1])

        default_order = ['channel' , 'angle' ,
                         'vertical' , 'horizontal']
        self.assertEqual(default_order[0], sino.dimension_labels[0])
        self.assertEqual(default_order[1], sino.dimension_labels[1])
        self.assertEqual(default_order[2], sino.dimension_labels[2])
        self.assertEqual(default_order[3], sino.dimension_labels[3])
        order = ['vertical' , 'horizontal', 'channel' , 'angle' ]
        ageometry.set_labels(order)
        sino = ageometry.allocate(0)

        self.assertEqual(order[0], sino.dimension_labels[0])
        self.assertEqual(order[1], sino.dimension_labels[1])
        self.assertEqual(order[2], sino.dimension_labels[2])
        self.assertEqual(order[2], sino.dimension_labels[2])

        try:
            z = AcquisitionData(np.random.randint(10, size=(2,3)), geometry=ageometry)
            self.assertTrue(False)
        except ValueError as ve:
            log.info(str(ve))
            self.assertTrue(True)


    def test_BlockGeometry_allocate_dtype(self):
        ig1 = ImageGeometry(3,3)
        ig2 = ImageGeometry(3,3, dtype=np.int16)
        bg = BlockGeometry(ig1,ig2)

        # print("The default dtype of the BlockImageGeometry is {}".format(bg.dtype))
        self.assertEqual(bg.dtype, (np.float32, np.int16))


    def dtype_allocate_test(self, geometry):
        classname = geometry.__class__.__name__
        # print("The default dtype of the {} is {}".format(classname , geometry.dtype))
        self.assertEqual(geometry.dtype, np.float32)

        #print("Change it to complex")
        geometry.dtype = np.complex64
        self.assertEqual(geometry.dtype, np.complex64)

        geometry.dtype = np.complex128
        self.assertEqual(geometry.dtype, np.complex128)

        geometry.dtype = complex
        self.assertEqual(geometry.dtype, complex)

        #print("Test {} allocate".format(classname ))
        data = geometry.allocate()
        #print("Data dtype is now {} ".format(geometry.dtype))
        self.assertEqual(data.dtype, geometry.dtype)
        #print("Data geometry dtype is now")
        #print(data.geometry.dtype)
        self.assertEqual(data.geometry.dtype, geometry.dtype)

        #print("Allocate data with different dtype, e.g: np.int64 from the same {}".format(classname ))
        data = geometry.allocate(dtype=np.int64)
        self.assertEqual(data.dtype, np.int64)
        #print("Data dtype is now {}".format(data.dtype))
        #print("Data geometry dtype is now {}".format(data.geometry.dtype))
        self.assertEqual(data.geometry.dtype, np.int64)
        self.assertEqual(data.dtype, np.int64)

        #print("The dtype of the {} remain unchanged ig.dtype =  {}".format(classname, geometry.dtype))
        self.assertEqual(geometry.dtype, complex)

        self.assertNotEqual(id(geometry), id(data.geometry))


    def test_ImageGeometry_allocate_dtype(self):
        #print("Test ImageGeometry dtype\n")
        ig = ImageGeometry(3,3)
        self.dtype_allocate_test(ig)


    def test_AcquisitionGeometry_allocate_dtype(self):
        # print("Test AcquisitionGeometry dtype\n")
        # Detectors
        detectors =  10

        # Angles
        angles = np.linspace(0,180,180, dtype='float32')

        # Setup acquisition geometry
        ag = AcquisitionGeometry.create_Parallel2D()\
                                .set_angles(angles)\
                                .set_panel(detectors, pixel_size=0.1)
        self.dtype_allocate_test(ag)


    def test_VectorGeometry_allocate_dtype(self):
        # print("Test VectorGeometry dtype\n")

        vg = VectorGeometry(3)
        self.dtype_allocate_test(vg)


    def complex_allocate_geometry_test(self, geometry):
        data = geometry.allocate(dtype=np.complex64)
        from array_api_compat import array_namespace
        xp = array_namespace(data.array)
        r = (1 + 1j*1)* xp.ones(data.shape, dtype=data.dtype)
        data.fill(r)
        self.assertAlmostEqual(data.squared_norm(), data.size * 2)
        np.testing.assert_almost_equal(data.abs().array, np.abs(r))

        data1 = geometry.allocate(dtype=np.float32)
        try:
            data1.fill(r)
            self.assertTrue(False)
        # except numpy.exceptions.ComplexWarning as err:
        except Exception as err:
            log.info(str(err))
            self.assertTrue(True)


    def test_ImageGeometry_allocate_complex(self):
        ig = ImageGeometry(2,2)
        self.complex_allocate_geometry_test(ig)

    def test_ImageGeometry_allocate_random_complex(self):
        ig = ImageGeometry(2,2)
        data=ig.allocate('random', dtype=np.complex64)
        self.assertTrue(data.array.dtype, np.complex64)
        self.assertNotEqual(np.sum(data.array).imag, 0)

        ig = ImageGeometry(2,2)
        data=ig.allocate('random_int', dtype=np.complex64)
        self.assertTrue(data.array.dtype, np.complex64)
        self.assertNotEqual(np.sum(data.array).imag, 0)



    def test_AcquisitionGeometry_allocate_complex(self):
        # Detectors
        detectors =  10

        # Angles
        angles = np.linspace(0,10,10, dtype='float32')

        # Setup acquisition geometry
        ag = AcquisitionGeometry.create_Parallel2D()\
                                .set_angles(angles)\
                                .set_panel(detectors, pixel_size=0.1)

        self.complex_allocate_geometry_test(ag)

    def test_AcquisitionGeometry_allocate_random_complex(self):

        detectors =  10

        # Angles
        angles = np.linspace(0,10,10, dtype='float32')

        # Setup acquisition geometry
        ag = AcquisitionGeometry.create_Parallel2D()\
                                .set_angles(angles)\
                                .set_panel(detectors, pixel_size=0.1)


        data=ag.allocate('random', dtype=np.complex64)
        self.assertTrue(data.array.dtype, np.complex64)
        self.assertNotEqual(np.sum(data.array).imag, 0)


        data=ag.allocate('random_int', dtype=np.complex64)
        self.assertTrue(data.array.dtype, np.complex64)
        self.assertNotEqual(np.sum(data.array).imag, 0)



    def test_VectorGeometry_allocate_complex(self):
        vg = VectorGeometry(3)
        self.complex_allocate_geometry_test(vg)

        data=vg.allocate('random', dtype=np.complex64)
        self.assertTrue(data.array.dtype, np.complex64)
        self.assertNotEqual(np.sum(data.array).imag, 0)

        data=vg.allocate('random_int', dtype=np.complex64)
        self.assertTrue(data.array.dtype, np.complex64)
        self.assertNotEqual(np.sum(data.array).imag, 0)


    def test_ImageGeometry_allocate_random_same_seed(self):
        vgeometry = ImageGeometry(voxel_num_x=4, voxel_num_y=3, channels=2)
        image1 = vgeometry.allocate('random', seed=0)
        image2 = vgeometry.allocate('random', seed=0)
        np.testing.assert_allclose(image1.as_array(), image2.as_array())


    def test_AcquisitionDataSubset(self):
        sgeometry = AcquisitionGeometry.create_Parallel3D().set_angles(np.linspace(0, 180, num=10)).set_panel((5,3)).set_channels(2)

        # expected dimension_labels

        self.assertListEqual([AcquisitionDimension["CHANNEL"] ,
                 AcquisitionDimension["ANGLE"] , AcquisitionDimension["VERTICAL"] ,
                 AcquisitionDimension["HORIZONTAL"]],
                              list(sgeometry.dimension_labels))
        sino = sgeometry.allocate()

        # test reshape
        new_order = [AcquisitionDimension["HORIZONTAL"] ,
                 AcquisitionDimension["CHANNEL"] , AcquisitionDimension["VERTICAL"] ,
                 AcquisitionDimension["ANGLE"]]
        sino.reorder(new_order)

        self.assertListEqual(new_order, list(sino.geometry.dimension_labels))

        ss1 = sino.get_slice(vertical = 0)
        self.assertListEqual([AcquisitionDimension["HORIZONTAL"] ,
                 AcquisitionDimension["CHANNEL"]  ,
                 AcquisitionDimension["ANGLE"]], list(ss1.geometry.dimension_labels))
        ss2 = sino.get_slice(vertical = 0, channel=0)
        self.assertListEqual([AcquisitionDimension["HORIZONTAL"] ,
                 AcquisitionDimension["ANGLE"]], list(ss2.geometry.dimension_labels))


    def test_ImageDataSubset(self):
        new_order = ['horizontal_x', 'channel', 'horizontal_y']


        vgeometry = ImageGeometry(voxel_num_x=4, voxel_num_y=3, channels=2, dimension_labels=new_order)
        # expected dimension_labels

        self.assertListEqual(new_order,
                              list(vgeometry.dimension_labels))
        vol = vgeometry.allocate()

        # test reshape
        new_order = ['channel', 'horizontal_y','horizontal_x']
        vol.reorder(new_order)

        self.assertListEqual(new_order, list(vol.geometry.dimension_labels))

        ss1 = vol.get_slice(horizontal_x = 0)
        self.assertListEqual(['channel', 'horizontal_y'], list(ss1.geometry.dimension_labels))

        vg = ImageGeometry(3,4,5,channels=2)
        self.assertListEqual([ImageDimension["CHANNEL"], ImageDimension["VERTICAL"],
                ImageDimension["HORIZONTAL_Y"], ImageDimension["HORIZONTAL_X"]],
                              list(vg.dimension_labels))
        ss2 = vg.allocate()
        ss3 = vol.get_slice(channel=0)
        self.assertListEqual([ImageDimension["HORIZONTAL_Y"], ImageDimension["HORIZONTAL_X"]], list(ss3.geometry.dimension_labels))

    def test_DataContainerSubset(self):
        dc = DataContainer(np.ones((2,3,4,5)))

        dc.dimension_labels =[AcquisitionDimension["CHANNEL"] ,
                 AcquisitionDimension["ANGLE"] , AcquisitionDimension["VERTICAL"] ,
                 AcquisitionDimension["HORIZONTAL"]]

        # test reshape
        new_order = [AcquisitionDimension["HORIZONTAL"] ,
                 AcquisitionDimension["CHANNEL"] , AcquisitionDimension["VERTICAL"] ,
                 AcquisitionDimension["ANGLE"]]
        dc.reorder(new_order)

        self.assertListEqual(new_order, list(dc.dimension_labels))

        ss1 = dc.get_slice(vertical=0)

        self.assertListEqual([AcquisitionDimension["HORIZONTAL"] ,
                 AcquisitionDimension["CHANNEL"]  ,
                 AcquisitionDimension["ANGLE"]], list(ss1.dimension_labels))

        ss2 = dc.get_slice(vertical=0, channel=0)
        self.assertListEqual([AcquisitionDimension["HORIZONTAL"] ,
                 AcquisitionDimension["ANGLE"]], list(ss2.dimension_labels))

        # Check we can get slice still even if force parameter is passed:
        ss3 = dc.get_slice(vertical=0, channel=0, force=True)
        self.assertListEqual([AcquisitionDimension["HORIZONTAL"] ,
                    AcquisitionDimension["ANGLE"]], list(ss3.dimension_labels))


    def test_DataContainerChaining(self):
        dc = self.create_DataContainer(256,256,256,1)

        dc.add(9,out=dc)\
          .subtract(1,out=dc)
        self.assertEqual(1+9-1,dc.as_array().flatten()[0])


    def test_reduction(self):
        dc = self.create_DataContainer(2,2,2,value=1)
        sqnorm = dc.squared_norm()
        norm = dc.norm()
        self.assertEqual(sqnorm, 8.0)
        np.testing.assert_almost_equal(norm, np.sqrt(8.0), decimal=7)
        sum = dc.sum(axis=('X','Z'))
        np.testing.assert_almost_equal(sum.as_array(), [np.float64(4),np.float64(4)])
        np.testing.assert_equal(sum.dimension_labels,('Y',))
        sum = dc.sum()
        np.testing.assert_almost_equal(sum, 8.0)


    def test_reduction_mean(self):
        ig = ImageGeometry(2,2)
        data = ig.allocate(0)
        np_arr = data.as_array()
        np_arr[0][0] = 0
        np_arr[0][1] = 1
        np_arr[1][0] = 2
        np_arr[1][1] = 3
        data.fill(np_arr)

        mean = data.mean()
        expected = np.float64(0+1+2+3)/np.float64(4)
        np.testing.assert_almost_equal(mean, expected)


    def directional_reduction_unary_test(self, data, test_func, expected_func, out, function_name):
        def error_message(function_name, test_name):
            return "Failed with reduction " + function_name + " on test " + test_name
        # test specifying function in 1 axis
        result = test_func(axis=data.dimension_labels[1])
        expected = expected_func(data.as_array(), axis=1)
        expected_dimension_labels = data.dimension_labels[0],data.dimension_labels[2]
        np.testing.assert_almost_equal(result.as_array(), expected, err_msg=error_message(function_name, "'with 1 axis'"))
        self.assertEqual(result.dimension_labels, expected_dimension_labels, f"{function_name} 'with 1 axis'")
        # test specifying axis with an int
        result = test_func(axis=1)
        np.testing.assert_almost_equal(result.as_array(), expected, err_msg=error_message(function_name, "'with 1 axis'"))
        self.assertEqual(result.dimension_labels,expected_dimension_labels, f"{function_name} 'with 1 axis'")
        # test specifying function in 2 axes
        result = test_func(axis=(data.dimension_labels[0],data.dimension_labels[1]))
        np.testing.assert_almost_equal(result.as_array(), expected_func(data.as_array(), axis=(0,1)), err_msg=error_message(function_name, "'with 2 axes'"))
        self.assertEqual(result.dimension_labels, (data.dimension_labels[2],), f"{function_name} 'with 2 axes'")
        # test specifying function in 2 axes with an int
        result = test_func(axis=(0,1))
        np.testing.assert_almost_equal(result.as_array(), expected_func(data.as_array(), axis=(0,1)), err_msg=error_message(function_name, "'with 2 axes'"))
        self.assertEqual(result.dimension_labels, (data.dimension_labels[2],), f"{function_name} 'with 2 axes'")
        # test specifying function in 3 axes
        result = test_func(axis=(data.dimension_labels[0],data.dimension_labels[1],data.dimension_labels[2]))
        np.testing.assert_almost_equal(result, expected_func(data.as_array()), err_msg=error_message(function_name, "'with 3 axes'"))
        # test providing a DataContainer to out
        expected_array = expected_func(data.as_array(), axis = 0)
        test_func(axis=0, out=out)
        np.testing.assert_almost_equal(out.as_array(), expected_array, err_msg=error_message(function_name, "'of out argument'"))
        self.assertEqual(out.dimension_labels, (data.dimension_labels[1],data.dimension_labels[2]), f"{function_name} 'of out argument'")
        test_func(axis=data.dimension_labels[0], out=out)
        np.testing.assert_almost_equal(out.as_array(), expected_array, err_msg=error_message(function_name, "'of out argument'"))
        self.assertEqual(out.dimension_labels, (data.dimension_labels[1],data.dimension_labels[2]), f"{function_name} 'of out argument'")
        # test providing a numpy array to out
        out = np.zeros((2,2), dtype=data.dtype)
        test_func(axis=0, out=out)
        np.testing.assert_almost_equal(out, expected_array, err_msg=error_message(function_name, "'of out argument'"))

    def test_directional_reduction_unary(self):
        np_arr = np.array([[[0,1],[2,3]],[[4,5],[6,7]]], dtype=np.float32)
        # create DataContainer test class
        dc =  DataContainer(np_arr, dimension_labels=('vertical', 'horizontal_y', 'horizontal_x'))
        dc_out = DataContainer(np.zeros((2,2)),dimension_labels=('horizontal_y', 'horizontal_x'))
        # create ImageData test class
        id = ImageGeometry(2,2,2).allocate(0)
        id.fill(np_arr)
        id_out = ImageGeometry(2,2).allocate(0)
        # create complex ImageData test class
        id_complex = ImageGeometry(2,2,2).allocate(0, dtype=complex)
        complex_arr = np.empty((2,2,2), dtype=complex)
        complex_arr.real = np_arr
        complex_arr.imag = np.array([[[7,6],[5,4]],[[3,2],[1,0]]])
        id_complex.fill(complex_arr)
        id_complex_out = ImageGeometry(2,2).allocate(0, dtype=complex)
        id_complex_out.fill(np.zeros((2,2), dtype=complex))
        # create AcquisitionData test class
        ag = AcquisitionGeometry.create_Parallel3D().set_angles(np.linspace(0, 180, num=2)).set_panel((2,2))
        ad = ag.allocate()
        ad.fill(np_arr)
        ag = AcquisitionGeometry.create_Parallel3D().set_angles(np.linspace(0, 180, num=0)).set_panel((2,2))
        ad_out = ag.allocate()
        ad_out.fill(np.zeros((2,2)))

        data_classes = [dc, id, id_complex, ad]
        out_classes = [dc_out,id_out,id_complex_out, ad_out]
        for j in np.arange(len(data_classes)):
            function_names = ['mean','sum','min','max']
            for i in np.arange(len(function_names)):
                self.directional_reduction_unary_test(data_classes[j], getattr(data_classes[j],function_names[i]), getattr(np, function_names[i]), out_classes[j], function_names[i])


    def test_mean_direction(self):
        np_arr = np.array([[[0,1],[2,3]],[[4,5],[6,7]]])
        ig = ImageGeometry(2,2,2)
        data = ig.allocate(0)
        data.fill(np_arr)

        # test specifying mean in 1 axis
        mean = data.mean(axis='horizontal_y')
        np.testing.assert_almost_equal(mean.as_array(), [[1.0, 2.0],[5.0, 6.0]])
        np.testing.assert_equal(mean.dimension_labels,('vertical','horizontal_x'))
        mean = data.mean(axis=1)
        np.testing.assert_almost_equal(mean.as_array(), [[1.0, 2.0],[5.0, 6.0]])
        # test specifying mean in 2 axes
        mean = data.mean(axis=('horizontal_y', 'vertical'))
        np.testing.assert_almost_equal(mean.as_array(), [3.0, 4.0])
        np.testing.assert_equal(mean.dimension_labels,('horizontal_x',))
        # test specifying mean in 3 axes
        expected = np.float64(0+1+2+3+4+5+6+7)/np.float64(8)
        mean = data.mean(axis=('horizontal_x','horizontal_y','vertical'))
        np.testing.assert_almost_equal(mean, expected)
        # test mean on VectorData
        np_arr = np.array([0,1,2,3,4])
        vg = VectorGeometry(5)
        vd = vg.allocate(0)
        vd.fill(np_arr)
        vd.dimension_labels = 'x'
        np.testing.assert_almost_equal(vd.mean(axis='x'), np.mean(vd))


    def test_multiply_out(self):
        ig = ImageGeometry(10,11,12)
        u = ig.allocate()
        a = np.ones(u.shape)

        u.fill(a)

        np.testing.assert_array_equal(a, u.as_array())

        #u = ig.allocate(ImageDimension["RANDOM_INT"], seed=1)
        l = functools.reduce(lambda x,y: x*y, (10,11,12), 1)

        a = np.zeros((l, ), dtype=np.float32)
        for i in range(l):
            a[i] = np.sin(2 * i* 3.1415/l)
        b = np.reshape(a, u.shape)
        u.fill(b)
        np.testing.assert_array_equal(b, u.as_array())

        u.multiply(2, out=u)
        c = b * 2
        np.testing.assert_allclose(u.as_array(), c)


    def test_sapyb_datacontainer_f(self):
        #a vec, b vec

        ig = ImageGeometry(10,10)
        d1 = ig.allocate(dtype=np.float32)
        d2 = ig.allocate(dtype=np.float32)
        a = ig.allocate(dtype=np.float32)
        b = ig.allocate(dtype=np.float32)

        d1.fill(np.asarray(np.arange(1,101).reshape(10,10), dtype=np.float32))
        d2.fill(np.asarray(np.arange(1,101).reshape(10,10), dtype=np.float32))
        a.fill(1.0/d1.as_array())
        b.fill(-1.0/d2.as_array())

        out = ig.allocate(-1,dtype=np.float32)
        # equals to 1 + -1 = 0
        out = d1.sapyb(a,d2,b)
        res = np.zeros_like(d1.as_array())
        np.testing.assert_array_equal(res, out.as_array())

        out.fill(0)
        d1.sapyb(a,d2,b, out)
        res = np.zeros_like(d1.as_array())
        np.testing.assert_array_equal(res, out.as_array())


    def test_sapyb_scalar_f(self):
        # a,b scalar
        ig = ImageGeometry(10,10)
        d1 = ig.allocate(1, dtype=np.float32)
        d2 = ig.allocate(2, dtype=np.float32)
        a = 2.
        b = -1.

        out = ig.allocate(-1,dtype=np.float32)
        # equals to 2*[1] + -1*[2] = 0
        out = d1.sapyb(a,d2,b)
        res = np.zeros_like(d1.as_array())
        np.testing.assert_array_equal(res, out.as_array())

        out.fill(0)
        d1.sapyb(a,d2,b, out)
        np.testing.assert_array_equal(res, out.as_array())

        d1.sapyb(a,d2,b, out=d1)
        np.testing.assert_array_equal(res, d1.as_array())

        d1.fill(1)
        d1.sapyb(a,d2,b, out=d2)
        np.testing.assert_array_equal(res, d2.as_array())


    def test_sapyb_datacontainer_scalar_f(self):
        #mix: a scalar and b DataContainer and a DataContainer and b scalar
        ig = ImageGeometry(10,10)
        d1 = ig.allocate(1., dtype=np.complex64)
        d2 = ig.allocate(2.,dtype=np.complex64)
        a = 2.+2j
        b = ig.allocate(-1.-1j, dtype=np.complex64)

        out = ig.allocate(-1,dtype=np.complex64)
        # equals to (2+2j)*[1] + -(1+j)*[2] = 0

        out = d1.sapyb(a,d2,b)
        res = ig.allocate(0, dtype=np.complex64)
        np.testing.assert_array_equal(res.as_array(), out.as_array())

        out.fill(-1)
        d1.sapyb(a,d2,b, out)
        np.testing.assert_array_equal(res.as_array(), out.as_array())

        out = d2.sapyb(b,d1,a)
        res = ig.allocate(0, dtype=np.complex64)
        np.testing.assert_array_equal(res.as_array(), out.as_array())

        out.fill(-1)
        d2.sapyb(b,d1,a, out)
        np.testing.assert_array_equal(res.as_array(), out.as_array())


    def test_sapyb_scalar_c(self):
        # a, b scalar
        ig = ImageGeometry(10,10)
        d1 = ig.allocate(1, dtype=np.complex64)
        d2 = ig.allocate(2, dtype=np.complex64)
        a = 2.+2j
        b = -1.-1j

        out = ig.allocate(-1,dtype=np.complex64)
        # equals to (2+2j)*[1] + -(1+j)*[2] = 0
        out = d1.sapyb(a,d2,b)
        res = np.zeros_like(d1.as_array())
        np.testing.assert_array_equal(res, out.as_array())

        out.fill(0)
        d1.sapyb(a,d2,b, out)
        np.testing.assert_array_equal(res, out.as_array())

        d1.sapyb(a,d2,b, out=d1)
        np.testing.assert_array_equal(res, d1.as_array())

        d1.fill(1)
        d1.sapyb(a,d2,b, out=d2)
        np.testing.assert_array_equal(res, d2.as_array())


    def test_sapyb_datacontainer_c(self):
        #a vec, b vec
        ig = ImageGeometry(10,10)
        d1 = ig.allocate(dtype=np.complex64)
        d2 = ig.allocate(dtype=np.complex64)
        a = ig.allocate(dtype=np.complex64)
        b = ig.allocate(dtype=np.complex64)

        arr = np.empty(ig.shape, dtype=np.complex64)
        arr.real = np.asarray(np.arange(1,101).reshape(10,10), dtype=np.float32)
        arr.imag = np.asarray(np.arange(1,101).reshape(10,10), dtype=np.float32)

        d1.fill(arr)

        arr.imag = -1* arr.imag
        d2.fill(arr)

        a.fill(d2.as_array())
        b.fill(d1.as_array())

        out = ig.allocate(-1,dtype=np.complex64)
        # equals to d1^ * d1 + d2^*d2 = d1**2 + d2**2 = 2* arr.norm = 2 * (arr.real **2 + arr.imag **2)
        out = d1.sapyb(a,d2,b)
        res = 2* (arr.real * arr.real + arr.imag * arr.imag)
        np.testing.assert_array_equal(res, out.as_array())

        out.fill(0)
        d1.sapyb(a,d2,b, out)
        np.testing.assert_array_equal(res, out.as_array())


    def test_sapyb_datacontainer_scalar_c(self):
        #mix: a scalar and b DataContainer and a DataContainer and b scalar
        ig = ImageGeometry(10,10)
        d1 = ig.allocate(1., dtype=np.complex64)
        d2 = ig.allocate(2.,dtype=np.complex64)
        a = 2.+2j
        b = ig.allocate(-1.-1j, dtype=np.complex64)


        out = ig.allocate(-1,dtype=np.complex64)
        # equals to (2+2j)*[1] + -(1+j)*[2] = 0

        out = d1.sapyb(a,d2,b)
        res = ig.allocate(0, dtype=np.complex64)
        np.testing.assert_array_equal(res.as_array(), out.as_array())

        out.fill(-1)
        d1.sapyb(a,d2,b, out)
        np.testing.assert_array_equal(res.as_array(), out.as_array())

        out = d2.sapyb(b,d1,a)
        res = ig.allocate(0, dtype=np.complex64)
        np.testing.assert_array_equal(res.as_array(), out.as_array())

        out.fill(-1)
        d2.sapyb(b,d1,a, out)
        np.testing.assert_array_equal(res.as_array(), out.as_array())


    def test_sapyb_scalar_f_c(self):
        # a,b scalar
        ig = ImageGeometry(10,10)
        d1 = ig.allocate(1, dtype=np.complex64)
        d2 = ig.allocate(2, dtype=np.float32)
        a = 2.+1j
        b = -1.

        # equals to 2*[1] + -1*[2] = 0
        out = d1.sapyb(a,d2,b)
        res = np.zeros_like(d1.as_array()) + 1j
        np.testing.assert_array_equal(res, out.as_array())

        out.fill(0)
        d1.sapyb(a,d2,b, out)
        np.testing.assert_array_equal(res, out.as_array())

        d1.sapyb(a,d2,b, out=d1)
        np.testing.assert_array_equal(res, d1.as_array())

        d1.fill(1)
        try:
            with self.assertRaises(np.core._exceptions.UFuncTypeError) as context:
                d1.sapyb(a,d2,b, out=d2)
        except AttributeError as ae:
            log.info("Probably numpy version too low: %s", ae)

        # print ("Exception thrown:", str(context.exception))

        # out is complex
        # d1.fill(1+0j)
        d2.fill(2)
        d1.sapyb(a,d2,b,out=d1)
        # 2+1j * [1+0j] -1 * [2]
        np.testing.assert_array_equal(1j * np.ones_like(d1.as_array()), d1.as_array())


    def test_min(self):
        ig = ImageGeometry(10,10)
        a = np.asarray(np.linspace(-10,10, num=100, endpoint=True), dtype=np.float32)
        a = a.reshape((10,10))
        d1 = ig.allocate(1)
        d1.fill(a)
        self.assertAlmostEqual(d1.min(), -10.)


    def test_min_direction(self):
        ig = ImageGeometry(2,2,2)
        data = ig.allocate(0)
        np_arr = np.array([[[0,1],[2,3]],[[4,5],[6,7]]])
        data.fill(np_arr)

        # test specifying min in 3 axes
        min = data.min(axis=('vertical','horizontal_x','horizontal_y'))
        self.assertAlmostEqual(min, 0)
        # test specifying min in 2 axes
        min = data.min(axis=('horizontal_y', 'vertical'))
        np.testing.assert_almost_equal(min.as_array(), [0.0, 1.0])
        # test specifying min in 1 axis
        min = data.min(axis='horizontal_x')
        expected = [[np.float64(0), np.float64(2)],[np.float64(4), np.float64(6)]]
        np.testing.assert_almost_equal(min.as_array(), expected)
        np.testing.assert_equal(min.dimension_labels,('vertical','horizontal_y'))
        # test specifying min in 1 axis using numpy axis argument
        min = data.min(axis=0)
        expected = [[np.float64(0), np.float64(1)],[np.float64(2), np.float64(3)]]
        np.testing.assert_almost_equal(min.as_array(), expected)


    def test_max(self):
        ig = ImageGeometry(10,10)
        a = np.asarray(np.linspace(-10,10, num=100, endpoint=True), dtype=np.float32)
        a = a.reshape((10,10))
        d1 = ig.allocate(1)
        d1.fill(a)
        self.assertAlmostEqual(d1.max(), 10.)


    def test_max_direction(self):
        ig = ImageGeometry(2,2,2)
        data = ig.allocate(0)
        np_arr = np.array([[[0,1],[2,3]],[[4,5],[6,7]]])
        data.fill(np_arr)

        # test specifying max in 3 axes
        max = data.max(axis=('vertical','horizontal_x','horizontal_y'))
        self.assertAlmostEqual(max, 7)
        # test specifying max in 2 axes
        max = data.max(axis=('horizontal_y', 'vertical'))
        np.testing.assert_almost_equal(max.as_array(), [6.0, 7.0])
        np.testing.assert_equal(max.dimension_labels,('horizontal_x',))
        # test specifying max in 1 axis
        max = data.max(axis='horizontal_x')
        expected = [[np.float64(1), np.float64(3)],[np.float64(5), np.float64(7)]]
        np.testing.assert_almost_equal(max.as_array(), expected)
        np.testing.assert_equal(max.dimension_labels,('vertical','horizontal_y'))
        # test specifying max in 1 axis using numpy axis argument
        max = data.max(axis=0)
        expected = [[np.float64(4), np.float64(5)],[np.float64(6), np.float64(7)]]
        np.testing.assert_almost_equal(max.as_array(), expected)


    def test_size(self):
        ig = ImageGeometry(10,10)
        d1 = ig.allocate(1)

        self.assertEqual( d1.size, 100 )

        sgeometry = AcquisitionGeometry.create_Parallel3D().set_angles(np.linspace(0, 180, num=10)).set_panel((5,3)).set_channels(2)

        ad = sgeometry.allocate()

        self.assertEqual( ad.size, 3*5*10*2 )


    def test_negation(self):
        X, Y, Z = 256, 512, 512
        a = np.ones((X, Y, Z), dtype='int32')

        ds = - DataContainer(a, False, ['X', 'Y', 'Z'])

        np.testing.assert_array_equal(ds.as_array(), -a)

    @parametrize("xp, device, raise_error, err_type", 
        [param(numpy, None, None, None, id="numpy"), 
         param(torch, None, None, None, id="torch_cpu"),
         ])
    def test_fill_dimension_ImageData(self, xp, device, raise_error, err_type):
        if xp is None:
            self.skipTest("torch not available")
        ig = ImageGeometry(2,3,4)
        u = ig.allocate(0)
        a = xp.ones((4,2))
        # default_labels = [ImageDimension["VERTICAL"], ImageDimension["HORIZONTAL_Y"], ImageDimension["HORIZONTAL_X"]]

        u.fill(a, horizontal_y=0)
        np.testing.assert_array_equal(u.get_slice(horizontal_y=0).as_array(), a)

        u.fill(2, horizontal_y=1)
        np.testing.assert_array_equal(u.get_slice(horizontal_y=1).as_array(), 2 * a)

        u.fill(2, horizontal_y=1)
        np.testing.assert_array_equal(u.get_slice(horizontal_y=1).as_array(), 2 * a)

        b = u.get_slice(horizontal_y=2)
        b.fill(3)
        u.fill(b, horizontal_y=2)
        np.testing.assert_array_equal(u.get_slice(horizontal_y=2).as_array(), 3 * a)


    def test_fill_dimension_AcquisitionData(self):
        ag = AcquisitionGeometry.create_Parallel3D()
        ag.set_channels(4)
        ag.set_panel([2,3])
        ag.set_angles([0,1,2,3,5])
        ag.set_labels(('horizontal','angle','vertical','channel'))
        u = ag.allocate(0)
        a = np.ones((4,2))
        # default_labels = [ImageDimension["VERTICAL"], ImageDimension["HORIZONTAL_Y"], ImageDimension["HORIZONTAL_X"]]

        data = u.as_array()
        axis_number = u.get_dimension_axis('horizontal_y')

        u.fill(a, horizontal_y=0)
        np.testing.assert_array_equal(u.subset(horizontal_y=0).as_array(), a)

        u.fill(2, horizontal_y=1)
        np.testing.assert_array_equal(u.subset(horizontal_y=0).as_array(), 2 * a)

        u.fill(2, horizontal_y=1)
        np.testing.assert_array_equal(u.subset(horizontal_y=1).as_array(), 2 * a)

        b = u.subset(horizontal_y=2)
        b.fill(3)
        u.fill(b, horizontal_y=2)
        np.testing.assert_array_equal(u.subset(horizontal_y=2).as_array(), 3 * a)

        # slice with 2 axis
        a = np.ones((2,))
        u.fill(a, horizontal_y=1, vertical=0)
        np.testing.assert_array_equal(u.subset(horizontal_y=1, vertical=0).as_array(), a)


    def test_fill_dimension_AcquisitionData(self):
        ag = AcquisitionGeometry.create_Parallel3D()
        ag.set_channels(4)
        ag.set_panel([2,3])
        ag.set_angles([0,1,2,3,5])
        ag.set_labels(('horizontal','angle','vertical','channel'))
        u = ag.allocate(0)
        # (2, 5, 3, 4)
        a = np.ones((2,5))
        # default_labels = [ImageDimension["VERTICAL"], ImageDimension["HORIZONTAL_Y"], ImageDimension["HORIZONTAL_X"]]
        b = u.get_slice(channel=0, vertical=0)
        data = u.as_array()

        u.fill(a, channel=0, vertical=0)
        np.testing.assert_array_equal(u.get_slice(channel=0, vertical=0).as_array(), a)

        u.fill(2, channel=0, vertical=0)
        np.testing.assert_array_equal(u.get_slice(channel=0, vertical=0).as_array(), 2 * a)

        u.fill(2, channel=0, vertical=0)
        np.testing.assert_array_equal(u.get_slice(channel=0, vertical=0).as_array(), 2 * a)

        b = u.get_slice(channel=0, vertical=0)
        b.fill(3)
        u.fill(b, channel=1, vertical=1)
        np.testing.assert_array_equal(u.get_slice(channel=1, vertical=1).as_array(), 3 * a)


    def test_vectordata_dot_product(self):
        x = np.array([1 + 1j, 2 - 5j])
        y = np.array([3 - 2j, 1 + 1j])

        a1 = np.dot(y, x.conjugate())
        a2 = np.dot(x.conjugate(), y)
        self.assertAlmostEqual( a1, a2)
        b1 = np.vdot(x, y)
        b2 = np.vdot(y, x).conjugate()
        self.assertAlmostEqual( b1, b2)

        print(a1, b1)
        print(a2, b2)

        xcil = VectorData(x)
        ycil = VectorData(y)
        self.assertAlmostEqual( ycil.dot(xcil), xcil.dot(ycil).conjugate())

    def test_acquisition_data_geometry_override(self):
        ag = AcquisitionGeometry.create_Cone2D([0,-100], [0,200], units='mm',detector_direction_x=[1, 0])
        ag.set_panel(120, pixel_size=0.1, origin='bottom-left')
        ag.set_angles(np.linspace(0., 360., 120, endpoint=False))
        ag.set_labels(('angle','horizontal'))

        ig = ag.get_ImageGeometry()

        dtype_before = ag.dtype

        sinogram64 = np.zeros((120, 120), dtype='float64')
        AcquisitionData(sinogram64, deep_copy=False, geometry=ag)

        self.assertEqual(dtype_before, ag.dtype)

    def test_geometry_dtype_change(self):
        ag = AcquisitionGeometry.create_Cone2D([0,-100], [0,200], units='mm',detector_direction_x=[1, 0])
        ag.set_panel(120, pixel_size=0.1, origin='bottom-left')
        ag.set_angles(np.linspace(0., 360., 120, endpoint=False))
        ag.set_labels(('angle','horizontal'))

        ig = ag.get_ImageGeometry()

        ag_dtype_before = ag.dtype
        ig_dtype_before = ig.dtype

        sinogram64 = np.zeros((120, 120), dtype='float64')

        AcquisitionData(sinogram64, deep_copy=False, geometry=ag)
        self.assertEqual(ag_dtype_before, ag.dtype)

        ImageData(sinogram64, deep_copy=False, geometry=ig)
        self.assertEqual(ig_dtype_before, ig.dtype)