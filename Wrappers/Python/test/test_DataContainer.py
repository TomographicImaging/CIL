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

import sys
import unittest
import numpy
from cil.framework import DataContainer
from cil.framework import ImageData
from cil.framework import AcquisitionData
from cil.framework import ImageGeometry, BlockGeometry, VectorGeometry
from cil.framework import AcquisitionGeometry
from timeit import default_timer as timer


def dt(steps):
    return steps[-1] - steps[-2]
def aid(x):
    # This function returns the memory
    # block address of an array.
    return x.__array_interface__['data'][0]



class TestDataContainer(unittest.TestCase):
    def create_simple_ImageData(self):
        N = 64
        ig = ImageGeometry(voxel_num_x=N, voxel_num_y=N)
        Phantom = ImageData(geometry=ig)

        x = Phantom.as_array()

        x[int(round(N/4)):int(round(3*N/4)),
          int(round(N/4)):int(round(3*N/4))] = 0.5
        x[int(round(N/8)):int(round(7*N/8)),
          int(round(3*N/8)):int(round(5*N/8))] = 1

        return (ig, Phantom)

    def create_DataContainer(self, X,Y,Z, value=1):
        steps = [timer()]
        a = value * numpy.ones((X, Y, Z), dtype='float32')
        steps.append(timer())
        t0 = dt(steps)
        #print("a refcount " , sys.getrefcount(a))
        ds = DataContainer(a, False, ['X', 'Y', 'Z'])
        return ds

    def test_creation_nocopy(self):
        shape = (2, 3, 4, 5)
        size = shape[0]
        for i in range(1, len(shape)):
            size = size * shape[i]
        #print("a refcount " , sys.getrefcount(a))
        a = numpy.asarray([i for i in range(size)])
        #print("a refcount " , sys.getrefcount(a))
        a = numpy.reshape(a, shape)
        #print("a refcount " , sys.getrefcount(a))
        ds = DataContainer(a, False, ['X', 'Y', 'Z', 'W'])
        #print("a refcount " , sys.getrefcount(a))
        self.assertEqual(id(a), id(ds.array))
        self.assertEqual(ds.dimension_labels, ('X', 'Y', 'Z', 'W'))

    def testGb_creation_nocopy(self):
        X, Y, Z = 512, 512, 512
        X, Y, Z = 256, 512, 512
        steps = [timer()]
        a = numpy.ones((X, Y, Z), dtype='float32')
        steps.append(timer())
        t0 = dt(steps)
        print("test clone")
        #print("a refcount " , sys.getrefcount(a))
        ds = DataContainer(a, False, ['X', 'Y', 'Z'])
        #print("a refcount " , sys.getrefcount(a))
        self.assertEqual(sys.getrefcount(a), 3)
        ds1 = ds.copy()
        self.assertNotEqual(aid(ds.as_array()), aid(ds1.as_array()))
        ds1 = ds.clone()
        self.assertNotEqual(aid(ds.as_array()), aid(ds1.as_array()))

    def testInlineAlgebra(self):
        print("Test Inline Algebra")
        X, Y, Z = 1024, 512, 512
        X, Y, Z = 256, 512, 512
        steps = [timer()]
        a = numpy.ones((X, Y, Z), dtype='float32')
        steps.append(timer())
        t0 = dt(steps)
        print(t0)
        #print("a refcount " , sys.getrefcount(a))
        ds = DataContainer(a, False, ['X', 'Y', 'Z'])
        #ds.__iadd__( 2 )
        ds += 2
        steps.append(timer())
        print(dt(steps))
        self.assertEqual(ds.as_array()[0][0][0], 3.)
        #ds.__isub__( 2 )
        ds -= 2
        steps.append(timer())
        print(dt(steps))
        self.assertEqual(ds.as_array()[0][0][0], 1.)
        #ds.__imul__( 2 )
        ds *= 2
        steps.append(timer())
        print(dt(steps))
        self.assertEqual(ds.as_array()[0][0][0], 2.)
        #ds.__idiv__( 2 )
        ds /= 2
        steps.append(timer())
        print(dt(steps))
        self.assertEqual(ds.as_array()[0][0][0], 1.)

        ds1 = ds.copy()
        #ds1.__iadd__( 1 )
        ds1 += 1
        #ds.__iadd__( ds1 )
        ds += ds1
        steps.append(timer())
        print(dt(steps))
        self.assertEqual(ds.as_array()[0][0][0], 3.)
        #ds.__isub__( ds1 )
        ds -= ds1
        steps.append(timer())
        print(dt(steps))
        self.assertEqual(ds.as_array()[0][0][0], 1.)
        #ds.__imul__( ds1 )
        ds *= ds1
        steps.append(timer())
        print(dt(steps))
        self.assertEqual(ds.as_array()[0][0][0], 2.)
        #ds.__idiv__( ds1 )
        ds /= ds1
        steps.append(timer())
        print(dt(steps))
        self.assertEqual(ds.as_array()[0][0][0], 1.)

    def test_unary_operations(self):
        print("Test unary operations")
        X, Y, Z = 1024, 512, 512
        X, Y, Z = 256, 512, 512
        steps = [timer()]
        a = -numpy.ones((X, Y, Z), dtype='float32')
        steps.append(timer())
        t0 = dt(steps)
        print(t0)
        #print("a refcount " , sys.getrefcount(a))
        ds = DataContainer(a, False, ['X', 'Y', 'Z'])

        ds.sign(out=ds)
        steps.append(timer())
        print(dt(steps))
        self.assertEqual(ds.as_array()[0][0][0], -1.)

        ds.abs(out=ds)
        steps.append(timer())
        print(dt(steps))
        self.assertEqual(ds.as_array()[0][0][0], 1.)

        ds.__imul__(2)
        ds.sqrt(out=ds)
        steps.append(timer())
        print(dt(steps))
        self.assertEqual(ds.as_array()[0][0][0],
                         numpy.sqrt(2., dtype='float32'))

    def test_binary_operations(self):
        self.binary_add()
        self.binary_subtract()
        self.binary_multiply()
        self.binary_divide()

    def binary_add(self):
        print("Test binary add")
        X, Y, Z = 512, 512, 512
        #X, Y, Z = 1024, 512, 512
        steps = [timer()]
        a = numpy.ones((X, Y, Z), dtype='float32')
        steps.append(timer())
        t0 = dt(steps)

        #print("a refcount " , sys.getrefcount(a))
        ds = DataContainer(a, False, ['X', 'Y', 'Z'])
        ds1 = ds.copy()
        out = ds.copy()

        steps.append(timer())
        ds.add(ds1, out=out)
        steps.append(timer())
        t1 = dt(steps)
        print("ds.add(ds1, out=ds)", dt(steps))
        steps.append(timer())
        ds2 = ds.add(ds1)
        steps.append(timer())
        t2 = dt(steps)
        print("ds2 = ds.add(ds1)", dt(steps))

        #self.assertLess(t1, t2)
        self.assertEqual(out.as_array()[0][0][0], 2.)
        self.assertNumpyArrayEqual(out.as_array(), ds2.as_array())
        
        ds0 = ds
        dt1 = 0
        dt2 = 0
        for i in range(10):
            steps.append(timer())
            ds0.add(2, out=out)
            steps.append(timer())
            print("ds0.add(2,out=out)", dt(steps), 3, ds0.as_array()[0][0][0])
            self.assertEqual(3., out.as_array()[0][0][0])

            dt1 += dt(steps)/10
            steps.append(timer())
            ds3 = ds0.add(2)
            steps.append(timer())
            print("ds3 = ds0.add(2)", dt(steps), 5, ds3.as_array()[0][0][0])
            dt2 += dt(steps)/10
        
        self.assertNumpyArrayEqual(out.as_array(), ds3.as_array())
        #self.assertLess(dt1, dt2)
        

    def binary_subtract(self):
        print("Test binary subtract")
        X, Y, Z = 512, 512, 512
        steps = [timer()]
        a = numpy.ones((X, Y, Z), dtype='float32')
        steps.append(timer())
        t0 = dt(steps)

        #print("a refcount " , sys.getrefcount(a))
        ds = DataContainer(a, False, ['X', 'Y', 'Z'])
        ds1 = ds.copy()
        out = ds.copy()

        steps.append(timer())
        ds.subtract(ds1, out=out)
        steps.append(timer())
        t1 = dt(steps)
        print("ds.subtract(ds1, out=ds)", dt(steps))
        self.assertEqual(0., out.as_array()[0][0][0])

        steps.append(timer())
        ds2 = out.subtract(ds1)
        self.assertEqual(-1., ds2.as_array()[0][0][0])

        steps.append(timer())
        t2 = dt(steps)
        print("ds2 = ds.subtract(ds1)", dt(steps))

        #self.assertLess(t1, t2)

        del ds1
        ds0 = ds.copy()
        steps.append(timer())
        ds0.subtract(2, out=ds0)
        #ds0.__isub__( 2 )
        steps.append(timer())
        print("ds0.subtract(2,out=ds0)", dt(
            steps), -1., ds0.as_array()[0][0][0])
        self.assertEqual(-1., ds0.as_array()[0][0][0])

        dt1 = dt(steps)
        ds3 = ds0.subtract(2)
        steps.append(timer())
        print("ds3 = ds0.subtract(2)", dt(steps), 0., ds3.as_array()[0][0][0])
        dt2 = dt(steps)
        #self.assertLess(dt1, dt2)
        self.assertEqual(-1., ds0.as_array()[0][0][0])
        self.assertEqual(-3., ds3.as_array()[0][0][0])

    def binary_multiply(self):
        print("Test binary multiply")
        X, Y, Z = 1024, 512, 512
        X, Y, Z = 256, 512, 512
        steps = [timer()]
        a = numpy.ones((X, Y, Z), dtype='float32')
        steps.append(timer())
        t0 = dt(steps)

        #print("a refcount " , sys.getrefcount(a))
        ds = DataContainer(a, False, ['X', 'Y', 'Z'])
        ds1 = ds.copy()

        steps.append(timer())
        ds.multiply(ds1, out=ds)
        steps.append(timer())
        t1 = dt(steps)
        print("ds.multiply(ds1, out=ds)", dt(steps))
        steps.append(timer())
        ds2 = ds.multiply(ds1)
        steps.append(timer())
        t2 = dt(steps)
        print("ds2 = ds.multiply(ds1)", dt(steps))

        #self.assertLess(t1, t2)

        ds0 = ds
        ds0.multiply(2, out=ds0)
        steps.append(timer())
        print("ds0.multiply(2,out=ds0)", dt(
            steps), 2., ds0.as_array()[0][0][0])
        self.assertEqual(2., ds0.as_array()[0][0][0])

        dt1 = dt(steps)
        ds3 = ds0.multiply(2)
        steps.append(timer())
        print("ds3 = ds0.multiply(2)", dt(steps), 4., ds3.as_array()[0][0][0])
        dt2 = dt(steps)
        #self.assertLess(dt1, dt2)
        self.assertEqual(4., ds3.as_array()[0][0][0])
        self.assertEqual(2., ds.as_array()[0][0][0])
        
        ds.multiply(2.5, out=ds0)
        self.assertEqual(2.5*2., ds0.as_array()[0][0][0])

    def binary_divide(self):
        print("Test binary divide")
        X, Y, Z = 1024, 512, 512
        X, Y, Z = 256, 512, 512
        steps = [timer()]
        a = numpy.ones((X, Y, Z), dtype='float32')
        steps.append(timer())
        t0 = dt(steps)

        #print("a refcount " , sys.getrefcount(a))
        ds = DataContainer(a, False, ['X', 'Y', 'Z'])
        ds1 = ds.copy()

        t1 = 0 
        t2 = 0
        for i in range(10):
            steps.append(timer())
            ds.divide(ds1, out=ds)
            steps.append(timer())
            t1 += dt(steps)/10.
            print("ds.divide(ds1, out=ds)", dt(steps))
            steps.append(timer())
            ds2 = ds.divide(ds1)
            steps.append(timer())
            t2 += dt(steps)/10.
            print("ds2 = ds.divide(ds1)", dt(steps))

        #self.assertLess(t1, t2)
        self.assertEqual(ds.as_array()[0][0][0], 1.)

        ds0 = ds
        ds0.divide(2, out=ds0)
        steps.append(timer())
        print("ds0.divide(2,out=ds0)", dt(steps), 0.5, ds0.as_array()[0][0][0])
        self.assertEqual(0.5, ds0.as_array()[0][0][0])

        dt1 = dt(steps)
        ds3 = ds0.divide(2)
        steps.append(timer())
        print("ds3 = ds0.divide(2)", dt(steps), 0.25, ds3.as_array()[0][0][0])
        dt2 = dt(steps)
        #self.assertLess(dt1, dt2)
        self.assertEqual(.25, ds3.as_array()[0][0][0])
        self.assertEqual(.5, ds.as_array()[0][0][0])

    def test_reverse_operand_algebra(self):
        print ("Test reverse operand algebra")

        number = 3/2
        
        X, Y, Z = 32, 64, 128
        a = numpy.ones((X, Y, Z), dtype='float32')
        ds = DataContainer(a * 3, False, ['X', 'Y', 'Z'])

        # rdiv
        b = number / ds
        numpy.testing.assert_array_almost_equal(a * 0.5, b.as_array())
        # radd
        number = 1
        b = number + ds
        numpy.testing.assert_array_almost_equal(a * 4, b.as_array())
        # rsub
        number = 3
        b = number - ds
        numpy.testing.assert_array_almost_equal(numpy.zeros_like(a), b.as_array())
        # rmul
        number = 1/3
        b = number * ds
        numpy.testing.assert_array_almost_equal(a, b.as_array())
        # rpow
        number = 2
        b = number ** ds
        numpy.testing.assert_array_almost_equal(a * 8, b.as_array())

    def test_creation_copy(self):
        shape = (2, 3, 4, 5)
        size = shape[0]
        for i in range(1, len(shape)):
            size = size * shape[i]
        #print("a refcount " , sys.getrefcount(a))
        a = numpy.asarray([i for i in range(size)])
        #print("a refcount " , sys.getrefcount(a))
        a = numpy.reshape(a, shape)
        #print("a refcount " , sys.getrefcount(a))
        ds = DataContainer(a, True, ['X', 'Y', 'Z', 'W'])
        #print("a refcount " , sys.getrefcount(a))
        self.assertEqual(sys.getrefcount(a), 2)
       
    def test_dot(self):
        a0 = numpy.asarray([i for i in range(2*3*4)])
        a1 = numpy.asarray([2*i for i in range(2*3*4)])
        
                
        ds0 = DataContainer(numpy.reshape(a0,(2,3,4)))
        ds1 = DataContainer(numpy.reshape(a1,(2,3,4)))
        
        numpy.testing.assert_equal(ds0.dot(ds1), a0.dot(a1))
        
        a2 = numpy.asarray([2*i for i in range(2*3*5)])
        ds2 = DataContainer(numpy.reshape(a2,(2,3,5)))
        
        # it should fail if the shape is wrong
        try:
            ds2.dot(ds0)
            self.assertTrue(False)
        except ValueError as ve:
            self.assertTrue(True)
            
        print ("test dot numpy")
        n0 = (ds0 * ds1).sum()
        n1 = ds0.as_array().ravel().dot(ds1.as_array().ravel())
        self.assertEqual(n0, n1)

    def test_exp_log(self):
        a0 = numpy.asarray([1 for i in range(2*3*4)])
                
        ds0 = DataContainer(numpy.reshape(a0,(2,3,4)), suppress_warning=True)
        # ds1 = DataContainer(numpy.reshape(a1,(2,3,4)), suppress_warning=True)
        b = ds0.exp().log()
        self.assertNumpyArrayEqual(ds0.as_array(), b.as_array())
        
        self.assertEqual(ds0.exp().as_array()[0][0][0], numpy.exp(1))
        self.assertEqual(ds0.log().as_array()[0][0][0], 0.)
        
        

    def test_ImageData(self):
        # create ImageData from geometry
        vgeometry = ImageGeometry(voxel_num_x=4, voxel_num_y=3, channels=2)
        #vol = ImageData(geometry=vgeometry)
        vol = vgeometry.allocate()
        self.assertEqual(vol.shape, (2, 3, 4))

        vol1 = vol + 1
        self.assertNumpyArrayEqual(vol1.as_array(), numpy.ones(vol.shape))

        vol1 = vol - 1
        self.assertNumpyArrayEqual(vol1.as_array(), -numpy.ones(vol.shape))

        vol1 = 2 * (vol + 1)
        self.assertNumpyArrayEqual(vol1.as_array(), 2 * numpy.ones(vol.shape))

        vol1 = (vol + 1) / 2
        self.assertNumpyArrayEqual(vol1.as_array(), numpy.ones(vol.shape) / 2)

        vol1 = vol + 1
        self.assertEqual(vol1.sum(), 2*3*4)
        vol1 = (vol + 2) ** 2
        self.assertNumpyArrayEqual(vol1.as_array(), numpy.ones(vol.shape) * 4)

        self.assertEqual(vol.number_of_dimensions, 3)
        
        ig2 = ImageGeometry (voxel_num_x=2,voxel_num_y=3,voxel_num_z=4, 
                     dimension_labels=[ImageGeometry.HORIZONTAL_X, ImageGeometry.HORIZONTAL_Y,
                 ImageGeometry.VERTICAL])
        data = ig2.allocate()
        self.assertNumpyArrayEqual(numpy.asarray(data.shape), numpy.asarray(ig2.shape))
        self.assertNumpyArrayEqual(numpy.asarray(data.shape), data.as_array().shape)

    def test_AcquisitionData(self):
        sgeometry = AcquisitionGeometry(dimension=2, angles=numpy.linspace(0, 180, num=10),
                                        geom_type='parallel', pixel_num_v=3,
                                        pixel_num_h=5, channels=2)
        #sino = AcquisitionData(geometry=sgeometry)
        sino = sgeometry.allocate()
        self.assertEqual(sino.shape, (2, 10, 3, 5))
        
        ag = AcquisitionGeometry (pixel_num_h=2,pixel_num_v=3,channels=4, dimension=2, angles=numpy.linspace(0, 180, num=10),
                                        geom_type='parallel', )
        print (ag.shape)
        print (ag.dimension_labels)
        
        data = ag.allocate()
        self.assertNumpyArrayEqual(numpy.asarray(data.shape), numpy.asarray(ag.shape))
        self.assertNumpyArrayEqual(numpy.asarray(data.shape), data.as_array().shape)
        
        print (data.shape, ag.shape, data.as_array().shape)
        
        ag2 = AcquisitionGeometry (pixel_num_h=2,pixel_num_v=3,channels=4, dimension=2, angles=numpy.linspace(0, 180, num=10),
                                                geom_type='parallel', 
                                                dimension_labels=[AcquisitionGeometry.VERTICAL ,
                         AcquisitionGeometry.ANGLE, AcquisitionGeometry.HORIZONTAL, AcquisitionGeometry.CHANNEL])
        
        data = ag2.allocate()
        print (data.shape, ag2.shape, data.as_array().shape)
        self.assertNumpyArrayEqual(numpy.asarray(data.shape), numpy.asarray(ag2.shape))
        self.assertNumpyArrayEqual(numpy.asarray(data.shape), data.as_array().shape)


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
            z = ImageData(numpy.random.randint(10, size=(2,3)), geometry=ig)
            self.assertTrue(False)
        except ValueError as ve:
            print (ve)
            self.assertTrue(True)

        #vgeometry.allocate('')
    def test_AcquisitionGeometry_allocate(self):
        ageometry = AcquisitionGeometry(dimension=2, 
                            angles=numpy.linspace(0, 180, num=10),
                            geom_type='parallel', pixel_num_v=3,
                            pixel_num_h=5, channels=2)
        sino = ageometry.allocate(0)
        shape = sino.shape
        print ("shape", shape)
        self.assertAlmostEqual(0.,sino.as_array()[0][0][0][0])
        self.assertAlmostEqual(0.,sino.as_array()[shape[0]-1][shape[1]-1][shape[2]-1][shape[3]-1])
        
        sino = ageometry.allocate(1)
        self.assertEqual(1,sino.as_array()[0][0][0][0])
        self.assertEqual(1,sino.as_array()[shape[0]-1][shape[1]-1][shape[2]-1][shape[3]-1])
        print (sino.dimension_labels, sino.shape, ageometry)
        
        default_order = ['channel' , 'angle' ,
                         'vertical' , 'horizontal']
        self.assertEqual(default_order[0], sino.dimension_labels[0])
        self.assertEqual(default_order[1], sino.dimension_labels[1])
        self.assertEqual(default_order[2], sino.dimension_labels[2])
        self.assertEqual(default_order[3], sino.dimension_labels[3])
        order = ['vertical' , 'horizontal', 'channel' , 'angle' ]
        ageometry.set_labels(order)
        sino = ageometry.allocate(0)
        print (sino.dimension_labels, sino.shape, ageometry)
        self.assertEqual(order[0], sino.dimension_labels[0])
        self.assertEqual(order[1], sino.dimension_labels[1])
        self.assertEqual(order[2], sino.dimension_labels[2])
        self.assertEqual(order[2], sino.dimension_labels[2])
                
        try:
            z = AcquisitionData(numpy.random.randint(10, size=(2,3)), geometry=ageometry)
            self.assertTrue(False)
        except ValueError as ve:
            print (ve)
            self.assertTrue(True)

    def test_BlockGeometry_allocate_dtype(self):
        ig1 = ImageGeometry(3,3)
        ig2 = ImageGeometry(3,3, dtype=numpy.int16)
        bg = BlockGeometry(ig1,ig2)

        # print("The default dtype of the BlockImageGeometry is {}".format(bg.dtype))   
        self.assertEqual(bg.dtype, (numpy.float32, numpy.int16))

    def dtype_allocate_test(self, geometry):
        classname = geometry.__class__.__name__
        # print("The default dtype of the {} is {}".format(classname , geometry.dtype))
        self.assertEqual(geometry.dtype, numpy.float32)

        #print("Change it to complex")
        geometry.dtype = numpy.complex
        #print("The {} dtype is now {} ".format(classname , geometry.dtype))         
        self.assertEqual(geometry.dtype, numpy.complex)

        #print("Test {} allocate".format(classname ))
        data = geometry.allocate()
        #print("Data dtype is now {} ".format(geometry.dtype))  
        self.assertEqual(data.dtype, geometry.dtype)
        #print("Data geometry dtype is now")
        #print(data.geometry.dtype)
        self.assertEqual(data.geometry.dtype, geometry.dtype) 

        #print("Allocate data with different dtype, e.g: numpy.int64 from the same {}".format(classname ))
        data = geometry.allocate(dtype=numpy.int64)
        self.assertEqual(data.dtype, numpy.int64) 
        #print("Data dtype is now {}".format(data.dtype))
        #print("Data geometry dtype is now {}".format(data.geometry.dtype))
        self.assertEqual(data.geometry.dtype, numpy.int64)
        self.assertEqual(data.dtype, numpy.int64)

        #print("The dtype of the {} remain unchanged ig.dtype =  {}".format(classname, geometry.dtype))
        self.assertEqual(geometry.dtype, numpy.complex)

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
        angles = numpy.linspace(0,180,180, dtype='float32')

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
        data = geometry.allocate(dtype=numpy.complex)
        print("Allocate complex array to a complex geometry")
        r = (1 + 1j*1)* numpy.ones(data.shape, dtype=data.dtype)
        data.fill(r)
        self.assertAlmostEqual(data.squared_norm(), data.size * 2)  
        numpy.testing.assert_almost_equal(data.abs().array, numpy.abs(r))              

        data1 = geometry.allocate(dtype=numpy.float32)
        try:
            data1.fill(r)
            self.assertTrue(False)
        except TypeError as err:
            print(err)    
            self.assertTrue(True)

    def test_ImageGeometry_allocate_complex(self):

        ig = ImageGeometry(2,2)
        self.complex_allocate_geometry_test(ig)

    def test_AcquisitionGeometry_allocate_complex(self):

        # Detectors
        detectors =  10

        # Angles
        angles = numpy.linspace(0,10,10, dtype='float32')

        # Setup acquisition geometry
        ag = AcquisitionGeometry.create_Parallel2D()\
                                .set_angles(angles)\
                                .set_panel(detectors, pixel_size=0.1)   

        self.complex_allocate_geometry_test(ag)

    def test_VectorGeometry_allocate_complex(self):

        vg = VectorGeometry(3)
        self.complex_allocate_geometry_test(vg)
        
    def test_ImageGeometry_allocate_random_same_seed(self):
        vgeometry = ImageGeometry(voxel_num_x=4, voxel_num_y=3, channels=2)
        image1 = vgeometry.allocate('random', seed=0)
        image2 = vgeometry.allocate('random', seed=0)
        numpy.testing.assert_array_equal(image1.as_array(), image2.as_array())

    def test_AcquisitionDataSubset(self):
        sgeometry = AcquisitionGeometry(dimension=2, angles=numpy.linspace(0, 180, num=10),
                                        geom_type='parallel', pixel_num_v=3,
                                        pixel_num_h=5, channels=2)
        # expected dimension_labels
        
        self.assertListEqual([AcquisitionGeometry.CHANNEL ,
                 AcquisitionGeometry.ANGLE , AcquisitionGeometry.VERTICAL ,
                 AcquisitionGeometry.HORIZONTAL],
                              list(sgeometry.dimension_labels))
        sino = sgeometry.allocate()

        # test reshape
        new_order = [AcquisitionGeometry.HORIZONTAL ,
                 AcquisitionGeometry.CHANNEL , AcquisitionGeometry.VERTICAL ,
                 AcquisitionGeometry.ANGLE]
        sino.reorder(new_order)

        self.assertListEqual(new_order, list(sino.geometry.dimension_labels))

        ss1 = sino.get_slice(vertical = 0)
        self.assertListEqual([AcquisitionGeometry.HORIZONTAL ,
                 AcquisitionGeometry.CHANNEL  ,
                 AcquisitionGeometry.ANGLE], list(ss1.geometry.dimension_labels))
        ss2 = sino.get_slice(vertical = 0, channel=0)
        self.assertListEqual([AcquisitionGeometry.HORIZONTAL ,
                 AcquisitionGeometry.ANGLE], list(ss2.geometry.dimension_labels))

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
        self.assertListEqual([ImageGeometry.CHANNEL, ImageGeometry.VERTICAL,
                ImageGeometry.HORIZONTAL_Y, ImageGeometry.HORIZONTAL_X],
                              list(vg.dimension_labels))
        ss2 = vg.allocate()
        ss3 = vol.get_slice(channel=0)
        self.assertListEqual([ImageGeometry.HORIZONTAL_Y, ImageGeometry.HORIZONTAL_X], list(ss3.geometry.dimension_labels))


    def assertNumpyArrayEqual(self, first, second):
        res = True
        try:
            numpy.testing.assert_array_equal(first, second)
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

    def assertNumpyArrayAlmostEqual(self, first, second, decimal=6):
        res = True
        try:
            numpy.testing.assert_array_almost_equal(first, second, decimal)
        except AssertionError as err:
            res = False
            print(err)
            print("expected " , second)
            print("actual " , first)

        self.assertTrue(res)
    def test_DataContainerChaining(self):
        dc = self.create_DataContainer(256,256,256,1)

        dc.add(9,out=dc)\
          .subtract(1,out=dc)
        self.assertEqual(1+9-1,dc.as_array().flatten()[0])
    def test_reduction(self):
        print ("test reductions")
        dc = self.create_DataContainer(2,2,2,value=1)
        sqnorm = dc.squared_norm()
        norm = dc.norm()
        self.assertEqual(sqnorm, 8.0)
        numpy.testing.assert_almost_equal(norm, numpy.sqrt(8.0), decimal=7)
    
    def test_reduction_mean(self):
        print ("test reduction: mean")
        ig = ImageGeometry(2,2)
        data = ig.allocate(0)
        np_arr = data.as_array()
        np_arr[0][0] = 0
        np_arr[0][1] = 1
        np_arr[1][0] = 2
        np_arr[1][1] = 3
        data.fill(np_arr)

        mean = data.mean()
        expected = numpy.float64(0+1+2+3)/numpy.float64(4)
        numpy.testing.assert_almost_equal(mean, expected)
        
        
    def test_multiply_out(self):
        print ("test multiply_out")
        import functools
        ig = ImageGeometry(10,11,12)
        u = ig.allocate()
        a = numpy.ones(u.shape)
        
        u.fill(a)
        
        numpy.testing.assert_array_equal(a, u.as_array())
        
        #u = ig.allocate(ImageGeometry.RANDOM_INT, seed=1)
        l = functools.reduce(lambda x,y: x*y, (10,11,12), 1)
        
        a = numpy.zeros((l, ), dtype=numpy.float32)
        for i in range(l):
            a[i] = numpy.sin(2 * i* 3.1415/l)
        b = numpy.reshape(a, u.shape)
        u.fill(b)
        numpy.testing.assert_array_equal(b, u.as_array())
        
        u.multiply(2, out=u)
        c = b * 2
        numpy.testing.assert_array_equal(u.as_array(), c)

    def test_axpby(self):
        print ("test axpby")
        ig = ImageGeometry(10,10)                                               
        d1 = ig.allocate(1)                                                     
        d2 = ig.allocate(2)                                                     
        out = ig.allocate(None)
        a = 2                                                 
        b = 1                                            
        # equals to 2 * [1] + 1 * [2] = [4]
        d1.axpby(a,b,d2,out)
        res = numpy.ones_like(d1.as_array()) * 4.
        numpy.testing.assert_array_equal(res, out.as_array())
    def test_axpby2(self):
        print ("test axpby2")
        N = 100
        ig = ImageGeometry(N,2*N,N*10)                                               
        d1 = ig.allocate(1)                                                     
        d2 = ig.allocate(2)                                                     
        out = ig.allocate(None)   
        a = 2                                                 
        b = 1        
        print ("allocated")                                              
        # equals to 2 * [1] + 1 * [2] = [4]
        d1.axpby(a,b,d2,out, num_threads=4)
        print ("calculated") 
        res = numpy.ones_like(d1.as_array()) * 4.
        numpy.testing.assert_array_equal(res, out.as_array())
    def test_axpby3(self):
        print ("test axpby3")
        #a vec, b float
        ig = ImageGeometry(10,10)                                               
        d1 = ig.allocate(1)                                                     
        d2 = ig.allocate(2)     
        a = ig.allocate(2)                                                  
        b = 1                                               
        out = ig.allocate(None)                                                 
        # equals to 2 * [1] + 1 * [2] = [4]
        d1.axpby(a,b,d2,out)
        res = numpy.ones_like(d1.as_array()) * 4.
        numpy.testing.assert_array_equal(res, out.as_array())
    def test_axpby4(self):
        print ("test axpby4")
        #a float, b vec
        ig = ImageGeometry(10,10)                                               
        d1 = ig.allocate(1)                                                     
        d2 = ig.allocate(2)     
        a = 2                                                  
        b = ig.allocate(1)                                                 
        out = ig.allocate(None)                                                 
        # equals to 2 * [1] + 1 * [2] = [4]
        d1.axpby(a,b,d2,out)
        res = numpy.ones_like(d1.as_array()) * 4.
        numpy.testing.assert_array_equal(res, out.as_array())
    def test_axpby5(self):
        print ("test axpby5")
        #a vec, b vec
        ig = ImageGeometry(10,10)                                               
        d1 = ig.allocate(1)                                                     
        d2 = ig.allocate(2)   
        a = ig.allocate(2)                                                  
        b = ig.allocate(1)                                                  
        out = ig.allocate(None)                                                 
        # equals to 2 * [1] + 1 * [2] = [4]
        d1.axpby(a,b,d2,out)
        res = numpy.ones_like(d1.as_array()) * 4.
        numpy.testing.assert_array_equal(res, out.as_array())
    def test_axpby6(self):
        print ("test axpby6")
        #a vec, b vec
        ig = ImageGeometry(10,10)                                               
        d1 = ig.allocate()                                                     
        d2 = ig.allocate()   
        a = ig.allocate()                                                  
        b = ig.allocate()         

        d1.fill(numpy.arange(1,101).reshape(10,10))
        d2.fill(numpy.arange(1,101).reshape(10,10))
        a.fill(1.0/d1.as_array())                                                  
        b.fill(-1.0/d2.as_array())   

        out = ig.allocate(None)                                                 
        # equals to 1 + -1 = 0
        d1.axpby(a,b,d2,out)
        res = numpy.zeros_like(d1.as_array())
        numpy.testing.assert_array_equal(res, out.as_array())
    def test_axpby7(self):
        print ("test axpby7")
        #a vec, b vec
        #daxpby
        ig = ImageGeometry(10,10)                                               
        d1 = ig.allocate()                                                     
        d2 = ig.allocate()   
        a = ig.allocate()                                                  
        b = ig.allocate()         

        d1.fill(numpy.arange(1,101).reshape(10,10))
        d2.fill(numpy.arange(1,101).reshape(10,10))
        a.fill(1.0/d1.as_array())                                                  
        b.fill(-1.0/d2.as_array())   

        out = ig.allocate(dtype=numpy.float64)                                                 
        # equals to 1 + -1 = 0
        d1.axpby(a,b,d2,out, dtype=numpy.float64)
        res = numpy.zeros_like(d1.as_array())
        numpy.testing.assert_array_equal(res, out.as_array())


    def test_sapyb_datacontainer_f(self):
        #a vec, b vec
        
        ig = ImageGeometry(10,10)                                               
        d1 = ig.allocate(dtype=numpy.float32)                                                     
        d2 = ig.allocate(dtype=numpy.float32)   
        a = ig.allocate(dtype=numpy.float32)                                                  
        b = ig.allocate(dtype=numpy.float32)         

        d1.fill(numpy.asarray(numpy.arange(1,101).reshape(10,10), dtype=numpy.float32))
        d2.fill(numpy.asarray(numpy.arange(1,101).reshape(10,10), dtype=numpy.float32))
        a.fill(1.0/d1.as_array())                                                  
        b.fill(-1.0/d2.as_array())   

        out = ig.allocate(-1,dtype=numpy.float32)                                                 
        # equals to 1 + -1 = 0
        out = d1.sapyb(a,d2,b)
        res = numpy.zeros_like(d1.as_array())
        numpy.testing.assert_array_equal(res, out.as_array())

        out.fill(0)
        d1.sapyb(a,d2,b, out)
        res = numpy.zeros_like(d1.as_array())
        numpy.testing.assert_array_equal(res, out.as_array())

    def test_sapyb_scalar_f(self):
        # a,b scalar
        ig = ImageGeometry(10,10)                                               
        d1 = ig.allocate(1, dtype=numpy.float32)                                                     
        d2 = ig.allocate(2, dtype=numpy.float32)   
        a = 2.
        b = -1.

        out = ig.allocate(-1,dtype=numpy.float32)                                                 
        # equals to 2*[1] + -1*[2] = 0
        out = d1.sapyb(a,d2,b)
        res = numpy.zeros_like(d1.as_array())
        numpy.testing.assert_array_equal(res, out.as_array())

        out.fill(0)
        d1.sapyb(a,d2,b, out)
        numpy.testing.assert_array_equal(res, out.as_array())

        d1.sapyb(a,d2,b, out=d1)
        numpy.testing.assert_array_equal(res, d1.as_array())

        d1.fill(1)
        d1.sapyb(a,d2,b, out=d2)
        numpy.testing.assert_array_equal(res, d2.as_array())
    
    def test_sapyb_datacontainer_scalar_f(self):
        #mix: a scalar and b DataContainer and a DataContainer and b scalar
        ig = ImageGeometry(10,10)                                               
        d1 = ig.allocate(1., dtype=numpy.complex64)                                                     
        d2 = ig.allocate(2.,dtype=numpy.complex64)   
        a = 2.+2j                                                
        b = ig.allocate(-1.-1j, dtype=numpy.complex64)         


        out = ig.allocate(-1,dtype=numpy.complex64)
        # equals to (2+2j)*[1] + -(1+j)*[2] = 0
        
        out = d1.sapyb(a,d2,b)
        res = ig.allocate(0, dtype=numpy.complex64)
        numpy.testing.assert_array_equal(res.as_array(), out.as_array())

        out.fill(-1)
        d1.sapyb(a,d2,b, out)
        numpy.testing.assert_array_equal(res.as_array(), out.as_array())

        out = d2.sapyb(b,d1,a)
        res = ig.allocate(0, dtype=numpy.complex64)
        numpy.testing.assert_array_equal(res.as_array(), out.as_array())

        out.fill(-1)
        d2.sapyb(b,d1,a, out)
        numpy.testing.assert_array_equal(res.as_array(), out.as_array())

    def test_sapyb_scalar_c(self):
        # a, b scalar
        ig = ImageGeometry(10,10)                                               
        d1 = ig.allocate(1, dtype=numpy.complex64)                                                     
        d2 = ig.allocate(2, dtype=numpy.complex64)   
        a = 2.+2j
        b = -1.-1j

        out = ig.allocate(-1,dtype=numpy.complex64)                                                 
        # equals to (2+2j)*[1] + -(1+j)*[2] = 0
        out = d1.sapyb(a,d2,b)
        res = numpy.zeros_like(d1.as_array())
        numpy.testing.assert_array_equal(res, out.as_array())

        out.fill(0)
        d1.sapyb(a,d2,b, out)
        numpy.testing.assert_array_equal(res, out.as_array())

        d1.sapyb(a,d2,b, out=d1)
        numpy.testing.assert_array_equal(res, d1.as_array())

        d1.fill(1)
        d1.sapyb(a,d2,b, out=d2)
        numpy.testing.assert_array_equal(res, d2.as_array())


    def test_sapyb_datacontainer_c(self):
        #a vec, b vec
        ig = ImageGeometry(10,10)                                               
        d1 = ig.allocate(dtype=numpy.complex64)                                                     
        d2 = ig.allocate(dtype=numpy.complex64)   
        a = ig.allocate(dtype=numpy.complex64)                                                  
        b = ig.allocate(dtype=numpy.complex64)         

        arr = numpy.empty(ig.shape, dtype=numpy.complex64)
        arr.real = numpy.asarray(numpy.arange(1,101).reshape(10,10), dtype=numpy.float32)
        arr.imag = numpy.asarray(numpy.arange(1,101).reshape(10,10), dtype=numpy.float32)

        d1.fill(arr)

        arr.imag = -1* arr.imag
        d2.fill(arr)

        a.fill(d2.as_array())                                                  
        b.fill(d1.as_array())   

        out = ig.allocate(-1,dtype=numpy.complex64)
        # equals to d1^ * d1 + d2^*d2 = d1**2 + d2**2 = 2* arr.norm = 2 * (arr.real **2 + arr.imag **2)
        out = d1.sapyb(a,d2,b)
        res = 2* (arr.real * arr.real + arr.imag * arr.imag)
        numpy.testing.assert_array_equal(res, out.as_array())

        out.fill(0)
        d1.sapyb(a,d2,b, out)
        numpy.testing.assert_array_equal(res, out.as_array())


    def test_sapyb_datacontainer_scalar_c(self):
        #mix: a scalar and b DataContainer and a DataContainer and b scalar
        ig = ImageGeometry(10,10)                                               
        d1 = ig.allocate(1., dtype=numpy.complex64)                                                     
        d2 = ig.allocate(2.,dtype=numpy.complex64)   
        a = 2.+2j                                                
        b = ig.allocate(-1.-1j, dtype=numpy.complex64)         


        out = ig.allocate(-1,dtype=numpy.complex64)
        # equals to (2+2j)*[1] + -(1+j)*[2] = 0
        
        out = d1.sapyb(a,d2,b)
        res = ig.allocate(0, dtype=numpy.complex64)
        numpy.testing.assert_array_equal(res.as_array(), out.as_array())

        out.fill(-1)
        d1.sapyb(a,d2,b, out)
        numpy.testing.assert_array_equal(res.as_array(), out.as_array())

        out = d2.sapyb(b,d1,a)
        res = ig.allocate(0, dtype=numpy.complex64)
        numpy.testing.assert_array_equal(res.as_array(), out.as_array())

        out.fill(-1)
        d2.sapyb(b,d1,a, out)
        numpy.testing.assert_array_equal(res.as_array(), out.as_array())

    def test_sapyb_scalar_f_c(self):
        # a,b scalar
        ig = ImageGeometry(10,10)                                               
        d1 = ig.allocate(1, dtype=numpy.complex64)                                                     
        d2 = ig.allocate(2, dtype=numpy.float32)   
        a = 2.+1j
        b = -1.

        # equals to 2*[1] + -1*[2] = 0
        out = d1.sapyb(a,d2,b)
        res = numpy.zeros_like(d1.as_array()) + 1j
        numpy.testing.assert_array_equal(res, out.as_array())

        out.fill(0)
        d1.sapyb(a,d2,b, out)
        numpy.testing.assert_array_equal(res, out.as_array())

        d1.sapyb(a,d2,b, out=d1)
        numpy.testing.assert_array_equal(res, d1.as_array())

        d1.fill(1)
        try:

            with self.assertRaises(numpy.core._exceptions.UFuncTypeError) as context:
                d1.sapyb(a,d2,b, out=d2)
        except AttributeError as ae:
            print ("Probably numpy version too low:", ae)

        # print ("Exception thrown:", str(context.exception))
        
        # out is complex
        # d1.fill(1+0j)
        d2.fill(2)
        d1.sapyb(a,d2,b,out=d1)
        # 2+1j * [1+0j] -1 * [2]
        numpy.testing.assert_array_equal(1j * numpy.ones_like(d1.as_array()), d1.as_array())
            
    def test_min(self):
        print ("test min")
        ig = ImageGeometry(10,10)     
        a = numpy.asarray(numpy.linspace(-10,10, num=100, endpoint=True), dtype=numpy.float32)
        a = a.reshape((10,10))
        d1 = ig.allocate(1)                                                     
        d1.fill(a)                                                     
        self.assertAlmostEqual(d1.min(), -10.)

    def test_max(self):
        print ("test max")
        ig = ImageGeometry(10,10)     
        a = numpy.asarray(numpy.linspace(-10,10, num=100, endpoint=True), dtype=numpy.float32)
        a = a.reshape((10,10))
        d1 = ig.allocate(1)                                                     
        d1.fill(a)                                                     
        self.assertAlmostEqual(d1.max(), 10.)

    def test_size(self):
        print ("test size")
        ig = ImageGeometry(10,10)     
        d1 = ig.allocate(1)                                                     
                                                
        self.assertEqual( d1.size, 100 )
        
        sgeometry = AcquisitionGeometry(dimension=2, angles=numpy.linspace(0, 180, num=10),
                                        geom_type='parallel', pixel_num_v=3,
                                        pixel_num_h=5, channels=2)
        ad = sgeometry.allocate()

        self.assertEqual( ad.size, 3*5*10*2 )
    
    def test_negation(self):
        X, Y, Z = 256, 512, 512
        a = numpy.ones((X, Y, Z), dtype='int32')
        
        ds = - DataContainer(a, False, ['X', 'Y', 'Z'])
        
        numpy.testing.assert_array_equal(ds.as_array(), -a)

    def test_fill_dimension_ImageData(self):
        ig = ImageGeometry(2,3,4)
        u = ig.allocate(0)
        a = numpy.ones((4,2))
        # default_labels = [ImageGeometry.VERTICAL, ImageGeometry.HORIZONTAL_Y, ImageGeometry.HORIZONTAL_X]
        
        data = u.as_array()
        axis_number = u.get_dimension_axis('horizontal_y')
        
        u.fill(a, horizontal_y=0)
        numpy.testing.assert_array_equal(u.subset(horizontal_y=0).as_array(), a)

        u.fill(2, horizontal_y=1)
        numpy.testing.assert_array_equal(u.subset(horizontal_y=1).as_array(), 2 * a)

        u.fill(2, horizontal_y=1)
        numpy.testing.assert_array_equal(u.subset(horizontal_y=1).as_array(), 2 * a)
        
        b = u.subset(horizontal_y=2)
        b.fill(3)
        u.fill(b, horizontal_y=2)
        numpy.testing.assert_array_equal(u.subset(horizontal_y=2).as_array(), 3 * a)

    def test_fill_dimension_AcquisitionData(self):
        ag = AcquisitionGeometry.create_Parallel3D()
        ag.set_channels(4)
        ag.set_panel([2,3])
        ag.set_angles([0,1,2,3,5])
        ag.set_labels(('horizontal','angle','vertical','channel'))
        u = ag.allocate(0)
        a = numpy.ones((4,2))
        # default_labels = [ImageGeometry.VERTICAL, ImageGeometry.HORIZONTAL_Y, ImageGeometry.HORIZONTAL_X]
        
        data = u.as_array()
        axis_number = u.get_dimension_axis('horizontal_y')
        
        u.fill(a, horizontal_y=0)
        numpy.testing.assert_array_equal(u.subset(horizontal_y=0).as_array(), a)

        u.fill(2, horizontal_y=1)
        numpy.testing.assert_array_equal(u.subset(horizontal_y=0).as_array(), 2 * a)

        u.fill(2, horizontal_y=1)
        numpy.testing.assert_array_equal(u.subset(horizontal_y=1).as_array(), 2 * a)
        
        b = u.subset(horizontal_y=2)
        b.fill(3)
        u.fill(b, horizontal_y=2)
        numpy.testing.assert_array_equal(u.subset(horizontal_y=2).as_array(), 3 * a)

        # slice with 2 axis
        a = numpy.ones((2,))
        u.fill(a, horizontal_y=1, vertical=0)
        numpy.testing.assert_array_equal(u.subset(horizontal_y=1, vertical=0).as_array(), a)


    def test_fill_dimension_AcquisitionData(self):
        ag = AcquisitionGeometry.create_Parallel3D()
        ag.set_channels(4)
        ag.set_panel([2,3])
        ag.set_angles([0,1,2,3,5])
        ag.set_labels(('horizontal','angle','vertical','channel'))
        u = ag.allocate(0)
        print (u.shape)
        # (2, 5, 3, 4)
        a = numpy.ones((2,5))
        # default_labels = [ImageGeometry.VERTICAL, ImageGeometry.HORIZONTAL_Y, ImageGeometry.HORIZONTAL_X]
        b = u.subset(channel=0, vertical=0)
        print(b.shape)
        data = u.as_array()
        
        u.fill(a, channel=0, vertical=0)
        print(u.shape)
        numpy.testing.assert_array_equal(u.subset(channel=0, vertical=0).as_array(), a)

        u.fill(2, channel=0, vertical=0)
        numpy.testing.assert_array_equal(u.subset(channel=0, vertical=0).as_array(), 2 * a)

        u.fill(2, channel=0, vertical=0)
        numpy.testing.assert_array_equal(u.subset(channel=0, vertical=0).as_array(), 2 * a)
        
        b = u.subset(channel=0, vertical=0)
        b.fill(3)
        u.fill(b, channel=1, vertical=1)
        numpy.testing.assert_array_equal(u.subset(channel=1, vertical=1).as_array(), 3 * a)
if __name__ == '__main__':
    unittest.main()
 
