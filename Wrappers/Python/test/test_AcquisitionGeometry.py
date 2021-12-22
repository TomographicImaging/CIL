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
import math
from cil.framework import AcquisitionGeometry, ImageGeometry

class Test_AcquisitionGeometry(unittest.TestCase):
    def test_create_Parallel2D(self):

        #default
        AG = AcquisitionGeometry.create_Parallel2D()
        numpy.testing.assert_allclose(AG.config.system.ray.direction, [0,1], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.detector.position, [0,0], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.detector.direction_x, [1,0], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.rotation_axis.position, [0,0], rtol=1E-6)

        #values
        ray_direction = [0.1, 3.0]
        detector_position = [-1.3,1000.0]
        detector_direction_x = [1,0.2]
        rotation_axis_position = [0.1,2]

        AG = AcquisitionGeometry.create_Parallel2D(ray_direction, detector_position, detector_direction_x, rotation_axis_position)

        ray_direction = numpy.asarray(ray_direction)
        detector_direction_x = numpy.asarray(detector_direction_x)

        ray_direction /= numpy.sqrt((ray_direction**2).sum())
        detector_direction_x /= numpy.sqrt((detector_direction_x**2).sum())

        numpy.testing.assert_allclose(AG.config.system.ray.direction, ray_direction, rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.detector.position, detector_position, rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.detector.direction_x, detector_direction_x, rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.rotation_axis.position, rotation_axis_position, rtol=1E-6)
    
    def test_create_Parallel3D(self):

        #default
        AG = AcquisitionGeometry.create_Parallel3D()
        numpy.testing.assert_allclose(AG.config.system.ray.direction, [0,1,0], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.detector.position, [0,0,0], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.detector.direction_x, [1,0,0], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.detector.direction_y, [0,0,1], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.rotation_axis.position, [0,0,0], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.rotation_axis.direction, [0,0,1], rtol=1E-6)

        #values
        ray_direction = [0.1, 3.0, 0.2]
        detector_position = [-1.3,1000.0, -1.0]
        detector_direction_x = [1,0.2, 0]
        detector_direction_y = [0.0,0,1]
        rotation_axis_position=[0.1, 2,-0.4]
        rotation_axis_direction=[-0.1,-0.3,1]

        AG = AcquisitionGeometry.create_Parallel3D(ray_direction, detector_position, detector_direction_x,detector_direction_y, rotation_axis_position,rotation_axis_direction)

        ray_direction = numpy.asarray(ray_direction)
        detector_direction_x = numpy.asarray(detector_direction_x)
        detector_direction_y = numpy.asarray(detector_direction_y)
        rotation_axis_direction = numpy.asarray(rotation_axis_direction)

        ray_direction /= numpy.sqrt((ray_direction**2).sum())
        detector_direction_x /= numpy.sqrt((detector_direction_x**2).sum())
        detector_direction_y /= numpy.sqrt((detector_direction_y**2).sum())
        rotation_axis_direction /= numpy.sqrt((rotation_axis_direction**2).sum())

        numpy.testing.assert_allclose(AG.config.system.ray.direction, ray_direction, rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.detector.position, detector_position, rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.detector.direction_x, detector_direction_x, rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.detector.direction_y, detector_direction_y, rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.rotation_axis.position, rotation_axis_position, rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.rotation_axis.direction, rotation_axis_direction, rtol=1E-6)


    def test_create_Cone2D(self):
        #default
        source_position = [0.1, -500.0]
        detector_position = [-1.3,1000.0]

        AG = AcquisitionGeometry.create_Cone2D(source_position, detector_position)
        numpy.testing.assert_allclose(AG.config.system.source.position, source_position, rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.detector.position, detector_position, rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.detector.direction_x, [1,0], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.rotation_axis.position, [0,0], rtol=1E-6)

        #values
        detector_direction_x = [1,0.2]
        rotation_axis_position = [0.1,2]

        AG = AcquisitionGeometry.create_Cone2D(source_position, detector_position, detector_direction_x, rotation_axis_position)

        detector_direction_x = numpy.asarray(detector_direction_x)
        detector_direction_x /= numpy.sqrt((detector_direction_x**2).sum())

        numpy.testing.assert_allclose(AG.config.system.source.position, source_position, rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.detector.position, detector_position, rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.detector.direction_x, detector_direction_x, rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.rotation_axis.position, rotation_axis_position, rtol=1E-6)
    
    def test_create_Cone3D(self):

        #default
        source_position = [0.1, -500.0,-2.0]
        detector_position = [-1.3,1000.0, -1.0]

        AG = AcquisitionGeometry.create_Cone3D(source_position, detector_position)
        numpy.testing.assert_allclose(AG.config.system.source.position, source_position, rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.detector.position, detector_position, rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.detector.direction_x, [1,0,0], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.detector.direction_y, [0,0,1], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.rotation_axis.position, [0,0,0], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.rotation_axis.direction, [0,0,1], rtol=1E-6)

        #values
        detector_direction_x = [1,0.2, 0]
        detector_direction_y = [0.0,0,1]
        rotation_axis_position=[0.1, 2,-0.4]
        rotation_axis_direction=[-0.1,-0.3,1]

        AG = AcquisitionGeometry.create_Cone3D(source_position, detector_position, detector_direction_x,detector_direction_y, rotation_axis_position,rotation_axis_direction)

        detector_direction_x = numpy.asarray(detector_direction_x)
        detector_direction_y = numpy.asarray(detector_direction_y)
        rotation_axis_direction = numpy.asarray(rotation_axis_direction)

        detector_direction_x /= numpy.sqrt((detector_direction_x**2).sum())
        detector_direction_y /= numpy.sqrt((detector_direction_y**2).sum())
        rotation_axis_direction /= numpy.sqrt((rotation_axis_direction**2).sum())

        numpy.testing.assert_allclose(AG.config.system.source.position, source_position, rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.detector.position, detector_position, rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.detector.direction_x, detector_direction_x, rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.detector.direction_y, detector_direction_y, rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.rotation_axis.position, rotation_axis_position, rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.rotation_axis.direction, rotation_axis_direction, rtol=1E-6)

    def test_SystemConfiguration(self):
        
        #SystemConfiguration error handeling
        AG = AcquisitionGeometry.create_Parallel3D()  

        #vector wrong length
        with self.assertRaises(ValueError):
            AG.config.system.detector.position = [-0.1,0.1]

        #detector xs and yumns should be perpendicular
        with self.assertRaises(ValueError):
            AG.config.system.detector.set_direction([1,0,0],[-0.1,0.1,1])

    def test_set_angles(self):

        AG = AcquisitionGeometry.create_Parallel2D()
        angles = numpy.linspace(0, 360, 10, dtype=numpy.float32)

        #default
        AG.set_angles(angles)
        numpy.testing.assert_allclose(AG.config.angles.angle_data, angles, rtol=1E-6)
        self.assertEqual(AG.config.angles.initial_angle, 0.0)
        self.assertEqual(AG.config.angles.angle_unit, 'degree')

        #values        
        AG.set_angles(angles, 0.1, 'radian')
        numpy.testing.assert_allclose(AG.config.angles.angle_data, angles, rtol=1E-6)
        self.assertEqual(AG.config.angles.initial_angle, 0.1)
        self.assertEqual(AG.config.angles.angle_unit, 'radian')

    def test_set_panel(self):
        AG = AcquisitionGeometry.create_Parallel3D()

        #default
        AG.set_panel([1000,2000])
        numpy.testing.assert_array_equal(AG.config.panel.num_pixels, [1000,2000])
        numpy.testing.assert_array_almost_equal(AG.config.panel.pixel_size, [1,1])

        #values
        AG.set_panel([1000,2000],[0.1,0.2])
        numpy.testing.assert_array_equal(AG.config.panel.num_pixels, [1000,2000])
        numpy.testing.assert_array_almost_equal(AG.config.panel.pixel_size, [0.1,0.2])

        #set 2D panel with 3D geometry
        with self.assertRaises(ValueError):
            AG.config.panel.num_pixels = [5]

    def test_set_channels(self):
        AG = AcquisitionGeometry.create_Parallel2D()

        #default
        AG.set_channels()
        self.assertEqual(AG.config.channels.num_channels, 1)

        #values
        AG.set_channels(3, ['r','g','b'])
        self.assertEqual(AG.config.channels.num_channels, 3)
        self.assertEqual(AG.config.channels.channel_labels, ['r','g','b'])

        #set wrong length list
        with self.assertRaises(ValueError):
            AG.config.channels.channel_labels = ['a']

    def test_set_labels(self):

        AG = AcquisitionGeometry.create_Parallel3D()
        AG.set_channels(4)
        AG.set_panel([2,3])
        AG.set_angles([0,1,2,3,5])

        #default
        self.assertEqual(AG.dimension_labels, ('channel','angle','vertical','horizontal'))
        self.assertEqual(AG.shape, (4,5,3,2))

        #values
        AG.set_angles([0])
        AG.set_labels(('horizontal','channel','vertical'))
        self.assertEqual(AG.dimension_labels, ('horizontal','channel','vertical'))
        self.assertEqual(AG.shape, (2,4,3))

    def test_equal(self):
        AG = AcquisitionGeometry.create_Parallel3D()
        AG.set_channels(4, ['a','b','c','d'])
        AG.set_panel([2,3])
        AG.set_angles([0,1,2,3,5])
        AG.set_labels(('horizontal','angle','vertical','channel'))

        AG2 = AcquisitionGeometry.create_Parallel3D()
        AG2.set_channels(4, ['a','b','c','d'])
        AG2.set_panel([2,3])
        AG2.set_angles([0,1,2,3,5])
        AG2.set_labels(('horizontal','angle','vertical','channel'))
        self.assertTrue(AG == AG2)

        #test not equal
        AG3 = AG2.copy()
        AG3.config.system.ray.direction = [1,0,0]
        self.assertFalse(AG == AG3)

        AG3 = AG2.copy()
        AG3.config.panel.num_pixels = [1,2]
        self.assertFalse(AG == AG3)

        AG3 = AG2.copy()
        AG3.config.channels.channel_labels = ['d','b','c','d']
        self.assertFalse(AG == AG3)

        AG3 = AG2.copy()
        AG3.config.angles.angle_unit ='radian'
        self.assertFalse(AG == AG3)

        AG3 = AG2.copy()
        AG3.config.angles.angle_data[0] = -1
        self.assertFalse(AG == AG3)

    def test_clone(self):
        AG = AcquisitionGeometry.create_Parallel3D()
        AG.set_channels(4)
        AG.set_panel([2,3])
        AG.set_angles([0,1,2,3,5])
        AG.set_labels(('horizontal','angle','vertical','channel'))

        AG2 = AG.clone()
        self.assertEqual(AG2, AG)

    def test_copy(self):
        AG = AcquisitionGeometry.create_Parallel3D()
        AG.set_channels(4)
        AG.set_panel([2,3])
        AG.set_angles([0,1,2,3,5])
        AG.set_labels(('horizontal','angle','vertical','channel'))

        AG2 = AG.copy()
        self.assertEqual(AG2, AG)

    def test_get_centre_slice(self):
        AG = AcquisitionGeometry.create_Parallel3D(detector_direction_y=[0,1,1])
        AG.set_panel([1000,2000],[1,1])
        AG_cs = AG.get_centre_slice()

        AG2 = AcquisitionGeometry.create_Parallel2D()
        AG2.set_panel([1000,1],[1,math.sqrt(0.5)])

        self.assertEqual(AG2, AG_cs)

    def test_allocate(self):
        AG = AcquisitionGeometry.create_Parallel3D()
        AG.set_channels(4)
        AG.set_panel([2,3])
        AG.set_angles([0,1,2,3,5])
        AG.set_labels(('horizontal','angle','vertical','channel'))

        test = AG.allocate()
        test2 = numpy.ndarray([2,5,3,4])
        self.assertEqual(test.shape, test2.shape)

    def test_get_ImageGeometry(self):

        AG = AcquisitionGeometry.create_Parallel2D()\
            .set_panel(num_pixels=[512,1],pixel_size=[0.1,0.1])      
        IG = AG.get_ImageGeometry()
        IG_gold = ImageGeometry(512,512,0,0.1,0.1,1,0,0,0,1)
        self.assertEqual(IG, IG_gold)

        AG = AcquisitionGeometry.create_Parallel3D()\
            .set_panel(num_pixels=[512,3],pixel_size=[0.1,0.2])
        IG = AG.get_ImageGeometry()
        IG_gold = ImageGeometry(512,512,3,0.1,0.1,0.2,0,0,0,1)
        self.assertEqual(IG, IG_gold)

        AG = AcquisitionGeometry.create_Cone2D(source_position=[0,-500], detector_position=[0.,500.])\
            .set_panel(num_pixels=[512,1],pixel_size=[0.1,0.2])
        IG = AG.get_ImageGeometry()
        IG_gold = ImageGeometry(512,512,0,0.05,0.05,1,0,0,0,1)
        self.assertEqual(IG, IG_gold)

        AG = AcquisitionGeometry.create_Cone3D(source_position=[0,-500,0], detector_position=[0.,500.,0])\
            .set_panel(num_pixels=[512,3],pixel_size=[0.1,0.2])
        IG = AG.get_ImageGeometry()
        IG_gold = ImageGeometry(512,512,3,0.05,0.05,0.1,0,0,0,1)
        self.assertEqual(IG, IG_gold)

        AG = AcquisitionGeometry.create_Cone3D(source_position=[0,-500,0], detector_position=[0.,500.,0])\
            .set_panel(num_pixels=[512,3],pixel_size=[0.1,0.2])
        IG = AG.get_ImageGeometry(resolution=0.5)
        IG_gold = ImageGeometry(256,256,2,0.1,0.1,0.2,0,0,0,1)
        self.assertEqual(IG, IG_gold)

        AG = AcquisitionGeometry.create_Cone3D(source_position=[0,-500,0], detector_position=[0.,500.,0])\
            .set_panel(num_pixels=[512,3],pixel_size=[0.1,0.2])
        IG = AG.get_ImageGeometry(resolution=2)
        IG_gold = ImageGeometry(1024,1024,6,0.025,0.025,0.05,0,0,0,1)
        self.assertEqual(IG, IG_gold)


class Test_Parallel2D(unittest.TestCase):
    
    def test_update_reference_frame(self):
        AG = AcquisitionGeometry.create_Parallel2D(detector_position=[0.,1000.], rotation_axis_position=[5.,2.])
        AG.config.system.update_reference_frame()

        numpy.testing.assert_allclose(AG.config.system.ray.direction, [0,1], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.detector.position, [-5,998], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.detector.direction_x, [1,0], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.rotation_axis.position, [0,0], rtol=1E-6)

    def test_align_reference_frame(self):
        AG = AcquisitionGeometry.create_Parallel2D(ray_direction=[0,-1], detector_position=[0.,-100.], rotation_axis_position=[10.,5.])
        AG.config.system.align_reference_frame()

        numpy.testing.assert_allclose(AG.config.system.ray.direction, [0,1], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.detector.position, [10,105], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.detector.direction_x, [-1,0], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.rotation_axis.position, [0,0], rtol=1E-6)

    def test_system_description(self):
        AG = AcquisitionGeometry.create_Parallel2D()
        self.assertTrue(AG.system_description=='simple')

        AG = AcquisitionGeometry.create_Parallel2D(detector_position=[5,0], rotation_axis_position=[5,0])
        self.assertTrue(AG.system_description=='simple')

        AG = AcquisitionGeometry.create_Parallel2D(ray_direction=[1,1])
        self.assertTrue(AG.system_description=='advanced')

        AG = AcquisitionGeometry.create_Parallel2D(rotation_axis_position=[5,0])
        self.assertTrue(AG.system_description=='offset')

    def test_get_centre_slice(self):
        AG = AcquisitionGeometry.create_Parallel2D()
        AG2 = AG.copy()

        AG2.config.system.get_centre_slice()
        self.assertEqual(AG.config.system, AG2.config.system)

    def test_calculate_magnification(self):
        AG = AcquisitionGeometry.create_Parallel2D()
        out = AG.config.system.calculate_magnification()
        self.assertEqual(out, [None, None, 1]) 

class Test_Parallel3D(unittest.TestCase):
    
    def test_update_reference_frame(self):
        #translate origin
        AG = AcquisitionGeometry.create_Parallel3D(detector_position=[0.,1000.,0], rotation_axis_position=[5.,2.,4.])
        AG.config.system.update_reference_frame()
        numpy.testing.assert_allclose(AG.config.system.ray.direction, [0,1,0], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.detector.position, [-5,998,-4], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.detector.direction_x, [1,0,0], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.rotation_axis.position, [0,0,0], rtol=1E-6)

        #align Z axis with rotate axis
        AG = AcquisitionGeometry.create_Parallel3D(detector_position=[0.,1000.,0], rotation_axis_position=[0.,0.,0.], rotation_axis_direction=[0,1,0])
        AG.config.system.update_reference_frame()
        numpy.testing.assert_allclose(AG.config.system.ray.direction, [0,0,1], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.detector.position, [0,0,1000], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.detector.direction_x, [1,0,0], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.detector.direction_y, [0,-1,0], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.rotation_axis.position, [0,0,0], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.rotation_axis.direction, [0,0,1], rtol=1E-6)

    def test_align_reference_frame(self):
        AG = AcquisitionGeometry.create_Parallel3D(ray_direction=[0,-1,0], detector_position=[0.,-100.,0], rotation_axis_position=[10.,5.,0], rotation_axis_direction=[0,0,-1])
        AG.config.system.align_reference_frame()

        numpy.testing.assert_allclose(AG.config.system.ray.direction, [0,1, 0], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.detector.position, [-10,105,0], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.detector.direction_x, [1,0,0], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.detector.direction_y, [0,0,-1], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.rotation_axis.position, [0,0,0], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.rotation_axis.direction, [0,0,1], rtol=1E-6)

    def test_system_description(self):
        AG = AcquisitionGeometry.create_Parallel3D()
        self.assertTrue(AG.system_description=='simple')

        AG = AcquisitionGeometry.create_Parallel3D(detector_position=[5,0,0], rotation_axis_position=[5,0,0])
        self.assertTrue(AG.system_description=='simple')

        AG = AcquisitionGeometry.create_Parallel3D(rotation_axis_position=[5,0,0])
        self.assertTrue(AG.system_description=='offset')

        AG = AcquisitionGeometry.create_Parallel3D(ray_direction = [1,1,0])
        self.assertTrue(AG.system_description=='advanced')


    def test_get_centre_slice(self):
        #returns the 2D version
        AG = AcquisitionGeometry.create_Parallel3D()
        AG2 = AcquisitionGeometry.create_Parallel2D()
        cs = AG.config.system.get_centre_slice()
        self.assertEqual(cs, AG2.config.system)

        #returns the 2D version
        AG = AcquisitionGeometry.create_Parallel3D(rotation_axis_direction=[-1,0,1],detector_direction_x=[1,0,1], detector_direction_y=[-1,0,1])
        AG2 = AcquisitionGeometry.create_Parallel2D()
        cs = AG.config.system.get_centre_slice()
        self.assertEqual(cs, AG2.config.system)

        #raise error if cannot extract a cnetre slice
        AG = AcquisitionGeometry.create_Parallel3D(ray_direction=[0,1,1])
        with self.assertRaises(ValueError):
            cs = AG.config.system.get_centre_slice()

        AG = AcquisitionGeometry.create_Parallel3D(detector_direction_x=[1,0,1], detector_direction_y=[-1,0,1])
        with self.assertRaises(ValueError):
            cs = AG.config.system.get_centre_slice()

    def test_calculate_magnification(self):
        AG = AcquisitionGeometry.create_Parallel3D()
        out = AG.config.system.calculate_magnification()
        self.assertEqual(out, [None, None, 1]) 

class Test_Cone2D(unittest.TestCase):
    
    def test_update_reference_frame(self):
        AG = AcquisitionGeometry.create_Cone2D(source_position=[0,-500], detector_position=[0.,1000.], rotation_axis_position=[5.,2.])
        AG.config.system.update_reference_frame()

        numpy.testing.assert_allclose(AG.config.system.source.position, [-5,-502], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.detector.position, [-5,998], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.detector.direction_x, [1,0], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.rotation_axis.position, [0,0], rtol=1E-6)

    def test_align_reference_frame(self):
        AG = AcquisitionGeometry.create_Cone2D(source_position=[5,-400], detector_position=[5.,500.], rotation_axis_position=[5.,0])
        AG.config.system.align_reference_frame()

        numpy.testing.assert_allclose(AG.config.system.source.position, [0,-400], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.detector.position, [0,500], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.detector.direction_x, [1,0], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.rotation_axis.position, [0,0], rtol=1E-6)

    def test_system_description(self):
        AG = AcquisitionGeometry.create_Cone2D(source_position = [0,-50],detector_position=[0,100])
        self.assertTrue(AG.system_description=='simple')

        AG = AcquisitionGeometry.create_Cone2D(source_position = [5,-50],detector_position=[5,100], rotation_axis_position=[5,0])
        self.assertTrue(AG.system_description=='simple')

        AG = AcquisitionGeometry.create_Cone2D(source_position = [5,-50],detector_position=[0,100])
        self.assertTrue(AG.system_description=='advanced')

        AG = AcquisitionGeometry.create_Cone2D(source_position = [0,-50],detector_position=[0,100], rotation_axis_position=[5,0])
        self.assertTrue(AG.system_description=='offset')

    def test_get_centre_slice(self):
        AG = AcquisitionGeometry.create_Cone2D(source_position=[0,-500], detector_position=[0.,1000.])
        AG2 = AG.copy()

        AG2.config.system.get_centre_slice()
        self.assertEqual(AG.config.system, AG2.config.system)

    def test_calculate_magnification(self):
        AG = AcquisitionGeometry.create_Cone2D(source_position=[0,-500], detector_position=[0.,1000.])
        out = AG.config.system.calculate_magnification()
        self.assertEqual(out, [500, 1000, 3]) 

        AG = AcquisitionGeometry.create_Cone2D(source_position=[0,-500], detector_position=[0.,1000.], rotation_axis_position=[0.,250.])
        out = AG.config.system.calculate_magnification()
        self.assertEqual(out, [750, 750, 2]) 

        AG = AcquisitionGeometry.create_Cone2D(source_position=[0,-500], detector_position=[0.,1000.], rotation_axis_position=[5.,0.])
        out = AG.config.system.calculate_magnification()
        source_to_object = numpy.sqrt(5.0**2 + 500.0**2)
        theta = math.atan2(5.0,500.0)
        source_to_detector = 1500.0/math.cos(theta)
        self.assertEqual(out, [source_to_object, source_to_detector - source_to_object, source_to_detector/source_to_object]) 

        AG = AcquisitionGeometry.create_Cone2D(source_position=[0,-500], detector_position=[0.,1000.], rotation_axis_position=[5.,0.],detector_direction_x=[math.sqrt(5),math.sqrt(5)])
        out = AG.config.system.calculate_magnification()
        source_to_object = numpy.sqrt(5.0**2 + 500.0**2)

        ab = (AG.config.system.rotation_axis.position - AG.config.system.source.position).astype(numpy.float64)/source_to_object

        #source_position + d * ab = detector_position + t * detector_direction_x
        #x: d *  ab[0] =  t * detector_direction_x[0]
        #y: -500 + d *  ab[1] = 1000 + t * detector_direction_x[1] 

        # t = (d *  ab[0]) / math.sqrt(5)
        # d = 1500 / (ab[1]  - ab[0])

        source_to_detector = 1500 / (ab[1]  - ab[0])

        self.assertEqual(out, [source_to_object, source_to_detector - source_to_object, source_to_detector/source_to_object]) 

class Test_Cone3D(unittest.TestCase):
    
    def test_update_reference_frame(self):
        #translate origin
        AG = AcquisitionGeometry.create_Cone3D(source_position=[0,-500,0],detector_position=[0.,1000.,0], rotation_axis_position=[5.,2.,4.])
        AG.config.system.update_reference_frame()
        numpy.testing.assert_allclose(AG.config.system.source.position, [-5,-502,-4], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.detector.position, [-5,998,-4], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.detector.direction_x, [1,0,0], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.rotation_axis.position, [0,0,0], rtol=1E-6)

        #align Z axis with rotate axis
        AG = AcquisitionGeometry.create_Cone3D(source_position=[0,-500,0],detector_position=[0.,1000.,0], rotation_axis_position=[0.,0.,0.], rotation_axis_direction=[0,1,0])
        AG.config.system.update_reference_frame()
        numpy.testing.assert_allclose(AG.config.system.source.position, [0,0,-500], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.detector.position, [0,0,1000], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.detector.direction_x, [1,0,0], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.detector.direction_y, [0,-1,0], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.rotation_axis.position, [0,0,0], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.rotation_axis.direction, [0,0,1], rtol=1E-6)

    def test_align_reference_frame(self):
        AG = AcquisitionGeometry.create_Cone3D(source_position=[5,500,0],detector_position=[5.,-1000.,0], rotation_axis_position=[5,0,0], rotation_axis_direction=[0,0,-1])
        AG.config.system.align_reference_frame()

        numpy.testing.assert_allclose(AG.config.system.source.position, [0,-500, 0], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.detector.position, [0,1000,0], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.detector.direction_x, [1,0,0], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.detector.direction_y, [0,0,-1], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.rotation_axis.position, [0,0,0], rtol=1E-6)
        numpy.testing.assert_allclose(AG.config.system.rotation_axis.direction, [0,0,1], rtol=1E-6)

    def test_system_description(self):
        AG = AcquisitionGeometry.create_Cone3D(source_position = [0,-50,0],detector_position=[0,100,0])
        self.assertTrue(AG.system_description=='simple')

        AG = AcquisitionGeometry.create_Cone3D(source_position = [-50,0,0],detector_position=[100,0,0], detector_direction_x=[0,-1,0])
        self.assertTrue(AG.system_description=='simple')

        AG = AcquisitionGeometry.create_Cone3D(source_position = [5,-50,0],detector_position=[5,100,0], rotation_axis_position=[5,0,0])
        self.assertTrue(AG.system_description=='simple')

        AG = AcquisitionGeometry.create_Cone3D(source_position = [5,-50,0],detector_position=[0,100,0])
        self.assertTrue(AG.system_description=='advanced')

        AG = AcquisitionGeometry.create_Cone3D(source_position = [0,-50,0],detector_position=[0,100,0], rotation_axis_position=[5,0,0])
        self.assertTrue(AG.system_description=='offset')

    def test_get_centre_slice(self):
        #returns the 2D version
        AG = AcquisitionGeometry.create_Cone3D(source_position=[0,-500,0], detector_position=[0,1000,0])
        AG2 = AcquisitionGeometry.create_Cone2D(source_position=[0,-500], detector_position=[0,1000])
        cs = AG.config.system.get_centre_slice()
        self.assertEqual(cs, AG2.config.system)

        #returns the 2D version
        AG = AcquisitionGeometry.create_Cone3D(source_position=[0,-500,0], detector_position=[0,1000,0], rotation_axis_direction=[-1,0,1], detector_direction_x=[1,0,1], detector_direction_y=[-1,0,1])
        AG2 = AcquisitionGeometry.create_Cone2D(source_position=[0,-500], detector_position=[0,1000])
        cs = AG.config.system.get_centre_slice()
        self.assertEqual(cs, AG2.config.system)

        #raise error if cannot extract a cnetre slice
        AG = AcquisitionGeometry.create_Cone3D(source_position=[0,-500,0], detector_position=[0,1000,0], rotation_axis_direction=[1,0,1])
        with self.assertRaises(ValueError):
            cs = AG.config.system.get_centre_slice()

        AG = AcquisitionGeometry.create_Cone3D(source_position=[0,-500,0], detector_position=[0,1000,0],detector_direction_x=[1,0,1], detector_direction_y=[-1,0,1])
        with self.assertRaises(ValueError):
            cs = AG.config.system.get_centre_slice()

    def test_calculate_magnification(self):        
        AG = AcquisitionGeometry.create_Cone3D(source_position=[0,-500,0], detector_position=[0.,1000.,0])
        out = AG.config.system.calculate_magnification()
        self.assertEqual(out, [500, 1000, 3]) 

        AG = AcquisitionGeometry.create_Cone3D(source_position=[0,-500,0], detector_position=[0.,1000.,0], rotation_axis_position=[0.,250.,0])
        out = AG.config.system.calculate_magnification()
        self.assertEqual(out, [750, 750, 2]) 

        AG = AcquisitionGeometry.create_Cone3D(source_position=[0,-500,0], detector_position=[0.,1000.,0], rotation_axis_position=[5.,0.,0])
        out = AG.config.system.calculate_magnification()
        source_to_object = numpy.sqrt(5.0**2 + 500.0**2)
        theta = math.atan2(5.0,500.0)
        source_to_detector = 1500.0/math.cos(theta)
        self.assertEqual(out, [source_to_object, source_to_detector - source_to_object, source_to_detector/source_to_object]) 

        AG = AcquisitionGeometry.create_Cone3D(source_position=[0,-500,0], detector_position=[0.,1000.,0], rotation_axis_position=[0.,0.,5.])
        out = AG.config.system.calculate_magnification()
        source_to_object = numpy.sqrt(5.0**2 + 500.0**2)
        theta = math.atan2(5.0,500.0)
        source_to_detector = 1500.0/math.cos(theta)
        self.assertEqual(out, [source_to_object, source_to_detector - source_to_object, source_to_detector/source_to_object]) 

        AG = AcquisitionGeometry.create_Cone3D(source_position=[0,-500,0], detector_position=[0.,1000.,0],detector_direction_y=[0,math.sqrt(5),math.sqrt(5)])
        out = AG.config.system.calculate_magnification()
        self.assertEqual(out, [500, 1000, 3]) 

        AG = AcquisitionGeometry.create_Cone3D(source_position=[0,-500,0], detector_position=[0.,1000.,0],detector_direction_x=[1,0.1,0.2],detector_direction_y=[-0.2,0,1])
        out = AG.config.system.calculate_magnification()
        self.assertEqual(out, [500, 1000, 3])         
        
        AG = AcquisitionGeometry.create_Cone3D(source_position=[0,-500,0], detector_position=[0.,1000.,0], rotation_axis_position=[5.,0.,0],detector_direction_x=[math.sqrt(5),math.sqrt(5),0])
        out = AG.config.system.calculate_magnification()
        source_to_object = numpy.sqrt(5.0**2 + 500.0**2)

        ab = (AG.config.system.rotation_axis.position - AG.config.system.source.position).astype(numpy.float64)/source_to_object

        #source_position + d * ab = detector_position + t * detector_direction_x
        #x: d *  ab[0] =  t * detector_direction_x[0]
        #y: -500 + d *  ab[1] = 1000 + t * detector_direction_x[1] 

        # t = (d *  ab[0]) / math.sqrt(5)
        # d = 1500 / (ab[1]  - ab[0])

        source_to_detector = 1500 / (ab[1]  - ab[0])
        self.assertEqual(out, [source_to_object, source_to_detector - source_to_object, source_to_detector/source_to_object]) 


