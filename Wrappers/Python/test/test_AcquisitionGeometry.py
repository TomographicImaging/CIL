# -*- coding: utf-8 -*-
#  Copyright 2020 United Kingdom Research and Innovation
#  Copyright 2020 The University of Manchester
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
import numpy as np
import math
import re
import io
import sys
from cil.framework import AcquisitionGeometry, ImageGeometry, BlockGeometry, AcquisitionData
from cil.framework.framework import SystemConfiguration
from cil.framework import Partitioner

initialise_tests()

class Test_AcquisitionGeometry(unittest.TestCase):
    def test_create_Parallel2D(self):

        #default
        AG = AcquisitionGeometry.create_Parallel2D()
        np.testing.assert_allclose(AG.config.system.ray.direction, [0,1], rtol=1E-6)
        np.testing.assert_allclose(AG.config.system.detector.position, [0,0], rtol=1E-6)
        np.testing.assert_allclose(AG.config.system.detector.direction_x, [1,0], rtol=1E-6)
        np.testing.assert_allclose(AG.config.system.rotation_axis.position, [0,0], rtol=1E-6)

        #values
        ray_direction = [0.1, 3.0]
        detector_position = [-1.3,1000.0]
        detector_direction_x = [1,0.2]
        rotation_axis_position = [0.1,2]

        AG = AcquisitionGeometry.create_Parallel2D(ray_direction, detector_position, detector_direction_x, rotation_axis_position)

        ray_direction = np.asarray(ray_direction)
        detector_direction_x = np.asarray(detector_direction_x)

        ray_direction /= np.sqrt((ray_direction**2).sum())
        detector_direction_x /= np.sqrt((detector_direction_x**2).sum())

        np.testing.assert_allclose(AG.config.system.ray.direction, ray_direction, rtol=1E-6)
        np.testing.assert_allclose(AG.config.system.detector.position, detector_position, rtol=1E-6)
        np.testing.assert_allclose(AG.config.system.detector.direction_x, detector_direction_x, rtol=1E-6)
        np.testing.assert_allclose(AG.config.system.rotation_axis.position, rotation_axis_position, rtol=1E-6)
    
    def test_create_Parallel3D(self):

        #default
        AG = AcquisitionGeometry.create_Parallel3D()
        np.testing.assert_allclose(AG.config.system.ray.direction, [0,1,0], rtol=1E-6)
        np.testing.assert_allclose(AG.config.system.detector.position, [0,0,0], rtol=1E-6)
        np.testing.assert_allclose(AG.config.system.detector.direction_x, [1,0,0], rtol=1E-6)
        np.testing.assert_allclose(AG.config.system.detector.direction_y, [0,0,1], rtol=1E-6)
        np.testing.assert_allclose(AG.config.system.rotation_axis.position, [0,0,0], rtol=1E-6)
        np.testing.assert_allclose(AG.config.system.rotation_axis.direction, [0,0,1], rtol=1E-6)

        #values
        ray_direction = [0.1, 3.0, 0.2]
        detector_position = [-1.3,1000.0, -1.0]
        detector_direction_x = [1,0.2, 0]
        detector_direction_y = [0.0,0,1]
        rotation_axis_position=[0.1, 2,-0.4]
        rotation_axis_direction=[-0.1,-0.3,1]

        AG = AcquisitionGeometry.create_Parallel3D(ray_direction, detector_position, detector_direction_x,detector_direction_y, rotation_axis_position,rotation_axis_direction)

        ray_direction = np.asarray(ray_direction)
        detector_direction_x = np.asarray(detector_direction_x)
        detector_direction_y = np.asarray(detector_direction_y)
        rotation_axis_direction = np.asarray(rotation_axis_direction)

        ray_direction /= np.sqrt((ray_direction**2).sum())
        detector_direction_x /= np.sqrt((detector_direction_x**2).sum())
        detector_direction_y /= np.sqrt((detector_direction_y**2).sum())
        rotation_axis_direction /= np.sqrt((rotation_axis_direction**2).sum())

        np.testing.assert_allclose(AG.config.system.ray.direction, ray_direction, rtol=1E-6)
        np.testing.assert_allclose(AG.config.system.detector.position, detector_position, rtol=1E-6)
        np.testing.assert_allclose(AG.config.system.detector.direction_x, detector_direction_x, rtol=1E-6)
        np.testing.assert_allclose(AG.config.system.detector.direction_y, detector_direction_y, rtol=1E-6)
        np.testing.assert_allclose(AG.config.system.rotation_axis.position, rotation_axis_position, rtol=1E-6)
        np.testing.assert_allclose(AG.config.system.rotation_axis.direction, rotation_axis_direction, rtol=1E-6)


    def test_create_Cone2D(self):
        #default
        source_position = [0.1, -500.0]
        detector_position = [-1.3,1000.0]

        AG = AcquisitionGeometry.create_Cone2D(source_position, detector_position)
        np.testing.assert_allclose(AG.config.system.source.position, source_position, rtol=1E-6)
        np.testing.assert_allclose(AG.config.system.detector.position, detector_position, rtol=1E-6)
        np.testing.assert_allclose(AG.config.system.detector.direction_x, [1,0], rtol=1E-6)
        np.testing.assert_allclose(AG.config.system.rotation_axis.position, [0,0], rtol=1E-6)

        #values
        detector_direction_x = [1,0.2]
        rotation_axis_position = [0.1,2]

        AG = AcquisitionGeometry.create_Cone2D(source_position, detector_position, detector_direction_x, rotation_axis_position)

        detector_direction_x = np.asarray(detector_direction_x)
        detector_direction_x /= np.sqrt((detector_direction_x**2).sum())

        np.testing.assert_allclose(AG.config.system.source.position, source_position, rtol=1E-6)
        np.testing.assert_allclose(AG.config.system.detector.position, detector_position, rtol=1E-6)
        np.testing.assert_allclose(AG.config.system.detector.direction_x, detector_direction_x, rtol=1E-6)
        np.testing.assert_allclose(AG.config.system.rotation_axis.position, rotation_axis_position, rtol=1E-6)
    
    def test_create_Cone3D(self):

        #default
        source_position = [0.1, -500.0,-2.0]
        detector_position = [-1.3,1000.0, -1.0]

        AG = AcquisitionGeometry.create_Cone3D(source_position, detector_position)
        np.testing.assert_allclose(AG.config.system.source.position, source_position, rtol=1E-6)
        np.testing.assert_allclose(AG.config.system.detector.position, detector_position, rtol=1E-6)
        np.testing.assert_allclose(AG.config.system.detector.direction_x, [1,0,0], rtol=1E-6)
        np.testing.assert_allclose(AG.config.system.detector.direction_y, [0,0,1], rtol=1E-6)
        np.testing.assert_allclose(AG.config.system.rotation_axis.position, [0,0,0], rtol=1E-6)
        np.testing.assert_allclose(AG.config.system.rotation_axis.direction, [0,0,1], rtol=1E-6)

        #values
        detector_direction_x = [1,0.2, 0]
        detector_direction_y = [0.0,0,1]
        rotation_axis_position=[0.1, 2,-0.4]
        rotation_axis_direction=[-0.1,-0.3,1]

        AG = AcquisitionGeometry.create_Cone3D(source_position, detector_position, detector_direction_x,detector_direction_y, rotation_axis_position,rotation_axis_direction)

        detector_direction_x = np.asarray(detector_direction_x)
        detector_direction_y = np.asarray(detector_direction_y)
        rotation_axis_direction = np.asarray(rotation_axis_direction)

        detector_direction_x /= np.sqrt((detector_direction_x**2).sum())
        detector_direction_y /= np.sqrt((detector_direction_y**2).sum())
        rotation_axis_direction /= np.sqrt((rotation_axis_direction**2).sum())

        np.testing.assert_allclose(AG.config.system.source.position, source_position, rtol=1E-6)
        np.testing.assert_allclose(AG.config.system.detector.position, detector_position, rtol=1E-6)
        np.testing.assert_allclose(AG.config.system.detector.direction_x, detector_direction_x, rtol=1E-6)
        np.testing.assert_allclose(AG.config.system.detector.direction_y, detector_direction_y, rtol=1E-6)
        np.testing.assert_allclose(AG.config.system.rotation_axis.position, rotation_axis_position, rtol=1E-6)
        np.testing.assert_allclose(AG.config.system.rotation_axis.direction, rotation_axis_direction, rtol=1E-6)


    def test_shift_detector_origin_bottom_left(self):
        initial_position = np.array([2.5, -1.3, 10.2])
        pixel_size = np.array([0.5, 0.7])
        detector_direction_x=np.array([1/math.sqrt(2),0,1/math.sqrt(2)])
        detector_direction_y=np.array([-1/math.sqrt(2),0,1/math.sqrt(2)])

        geometry = AcquisitionGeometry.create_Parallel3D(detector_position=initial_position, detector_direction_x=detector_direction_x, detector_direction_y=detector_direction_y)\
                                      .set_panel([10, 5], [0.5, 0.7], origin='bottom-left')\
                                      .set_angles([0])
        # Test horizontal shift to the left
        shift = -1.5
        geometry.config.shift_detector_in_plane(shift, 'horizontal')
        updated_position = geometry.config.system.detector.position
        expected_position = initial_position - detector_direction_x * pixel_size[0] * shift
        np.testing.assert_array_almost_equal(updated_position, expected_position)

        # Test horizontal shift to the right
        shift = 3.0
        geometry.config.shift_detector_in_plane(shift, 'horizontal')
        updated_position = geometry.config.system.detector.position
        expected_position = expected_position - detector_direction_x * pixel_size[0] * shift
        np.testing.assert_array_almost_equal(updated_position, expected_position)

        # Test vertical shift down
        shift = -1.5
        geometry.config.shift_detector_in_plane(shift, 'vertical')
        updated_position = geometry.config.system.detector.position
        expected_position = expected_position - detector_direction_y * pixel_size[1] * shift
        np.testing.assert_array_almost_equal(updated_position, expected_position)

        # Test vertical shift up
        shift = 3.0
        geometry.config.shift_detector_in_plane(shift, 'vertical')
        updated_position = geometry.config.system.detector.position
        expected_position = expected_position - detector_direction_y * pixel_size[1] * shift
        np.testing.assert_array_almost_equal(updated_position, expected_position)


    def test_shift_detector_origin_top_right(self):
        initial_position = np.array([2.5, -1.3, 10.2])
        detector_direction_x=np.array([1/math.sqrt(2),0,1/math.sqrt(2)])
        detector_direction_y=np.array([-1/math.sqrt(2),0,1/math.sqrt(2)])

        pixel_size = np.array([0.5, 0.7])
        geometry = AcquisitionGeometry.create_Parallel3D(detector_position=initial_position, detector_direction_x=detector_direction_x, detector_direction_y=detector_direction_y)\
                                      .set_panel([10, 5], [0.5, 0.7], origin='top-right')\
                                      .set_angles([0])

        # Test horizontal shift to the right
        shift = -1.5
        geometry.config.shift_detector_in_plane(shift, 'horizontal')
        updated_position = geometry.config.system.detector.position
        expected_position = initial_position + detector_direction_x * pixel_size[0] * shift
        np.testing.assert_array_almost_equal(updated_position, expected_position)

        # Test horizontal shift to the left
        shift = 3.0
        geometry.config.shift_detector_in_plane(shift, 'horizontal')
        updated_position = geometry.config.system.detector.position
        expected_position = expected_position + detector_direction_x * pixel_size[0] * shift
        np.testing.assert_array_almost_equal(updated_position, expected_position)

        # Test vertical shift up
        shift = -1.5
        geometry.config.shift_detector_in_plane(shift, 'vertical')
        updated_position = geometry.config.system.detector.position
        expected_position = expected_position + detector_direction_y * pixel_size[1] * shift
        np.testing.assert_array_almost_equal(updated_position, expected_position)

        # Test vertical shift down
        shift = 3.0
        geometry.config.shift_detector_in_plane(shift, 'vertical')
        updated_position = geometry.config.system.detector.position
        expected_position = expected_position +  detector_direction_y * pixel_size[1] * shift
        np.testing.assert_array_almost_equal(updated_position, expected_position)


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
        angles = np.linspace(0, 360, 10, dtype=np.float32)

        #default
        AG.set_angles(angles)
        np.testing.assert_allclose(AG.config.angles.angle_data, angles, rtol=1E-6)
        self.assertEqual(AG.config.angles.initial_angle, 0.0)
        self.assertEqual(AG.config.angles.angle_unit, 'degree')

        #values        
        AG.set_angles(angles, 0.1, 'radian')
        np.testing.assert_allclose(AG.config.angles.angle_data, angles, rtol=1E-6)
        self.assertEqual(AG.config.angles.initial_angle, 0.1)
        self.assertEqual(AG.config.angles.angle_unit, 'radian')

    def test_set_panel(self):
        AG = AcquisitionGeometry.create_Parallel3D()

        #default
        AG.set_panel([1000,2000])
        np.testing.assert_array_equal(AG.config.panel.num_pixels, [1000,2000])
        np.testing.assert_array_almost_equal(AG.config.panel.pixel_size, [1,1])

        #values
        AG.set_panel([1000,2000],[0.1,0.2])
        np.testing.assert_array_equal(AG.config.panel.num_pixels, [1000,2000])
        np.testing.assert_array_almost_equal(AG.config.panel.pixel_size, [0.1,0.2])

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

    def test_get_centre_of_rotation(self):

        # Functionality is tested in specific implementations
        # this checks the pixel size scaling and return format for each geometry type
        
        gold1_2D = {'offset':(0.25,'units distance'), 'angle':(0.0,'radian')}
        gold2_2D = {'offset':(0.5,'pixels'), 'angle':(0.0,'degree')}
        gold1_3D = {'offset':(0.25,'units distance'), 'angle':(math.pi/4,'radian')}
        gold2_3D = {'offset':(0.5,'pixels'), 'angle':(45,'degree')}

        #check outputs for each geometry type
        ag = AcquisitionGeometry.create_Parallel2D(rotation_axis_position=[0.25, 0.0]).set_panel(10,0.5)
        out1 = ag.get_centre_of_rotation()
        out2 = ag.get_centre_of_rotation(distance_units='pixels', angle_units='degree')
        self.assertDictEqual(gold1_2D, out1, "Failed Parallel2D")
        self.assertDictEqual(gold2_2D, out2, "Failed Parallel2D")

        ag = AcquisitionGeometry.create_Parallel3D(rotation_axis_position=[0.25, 0.0, 0.0], rotation_axis_direction=[0.5,0.0,0.5]).set_panel([10,10],[0.5,0.5])
        out1 = ag.get_centre_of_rotation()
        out2 = ag.get_centre_of_rotation(distance_units='pixels', angle_units='degree')
        self.assertDictEqual(gold1_3D, out1, "Failed Parallel3D")
        self.assertDictEqual(gold2_3D, out2, "Failed Parallel3D")

        ag = AcquisitionGeometry.create_Cone2D([0,-50], [0,50],rotation_axis_position=[0.125, 0.0]).set_panel(10,0.5)
        out1 = ag.get_centre_of_rotation()
        out2 = ag.get_centre_of_rotation(distance_units='pixels', angle_units='degree')
        self.assertDictEqual(gold1_2D, out1, "Failed Cone2D")
        self.assertDictEqual(gold2_2D, out2, "Failed Cone2D")

        ag = AcquisitionGeometry.create_Cone3D([0,-50,0], [0,50,0], rotation_axis_position=[0.125, 0.0, 0.0], rotation_axis_direction=[0.5,0.0,0.5]).set_panel([10,10],[0.5,0.5])
        out1 = ag.get_centre_of_rotation()
        out2 = ag.get_centre_of_rotation(distance_units='pixels', angle_units='degree')
        self.assertDictEqual(gold1_3D, out1, "Failed Cone3D")
        self.assertDictEqual(gold2_3D, out2, "Failed Cone3D")

        with self.assertRaises(ValueError):
            ag.get_centre_of_rotation(distance_units='bad input')

        with self.assertRaises(ValueError):
            ag.get_centre_of_rotation(angle_units='bad input')


    def test_set_centre_of_rotation(self):
        # Functionality is tested in specific implementations
        # this checks the pixel size scaling and return format for each geometry type
        
        gold_2D = {'offset':(0.25,'units distance'), 'angle':(0.0,'radian')}
        gold_3D = {'offset':(0.25,'units distance'), 'angle':(math.pi/4,'radian')}

        #check outputs for each geometry type
        ag = AcquisitionGeometry.create_Parallel2D().set_panel(10,0.5)
        ag.set_centre_of_rotation(0.25)
        out = ag.get_centre_of_rotation()
        self.assertDictEqual(gold_2D, out, "Failed Parallel2D default")

        ag.set_centre_of_rotation(0.5, 'pixels')
        out = ag.get_centre_of_rotation()
        self.assertDictEqual(gold_2D, out, "Failed Parallel2D unit")

        ag = AcquisitionGeometry.create_Parallel3D().set_panel([10,10],[0.5,0.5])
        ag.set_centre_of_rotation(0.25, angle=math.pi/4)
        out = ag.get_centre_of_rotation()
        self.assertDictEqual(gold_3D, out, "Failed Parallel3D default")

        ag.set_centre_of_rotation(0.5, 'pixels', 45, 'degree')
        out = ag.get_centre_of_rotation()        
        self.assertDictEqual(gold_3D, out, "Failed Parallel3D units")

        ag = AcquisitionGeometry.create_Cone2D([0,-50], [0,50]).set_panel(10,0.5)
        ag.set_centre_of_rotation(0.25)
        out = ag.get_centre_of_rotation()
        self.assertDictEqual(gold_2D, out, "Failed Cone2D default")

        ag.set_centre_of_rotation(0.5, 'pixels')
        out = ag.get_centre_of_rotation()       
        self.assertDictEqual(gold_2D, out, "Failed Cone2D units")

        ag = AcquisitionGeometry.create_Cone3D([0,-50,0], [0,50,0]).set_panel([10,10],[0.5,0.5])
        ag.set_centre_of_rotation(0.25, angle=math.pi/4)
        out = ag.get_centre_of_rotation()
        self.assertDictEqual(gold_3D, out, "Failed Cone3D default")

        ag.set_centre_of_rotation(0.5,'pixels', 45, 'degree')
        out = ag.get_centre_of_rotation()        
        self.assertDictEqual(gold_3D, out, "Failed Cone3D units")

        with self.assertRaises(ValueError):
            ag.set_centre_of_rotation(distance_units='bad input')

        with self.assertRaises(ValueError):
            ag.set_centre_of_rotation(angle_units='bad input')


    def test_set_centre_of_rotation_by_slice(self):
        # Functionality is tested in specific implementations
        # this checks the pixel size scaling and return format for each geometry type
        
        gold_2D = {'offset':(0.25,'units distance'), 'angle':(0.0,'radian')}
        gold_3D = {'offset':(0.25,'units distance'), 'angle':(math.pi/4,'radian')}

        #check outputs for each geometry type
        ag = AcquisitionGeometry.create_Parallel2D().set_panel(10,0.5)
        ag.set_centre_of_rotation_by_slice(0.5)
        out = ag.get_centre_of_rotation()
        self.assertDictEqual(gold_2D, out, "Failed Parallel2D")

        ag = AcquisitionGeometry.create_Parallel3D().set_panel([10,10],[0.5,0.5])
        ag.set_centre_of_rotation_by_slice(-4.5, -5, 5.5, 5)
        out = ag.get_centre_of_rotation()        
        self.assertDictEqual(gold_3D, out, "Failed Parallel3D")

        ag = AcquisitionGeometry.create_Cone2D([0,-50], [0,50]).set_panel(10,0.5)
        ag.set_centre_of_rotation_by_slice(0.5)
        out = ag.get_centre_of_rotation()       
        self.assertDictEqual(gold_2D, out, "Failed Cone2D")

        ag = AcquisitionGeometry.create_Cone3D([0,-50,0], [0,50,0]).set_panel([10,10],[0.5,0.5])
        ag.set_centre_of_rotation_by_slice(-4.5, -5, 5.5, 5)
        out = ag.get_centre_of_rotation()        
        self.assertDictEqual(gold_3D, out, "Failed Cone3D")


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
        test2 = np.ndarray([2,5,3,4])
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


class AlignGeometries(unittest.TestCase):

    def test_set_origin(self):

        AG = AcquisitionGeometry.create_Cone3D(source_position=[0,-50,0], detector_position=[0.,100.,0])\
            .set_panel(num_pixels=[50,50])    

        self.assertTrue(True)


    def test_rotation_vec_to_y(self):

        M = SystemConfiguration.rotation_vec_to_y([0,1])
        a = np.array([[1, 0],[0, 1]])
        np.testing.assert_allclose(M,a, atol=1e-6)
        
        M = SystemConfiguration.rotation_vec_to_y([0,-1])
        a = np.array([[-1, 0],[0, -1]])
        np.testing.assert_allclose(M,a, atol=1e-6)

        M = SystemConfiguration.rotation_vec_to_y([1,1])
        a = np.array([[0.70710678, -0.70710678],[0.70710678, 0.70710678]])
        np.testing.assert_allclose(M,a, atol=1e-6)

        M = SystemConfiguration.rotation_vec_to_y([1,-1])
        a = np.array([[-0.70710678, -0.70710678],[0.70710678, -0.70710678]])
        np.testing.assert_allclose(M,a, atol=1e-6)

        M = SystemConfiguration.rotation_vec_to_y([0,1,0])
        a = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])        
        np.testing.assert_allclose(M,a, atol=1e-6)

        M = SystemConfiguration.rotation_vec_to_y([0,-1,0])
        a = np.array([[-1, 0, 0],[0, -1, 0],[0, 0, 1]])             
        np.testing.assert_allclose(M,a, atol=1e-6)

        M = SystemConfiguration.rotation_vec_to_y([0,1,1])
        a = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])     
        np.testing.assert_allclose(M,a, atol=1e-6)

        M = SystemConfiguration.rotation_vec_to_y([0,-1,1])
        a = np.array([[-1, 0, 0],[0, -1, 0],[0, 0, 1]])             
        np.testing.assert_allclose(M,a, atol=1e-6)

        M = SystemConfiguration.rotation_vec_to_y([1,1,0])
        a = np.array([[0.70710678, -0.70710678, 0],[0.70710678, 0.70710678, 0],[0, 0, 1]])       
        np.testing.assert_allclose(M,a, atol=1e-6)

        M = SystemConfiguration.rotation_vec_to_y([1,-1,0])
        a = np.array([[-0.70710678, -0.70710678, 0],[0.70710678, -0.70710678, 0],[0, 0, 1]])               
        np.testing.assert_allclose(M,a, atol=1e-6)

        M = SystemConfiguration.rotation_vec_to_y([1,-1,1])
        a = np.array([[-0.70710678, -0.70710678, 0],[0.70710678, -0.70710678, 0],[0, 0, 1]])       
        np.testing.assert_allclose(M,a, atol=1e-6)


    def test_rotation_vec_z(self):

        M = SystemConfiguration.rotation_vec_to_z([0,0,1])
        a = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])   
        np.testing.assert_allclose(M,a, atol=1e-6)

        M = SystemConfiguration.rotation_vec_to_z([0,0,-1])
        a = np.array([[1, 0, 0],[0, -1, 0],[0, 0, -1]])           
        np.testing.assert_allclose(M,a, atol=1e-6)

        M = SystemConfiguration.rotation_vec_to_z([1,0,0])
        a = np.array([[0, 0, -1],[0, 1, 0],[1, 0, 0]])           
        np.testing.assert_allclose(M,a, atol=1e-6)

        M = SystemConfiguration.rotation_vec_to_z([-1,0,0])
        a = np.array([[0, 0, 1],[0, 1, 0],[-1, 0, 0]])           
        np.testing.assert_allclose(M,a, atol=1e-6)

        M = SystemConfiguration.rotation_vec_to_z([0,1,0])
        a = np.array([[1, 0, 0],[0, 0, -1],[0, 1, 0]])           
        np.testing.assert_allclose(M,a, atol=1e-6)

        M = SystemConfiguration.rotation_vec_to_z([0,-1,0])
        a = np.array([[1, 0, 0],[0, 0, 1],[0, -1, 0]])  
        np.testing.assert_allclose(M,a, atol=1e-6)

        M = SystemConfiguration.rotation_vec_to_z([1,-1,0])
        a = np.array([[0.5, 0.5, -0.70710678],[0.5, 0.5, 0.70710678],[0.70710678, -0.70710678, 0]])          
        np.testing.assert_allclose(M,a, atol=1e-6)

        M = SystemConfiguration.rotation_vec_to_z([1,0,1])
        a = np.array([[0.70710678, 0, -0.70710678],[0,1,0],[0.70710678, 0, 0.70710678]])          
        np.testing.assert_allclose(M,a, atol=1e-6)

        M = SystemConfiguration.rotation_vec_to_z([0,1,-1])
        a = np.array([[1,0,0],[0, -0.70710678, -0.70710678],[0, 0.70710678, -0.70710678]])          

        np.testing.assert_allclose(M,a, atol=1e-6)

        M = SystemConfiguration.rotation_vec_to_z([-1,-1,-1])
        a = np.array([[0.21132491, -0.78867509, 0.57735025],[ -0.78867509, 0.21132491, 0.57735025],[-0.57735025, -0.57735025, -0.57735025]])          

        np.testing.assert_allclose(M,a, atol=1e-6)


class Test_Parallel2D(unittest.TestCase):


    def test_align_reference_frame_cil(self):
    
        ag = AcquisitionGeometry.create_Parallel2D(ray_direction=[0,-1], detector_position=[0.,-100.], rotation_axis_position=[10.,5.])
        ag.config.system.align_reference_frame('cil')

        np.testing.assert_allclose(ag.config.system.ray.direction, [0,1], rtol=1E-6)
        np.testing.assert_allclose(ag.config.system.detector.position, [10,105], rtol=1E-6)
        np.testing.assert_allclose(ag.config.system.detector.direction_x, [-1,0], rtol=1E-6)
        np.testing.assert_allclose(ag.config.system.rotation_axis.position, [0,0], rtol=1E-6)


    def test_align_reference_frame_tigre(self):

        ag = AcquisitionGeometry.create_Parallel2D(ray_direction=[0,-1], detector_position=[0.,-100.], rotation_axis_position=[10.,5.])
        ag.config.system.align_reference_frame('tigre')

        np.testing.assert_allclose(ag.config.system.ray.direction, [0,1], rtol=1E-6)
        np.testing.assert_allclose(ag.config.system.detector.position, [10,105], rtol=1E-6)
        np.testing.assert_allclose(ag.config.system.detector.direction_x, [-1,0], rtol=1E-6)
        np.testing.assert_allclose(ag.config.system.rotation_axis.position, [0,0], rtol=1E-6)


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

    def test_calculate_centre_of_rotation(self):
        AG = AcquisitionGeometry.create_Parallel2D()
        out = AG.config.system.calculate_centre_of_rotation()
        gold = {'offset':(0,'units')}
        gold = (0,0)
        np.testing.assert_allclose(out, gold, err_msg="Failed basic")


        AG = AcquisitionGeometry.create_Parallel2D(rotation_axis_position=[0.5,0.])
        out = AG.config.system.calculate_centre_of_rotation()
        gold = (0.5,0)
        np.testing.assert_allclose(out, gold, err_msg="Failed positive offset")

        AG = AcquisitionGeometry.create_Parallel2D(rotation_axis_position=[-0.5,0.])
        out = AG.config.system.calculate_centre_of_rotation()
        gold = (-0.5,0)
        np.testing.assert_allclose(out, gold, err_msg="Failed negative offset")

        AG = AcquisitionGeometry.create_Parallel2D(rotation_axis_position=[0.5,0.], detector_direction_x=[-1,0])
        out = AG.config.system.calculate_centre_of_rotation()
        gold = (-0.5,0)
        np.testing.assert_allclose(out, gold, err_msg="Failed detector direction")        

        theta = math.pi/4 #detector angle
        distance = 0.5 / math.cos(theta)
        AG = AcquisitionGeometry.create_Parallel2D(detector_direction_x=[0.5,0.5],rotation_axis_position=[0.5,0.])
        out = AG.config.system.calculate_centre_of_rotation()
        gold = (distance,0)
        np.testing.assert_allclose(out, gold, err_msg="Failed with rotated detector")

    def test_set_centre_of_rotation(self):
        AG = AcquisitionGeometry.create_Parallel2D()

        gold = (1.5, 0)
        AG.config.system.set_centre_of_rotation(gold[0])
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed positive offset")

        gold = (-1.5, 0)
        AG.config.system.set_centre_of_rotation(gold[0])
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed negative offset")

        gold = (0, 0)
        AG.config.system.set_centre_of_rotation(gold[0])
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed reset offset",atol=1e-10)

        AG = AcquisitionGeometry.create_Parallel2D(detector_direction_x=[-1,0])

        offset_in = 1.5
        gold = (1.5, 0)
        AG.config.system.set_centre_of_rotation(gold[0])
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed positive offset")

        gold = (-1.5, 0)
        AG.config.system.set_centre_of_rotation(gold[0])
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed negative offset")

        gold = (0, 0)
        AG.config.system.set_centre_of_rotation(gold[0])
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg= "Failed reset offset",atol=1e-10)


        AG = AcquisitionGeometry.create_Parallel2D(detector_direction_x=[0.5,0.5],rotation_axis_position=[0.5,0.])
        gold = (1.5, 0)
        AG.config.system.set_centre_of_rotation(gold[0])
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed positive offset")

        gold = (-1.5, 0)
        AG.config.system.set_centre_of_rotation(gold[0])
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed negative offset")

        offset_in = 0
        gold = (0, 0)
        AG.config.system.set_centre_of_rotation(gold[0])
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed reset offset",atol=1e-10)


class Test_Parallel3D(unittest.TestCase):

    def test_align_reference_frame_cil(self):
        
        ag = AcquisitionGeometry.create_Parallel3D(ray_direction=[0,-1,0], detector_position=[0.,-100.,0], rotation_axis_position=[10.,5.,0], rotation_axis_direction=[0,0,-1])
        ag.config.system.align_reference_frame('cil')

        np.testing.assert_allclose(ag.config.system.ray.direction, [0,1, 0], rtol=1E-6)
        np.testing.assert_allclose(ag.config.system.detector.position, [-10,105,0], rtol=1E-6)
        np.testing.assert_allclose(ag.config.system.detector.direction_x, [1,0,0], rtol=1E-6)
        np.testing.assert_allclose(ag.config.system.detector.direction_y, [0,0,-1], rtol=1E-6)
        np.testing.assert_allclose(ag.config.system.rotation_axis.position, [0,0,0], rtol=1E-6)
        np.testing.assert_allclose(ag.config.system.rotation_axis.direction, [0,0,1], rtol=1E-6)


    def test_align_reference_frame_tigre(self):

        ag = AcquisitionGeometry.create_Parallel3D(ray_direction=[0,-1,0], detector_position=[0.,-100.,0], rotation_axis_position=[10.,5.,0], rotation_axis_direction=[0,0,-1])
        ag.config.system.align_reference_frame('tigre')

        np.testing.assert_allclose(ag.config.system.ray.direction, [0,1, 0], rtol=1E-6)
        np.testing.assert_allclose(ag.config.system.detector.position, [-10,105,0], rtol=1E-6)
        np.testing.assert_allclose(ag.config.system.detector.direction_x, [1,0,0], rtol=1E-6)
        np.testing.assert_allclose(ag.config.system.detector.direction_y, [0,0,-1], rtol=1E-6)
        np.testing.assert_allclose(ag.config.system.rotation_axis.position, [0,0,0], rtol=1E-6)
        np.testing.assert_allclose(ag.config.system.rotation_axis.direction, [0,0,1], rtol=1E-6)


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

    def test_calculate_centre_of_rotation(self):

        AG = AcquisitionGeometry.create_Parallel3D()
        out = AG.config.system.calculate_centre_of_rotation()
        gold = (0,0)
        np.testing.assert_allclose(out, gold, err_msg="Failed basic")

        angle = math.pi/4
        AG = AcquisitionGeometry.create_Parallel3D(rotation_axis_position=[0.5,0.,0.], rotation_axis_direction=[0.5,0,0.5])
        out = AG.config.system.calculate_centre_of_rotation()
        gold = (0.5,angle)
        np.testing.assert_allclose(out, gold, err_msg="Failed positive offset")

        AG = AcquisitionGeometry.create_Parallel3D(rotation_axis_position=[-0.5,0.,0.], rotation_axis_direction=[-0.5,0,0.5])
        out = AG.config.system.calculate_centre_of_rotation()
        gold = (-0.5,-angle)
        np.testing.assert_allclose(out, gold, err_msg="Failed negative offset")

        AG = AcquisitionGeometry.create_Parallel3D(rotation_axis_position=[0.5,0.,0.], rotation_axis_direction=[0.5,0,0.5], detector_direction_x=[-1,0,0.])
        out = AG.config.system.calculate_centre_of_rotation()
        gold = (-0.5,-angle)
        np.testing.assert_allclose(out, gold, err_msg="Failed detector direction_x")        

        AG = AcquisitionGeometry.create_Parallel3D(rotation_axis_position=[0.5,0.,0.], rotation_axis_direction=[0.5,0,0.5], detector_direction_y=[0,0,-1])
        out = AG.config.system.calculate_centre_of_rotation()
        gold = (0.5,math.pi-angle)
        np.testing.assert_allclose(out, gold, err_msg="Failed detector direction_y")        

        AG = AcquisitionGeometry.create_Parallel3D(rotation_axis_position=[0.5,0.,0.], rotation_axis_direction=[-0.5,0,-0.5], detector_direction_y=[0,0,-1])
        out = AG.config.system.calculate_centre_of_rotation()
        gold = (0.5,-angle)
        np.testing.assert_allclose(out, gold, err_msg="Failed invert rotate axis")        


        theta = math.pi/4 #detector angle
        distance = 0.5 / math.cos(theta)
        AG = AcquisitionGeometry.create_Parallel3D(detector_direction_x=[0.5,0.5,0.],rotation_axis_position=[0.5,0.,0.])
        out = AG.config.system.calculate_centre_of_rotation()
        gold = {'offset':(distance,'units')}
        gold = (distance,0)
        np.testing.assert_allclose(out, gold, err_msg="Failed rotated detector")

        AG = AcquisitionGeometry.create_Parallel3D(detector_direction_y=[0.0,-0.5,0.5],rotation_axis_position=[0.5,0.,0.])
        out = AG.config.system.calculate_centre_of_rotation()
        gold = (0.5,0)
        np.testing.assert_allclose(out, gold, err_msg="Failed tilted detector")


    def test_set_centre_of_rotation(self):
        AG = AcquisitionGeometry.create_Parallel3D()

        gold = (1.5, 0)
        AG.config.system.set_centre_of_rotation(*gold)
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed positive offset")


        gold = (-1.5, 0)
        AG.config.system.set_centre_of_rotation(*gold)
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed negative offset")

        gold = (0, 0)
        AG.config.system.set_centre_of_rotation(*gold)
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed reset offset")

        gold = (0.0, 0.2)
        AG.config.system.set_centre_of_rotation(*gold)
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed positive angle")

        gold = (0.0, -0.2)
        AG.config.system.set_centre_of_rotation(*gold)
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed negative angle")

        gold = (0, 0)
        AG.config.system.set_centre_of_rotation(*gold)
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed reset angle")

        gold = (-1.5, -0.2)
        AG.config.system.set_centre_of_rotation(*gold)
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed combination A")

        gold = (1.5, 0.2)
        AG.config.system.set_centre_of_rotation(*gold)
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed combination B")



        AG = AcquisitionGeometry.create_Parallel3D(detector_direction_x=[-1,0,0.])

        gold = (-1.5, -0.2)
        AG.config.system.set_centre_of_rotation(*gold)
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed invert detector x A")

        gold = (1.5, 0.2)
        AG.config.system.set_centre_of_rotation(*gold)
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed invert detector x B")


        AG = AcquisitionGeometry.create_Parallel3D(rotation_axis_direction=[0,0,-1])
        
        gold = (-1.5, -0.2)
        AG.config.system.set_centre_of_rotation(*gold)
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed invert rotate axis A")

        gold = (1.5, 0.2)
        AG.config.system.set_centre_of_rotation(*gold)
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed invert rotate axis B")


        AG = AcquisitionGeometry.create_Parallel3D(detector_direction_y=[0,0,-1])
        gold = (-1.5, -0.2)
        AG.config.system.set_centre_of_rotation(*gold)
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed invert detector y A")

        gold = (1.5, 0.2)
        AG.config.system.set_centre_of_rotation(*gold)
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed invert detector y A")

        AG = AcquisitionGeometry.create_Parallel3D(rotation_axis_position=[0.5,0.,0.], rotation_axis_direction=[-0.5,0,-0.5], detector_direction_y=[0,0,-1])
        gold = (-1.5, -0.2)
        AG.config.system.set_centre_of_rotation(*gold)
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed rolled and inverted detector x A")

        gold = (1.5, 0.2)
        AG.config.system.set_centre_of_rotation(*gold)
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed rolled and inverted detector x B")


        AG = AcquisitionGeometry.create_Parallel3D(detector_direction_x=[0.5,0.5,0],rotation_axis_position=[0.5,0.,0])
        gold = (-1.5, -0.2)
        AG.config.system.set_centre_of_rotation(*gold)
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed rotated detector x A")

        gold = (1.5, 0.2)
        AG.config.system.set_centre_of_rotation(*gold)
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed rotated detector x B")

        AG = AcquisitionGeometry.create_Parallel3D(detector_direction_y=[0.0,-0.5,0.5],rotation_axis_position=[0.5,0.,0.])
        gold = (-1.5, -0.2)
        AG.config.system.set_centre_of_rotation(*gold)
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed tilted detector x A")

        gold = (1.5, 0.2)
        AG.config.system.set_centre_of_rotation(*gold)
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed tilted detector x B")

class Test_Cone2D(unittest.TestCase):
    
    def test_align_reference_frame_cil(self):

        AG = AcquisitionGeometry.create_Cone2D(source_position=[0,50], detector_position=[0.,-100.], rotation_axis_position=[5.,2.])
        AG.set_panel(100)

        AG.config.system.align_reference_frame('cil')

        np.testing.assert_allclose(AG.config.system.source.position, [5,-48], rtol=1E-6)
        np.testing.assert_allclose(AG.config.system.detector.position, [5,102], rtol=1E-6)
        np.testing.assert_allclose(AG.config.system.detector.direction_x, [-1,0], rtol=1E-6)
        np.testing.assert_allclose(AG.config.system.rotation_axis.position, [0,0], rtol=1E-6)


    def test_align_reference_frame_tigre(self):

        ag = AcquisitionGeometry.create_Cone2D(source_position=[0,50], detector_position=[0.,-100.], rotation_axis_position=[5.,2])
        ag.set_panel(100)

        ag_align = ag.copy()
        ag_align.config.system.align_reference_frame('tigre')

        np.testing.assert_allclose(ag_align.config.system.source.position, [0,-ag.dist_source_center], atol=1E-6)
        np.testing.assert_allclose(ag_align.config.system.rotation_axis.position, [0,0], rtol=1E-6)
        
        cos_theta = abs(ag.config.system.source.position[1]-ag.config.system.rotation_axis.position[1])/ ag.dist_source_center
        sin_theta = math.sin(math.acos(cos_theta))

        vec = ag.config.system.detector.position-ag.config.system.source.position
        tmp = abs(vec[1])*cos_theta
        det_y = tmp - ag.dist_source_center
        det_x =np.sqrt(vec[1] ** 2 - tmp **2)
        
        np.testing.assert_allclose(ag_align.config.system.detector.position, [det_x, det_y], rtol=1E-6)

        dir_x = -ag.config.system.detector.direction_x[0] * cos_theta
        dir_y = ag.config.system.detector.direction_x[0] * sin_theta
        np.testing.assert_allclose(ag_align.config.system.detector.direction_x, [dir_x, dir_y], rtol=1E-6)
        

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
        np.testing.assert_almost_equal(out, [500, 1000, 3]) 

        AG = AcquisitionGeometry.create_Cone2D(source_position=[0,-500], detector_position=[0.,1000.], rotation_axis_position=[0.,250.])
        out = AG.config.system.calculate_magnification()
        np.testing.assert_almost_equal(out, [750, 750, 2]) 

        AG = AcquisitionGeometry.create_Cone2D(source_position=[0,-500], detector_position=[0.,1000.], rotation_axis_position=[5.,0.])
        out = AG.config.system.calculate_magnification()
        source_to_object = np.sqrt(5.0**2 + 500.0**2)
        theta = math.atan2(5.0,500.0)
        source_to_detector = 1500.0/math.cos(theta)
        np.testing.assert_almost_equal(out, [source_to_object, source_to_detector - source_to_object, source_to_detector/source_to_object]) 

        AG = AcquisitionGeometry.create_Cone2D(source_position=[0,-500], detector_position=[0.,1000.], rotation_axis_position=[5.,0.],detector_direction_x=[math.sqrt(5),math.sqrt(5)])
        out = AG.config.system.calculate_magnification()
        source_to_object = np.sqrt(5.0**2 + 500.0**2)

        ab = (AG.config.system.rotation_axis.position - AG.config.system.source.position)/source_to_object

        #source_position + d * ab = detector_position + t * detector_direction_x
        #x: d *  ab[0] =  t * detector_direction_x[0]
        #y: -500 + d *  ab[1] = 1000 + t * detector_direction_x[1] 

        # t = (d *  ab[0]) / math.sqrt(5)
        # d = 1500 / (ab[1]  - ab[0])

        source_to_detector = 1500 / (ab[1]  - ab[0])

        np.testing.assert_almost_equal(out, [source_to_object, source_to_detector - source_to_object, source_to_detector/source_to_object]) 

    def test_calculate_centre_of_rotation(self):
        AG = AcquisitionGeometry.create_Cone2D(source_position=[0,-500], detector_position=[0.,1000.])
        out = AG.config.system.calculate_centre_of_rotation()
        gold = (0,0)
        np.testing.assert_allclose(out, gold, err_msg="Failed basic")

        AG = AcquisitionGeometry.create_Cone2D(source_position=[0,-500], detector_position=[0.,1000.], rotation_axis_position=[0.5,0.])
        out = AG.config.system.calculate_centre_of_rotation()
        gold = (1.5,0)
        np.testing.assert_allclose(out, gold, err_msg="Failed positive offset")

        AG = AcquisitionGeometry.create_Cone2D(source_position=[0,-500], detector_position=[0.,1000.], rotation_axis_position=[-0.5,0.])
        out = AG.config.system.calculate_centre_of_rotation()
        gold = (-1.5,0)
        np.testing.assert_allclose(out, gold, err_msg="Failed negative offset")

        AG = AcquisitionGeometry.create_Cone2D(source_position=[0,-500], detector_position=[0.,1000.], rotation_axis_position=[0.5,0.], detector_direction_x=[-1,0])
        out = AG.config.system.calculate_centre_of_rotation()
        gold = (-1.5,0)
        np.testing.assert_allclose(out, gold, err_msg="Failed detector direction")

        #offset * mag = 1
        theta = math.pi/4 #detector angle
        phi = math.atan2(1,1000) #ray through rotation axis angle

        L = math.sin(theta)
        X1 = L / math.tan(theta)
        X2 = L / math.tan(math.pi/2 - theta- phi)
        distance = X1 + X2

        AG = AcquisitionGeometry.create_Cone2D(source_position=[0,-500], detector_position=[0.,500.], detector_direction_x=[0.5,0.5],rotation_axis_position=[0.5,0.])
        out = AG.config.system.calculate_centre_of_rotation()
        gold = (distance,0)
        np.testing.assert_allclose(out, gold, err_msg="Failed with rotated detector")

    def test_set_centre_of_rotation(self):

        AG = AcquisitionGeometry.create_Cone2D(source_position=[0,-500], detector_position=[0.,1000.])

        offset_in = 1.5
        AG.config.system.set_centre_of_rotation(offset_in)
        out = AG.config.system.calculate_centre_of_rotation()
        gold = (offset_in,0)
        np.testing.assert_allclose(out, gold, err_msg="Failed positive offset")

        offset_in = -1.5
        AG.config.system.set_centre_of_rotation(offset_in)
        out = AG.config.system.calculate_centre_of_rotation()
        gold = (offset_in,0)
        np.testing.assert_allclose(out, gold, err_msg="Failed negative offset")

        offset_in = 0
        AG.config.system.set_centre_of_rotation(offset_in)
        out = AG.config.system.calculate_centre_of_rotation()
        gold = (offset_in,0)
        np.testing.assert_allclose(out, gold, err_msg="Failed reset offset", atol=1e-10)

        AG = AcquisitionGeometry.create_Cone2D(source_position=[0,-500], detector_position=[0.,1000.], detector_direction_x=[-1,0])

        offset_in = 1.5
        AG.config.system.set_centre_of_rotation(offset_in)
        out = AG.config.system.calculate_centre_of_rotation()
        gold = (offset_in,0)
        np.testing.assert_allclose(out, gold, err_msg= "Failed positive offset")

        offset_in = -1.5
        AG.config.system.set_centre_of_rotation(offset_in)
        out = AG.config.system.calculate_centre_of_rotation()
        gold = (offset_in,0)
        np.testing.assert_allclose(out, gold, err_msg="Failed negative offset")

        offset_in = 0
        AG.config.system.set_centre_of_rotation(offset_in)
        out = AG.config.system.calculate_centre_of_rotation()
        gold = (offset_in,0)
        np.testing.assert_allclose(out, gold, err_msg="Failed reset offset", atol=1e-10)


        AG = AcquisitionGeometry.create_Cone2D(source_position=[0,-500], detector_position=[0.,500.], detector_direction_x=[0.5,0.5],rotation_axis_position=[0.5,0.])
        offset_in = 1.5
        AG.config.system.set_centre_of_rotation(offset_in)
        out = AG.config.system.calculate_centre_of_rotation()
        gold = (offset_in,0)
        np.testing.assert_allclose(out, gold, err_msg="Failed positive offset")

        offset_in = -1.5
        AG.config.system.set_centre_of_rotation(offset_in)
        out = AG.config.system.calculate_centre_of_rotation()
        gold = (offset_in,0)
        np.testing.assert_allclose(out, gold, err_msg="Failed negative offset")

        offset_in = 0
        AG.config.system.set_centre_of_rotation(offset_in)
        out = AG.config.system.calculate_centre_of_rotation()
        gold = (offset_in,0)
        np.testing.assert_allclose(out, gold, err_msg="Failed reset offset", atol=1e-10)


class Test_Cone3D(unittest.TestCase):
    
    def test_align_reference_frame_cil(self):
        AG = AcquisitionGeometry.create_Cone3D(source_position=[5,500,0],detector_position=[5.,-1000.,0], rotation_axis_position=[5,0,0], rotation_axis_direction=[0,0,-1])
        AG.config.system.align_reference_frame('cil')

        np.testing.assert_allclose(AG.config.system.source.position, [0,-500, 0], rtol=1E-6)
        np.testing.assert_allclose(AG.config.system.detector.position, [0,1000,0], rtol=1E-6)
        np.testing.assert_allclose(AG.config.system.detector.direction_x, [1,0,0], rtol=1E-6)
        np.testing.assert_allclose(AG.config.system.detector.direction_y, [0,0,-1], rtol=1E-6)
        np.testing.assert_allclose(AG.config.system.rotation_axis.position, [0,0,0], rtol=1E-6)
        np.testing.assert_allclose(AG.config.system.rotation_axis.direction, [0,0,1], rtol=1E-6)


    def test_align_reference_frame_tigre(self):
        AG = AcquisitionGeometry.create_Cone3D(source_position=[5,500,0],detector_position=[5.,-1000.,0], rotation_axis_position=[5,0,0], rotation_axis_direction=[0,0,-1])
        AG.config.system.align_reference_frame('tigre')

        np.testing.assert_allclose(AG.config.system.source.position, [0,-500, 0], rtol=1E-6)
        np.testing.assert_allclose(AG.config.system.detector.position, [0,1000,0], rtol=1E-6)
        np.testing.assert_allclose(AG.config.system.detector.direction_x, [1,0,0], rtol=1E-6)
        np.testing.assert_allclose(AG.config.system.detector.direction_y, [0,0,-1], rtol=1E-6)
        np.testing.assert_allclose(AG.config.system.rotation_axis.position, [0,0,0], rtol=1E-6)
        np.testing.assert_allclose(AG.config.system.rotation_axis.direction, [0,0,1], rtol=1E-6)


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

        #raise error if cannot extract a centre slice
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
        source_to_object = np.sqrt(5.0**2 + 500.0**2)
        theta = math.atan2(5.0,500.0)
        source_to_detector = 1500.0/math.cos(theta)
        self.assertEqual(out, [source_to_object, source_to_detector - source_to_object, source_to_detector/source_to_object]) 

        AG = AcquisitionGeometry.create_Cone3D(source_position=[0,-500,0], detector_position=[0.,1000.,0], rotation_axis_position=[0.,0.,5.])
        out = AG.config.system.calculate_magnification()
        source_to_object = np.sqrt(5.0**2 + 500.0**2)
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
        source_to_object = np.sqrt(5.0**2 + 500.0**2)

        ab = (AG.config.system.rotation_axis.position - AG.config.system.source.position).astype(np.float64)/source_to_object

        #source_position + d * ab = detector_position + t * detector_direction_x
        #x: d *  ab[0] =  t * detector_direction_x[0]
        #y: -500 + d *  ab[1] = 1000 + t * detector_direction_x[1] 

        # t = (d *  ab[0]) / math.sqrt(5)
        # d = 1500 / (ab[1]  - ab[0])

        source_to_detector = 1500 / (ab[1]  - ab[0])
        self.assertEqual(out, [source_to_object, source_to_detector - source_to_object, source_to_detector/source_to_object]) 

    def test_calculate_centre_of_rotation(self):
        AG = AcquisitionGeometry.create_Cone3D(source_position=[0,-500,0], detector_position=[0.,1000.,0])
        out = AG.config.system.calculate_centre_of_rotation()
        gold = (0,0)
        np.testing.assert_allclose(out, gold, err_msg="Failed basic")

        angle = math.pi/4 
        AG = AcquisitionGeometry.create_Cone3D(source_position=[0,-500,0], detector_position=[0.,1000.,0], rotation_axis_position=[0.5,0.,0.], rotation_axis_direction=[0.5,0,0.5])
        out = AG.config.system.calculate_centre_of_rotation()
        gold = (1.5,angle)
        np.testing.assert_allclose(out, gold, err_msg="Failed positive offset")

        AG = AcquisitionGeometry.create_Cone3D(source_position=[0,-500,0], detector_position=[0.,1000.,0], rotation_axis_position=[-0.5,0.,0.], rotation_axis_direction=[-0.5,0,0.5])
        out = AG.config.system.calculate_centre_of_rotation()
        gold = (-1.5,-angle)
        np.testing.assert_allclose(out, gold, err_msg="Failed negative offset")

        AG = AcquisitionGeometry.create_Cone3D(source_position=[0,-500,0], detector_position=[0.,1000.,0], rotation_axis_position=[0.5,0.,0.], rotation_axis_direction=[0.5,0,0.5], detector_direction_x=[-1,0,0.])
        out = AG.config.system.calculate_centre_of_rotation()
        gold = (-1.5,-angle)
        np.testing.assert_allclose(out, gold, err_msg="Failed detector direction_x")        

        AG = AcquisitionGeometry.create_Cone3D(source_position=[0,-500,0], detector_position=[0.,1000.,0], rotation_axis_position=[0.5,0.,0.], rotation_axis_direction=[0.5,0,0.5], detector_direction_y=[0,0,-1])
        out = AG.config.system.calculate_centre_of_rotation()
        gold = (1.5,math.pi-angle)
        np.testing.assert_allclose(out, gold, err_msg="Failed detector direction_y")        

        AG = AcquisitionGeometry.create_Cone3D(source_position=[0,-500,0], detector_position=[0.,1000.,0], rotation_axis_position=[0.5,0.,0.], rotation_axis_direction=[-0.5,0,-0.5], detector_direction_y=[0,0,-1])
        out = AG.config.system.calculate_centre_of_rotation()
        gold = (1.5,-angle)
        np.testing.assert_allclose(out, gold, err_msg= "Failed invert rotate axis")        

        AG = AcquisitionGeometry.create_Cone3D(source_position=[0,-500,0], detector_position=[0.,1000.,0], rotation_axis_position=[0.5,0.,0.], rotation_axis_direction=[0,0,-1])
        out = AG.config.system.calculate_centre_of_rotation()
        gold = (1.5,math.pi)
        np.testing.assert_allclose(out, gold, err_msg="Failed invert rotate axis")        

        #offset * mag = 1
        theta = math.pi/4 #detector angle
        phi = math.atan2(1,1000) #ray through rotation axis angle

        L = math.sin(theta)
        X1 = L / math.tan(theta)
        X2 = L / math.tan(math.pi/2 - theta- phi)
        distance = X1 + X2

        AG = AcquisitionGeometry.create_Cone3D(source_position=[0,-500,0], detector_position=[0.,500.,0], detector_direction_x=[0.5,0.5,0],rotation_axis_position=[0.5,0.,0])
        out = AG.config.system.calculate_centre_of_rotation()
        gold = (distance,0)
        np.testing.assert_allclose(out, gold, err_msg="Failed with rotated detector")


        #offset * mag = 1
        theta = math.pi/4 #detector angle
        phi = math.atan2(1,1000) #ray through rotation axis angle
        psi = math.atan2(1,500) 
        Y = 2 * math.sin(math.pi/2-psi) / math.sin(math.pi/2-theta+psi)
        L = -Y * math.sin(theta)

        X = L * math.tan(phi)
        angle = math.atan2(X,Y)

        AG = AcquisitionGeometry.create_Cone3D(source_position=[0,-500,0], detector_position=[0.,500.,0], detector_direction_y=[0.0,-0.5,0.5],rotation_axis_position=[0.5,0.,0.])
        out = AG.config.system.calculate_centre_of_rotation()
        gold = (1.0,angle)
        np.testing.assert_allclose(out, gold, err_msg="Failed tilted detector")

    def test_set_centre_of_rotation(self):

        AG = AcquisitionGeometry.create_Cone3D(source_position=[0,-500,0], detector_position=[0.,1000.,0])

        gold = (1.5, 0)
        AG.config.system.set_centre_of_rotation(*gold)
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed positive offset")

        gold = (-1.5, 0)
        AG.config.system.set_centre_of_rotation(*gold)
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed negative offset")

        gold = (0, 0)
        AG.config.system.set_centre_of_rotation(*gold)
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed reset offset")

        gold = (0.0, 0.2)
        AG.config.system.set_centre_of_rotation(*gold)
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed positive angle")

        gold = (0.0, -0.2)
        AG.config.system.set_centre_of_rotation(*gold)
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed negative angle")

        gold = (0.0, 0.0)
        AG.config.system.set_centre_of_rotation(*gold)
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed reset angle")

        gold = (-1.5, -0.2)
        AG.config.system.set_centre_of_rotation(*gold)
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed combination A")

        gold = (1.5, 0.2)
        AG.config.system.set_centre_of_rotation(*gold)
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed combination A")



        AG = AcquisitionGeometry.create_Cone3D(source_position=[0,-500,0], detector_position=[0.,1000.,0], detector_direction_x=[-1,0,0.])
        gold = (-1.5, -0.2)
        AG.config.system.set_centre_of_rotation(*gold)
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed invert detector x A")

        gold = (1.5, 0.2)
        AG.config.system.set_centre_of_rotation(*gold)
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed invert detector x B")


        AG = AcquisitionGeometry.create_Cone3D(source_position=[0,-500,0], detector_position=[0.,1000.,0], rotation_axis_direction=[0,0,-1])
        
        gold = (-1.5, -0.2)
        AG.config.system.set_centre_of_rotation(*gold)
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed invert rotate axis A")

        gold = (1.5, 0.2)
        AG.config.system.set_centre_of_rotation(*gold)
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed invert rotate axis B")


        AG = AcquisitionGeometry.create_Cone3D(source_position=[0,-500,0], detector_position=[0.,1000.,0], detector_direction_y=[0,0,-1])
        gold = (-1.5, -0.2)
        AG.config.system.set_centre_of_rotation(*gold)
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed invert detector y A")

        gold = (1.5, 0.2)
        AG.config.system.set_centre_of_rotation(*gold)
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed invert detector y A")

        AG = AcquisitionGeometry.create_Cone3D(source_position=[0,-500,0], detector_position=[0.,1000.,0], rotation_axis_position=[0.5,0.,0.], rotation_axis_direction=[-0.5,0,-0.5], detector_direction_y=[0,0,-1])
        gold = (-1.5, -0.2)
        AG.config.system.set_centre_of_rotation(*gold)
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed rolled and inverted detector x A")

        gold = (1.5, 0.2)
        AG.config.system.set_centre_of_rotation(*gold)
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed rolled and inverted detector x A")


        AG = AcquisitionGeometry.create_Cone3D(source_position=[0,-500,0], detector_position=[0.,500.,0], detector_direction_x=[0.5,0.5,0],rotation_axis_position=[0.5,0.,0])
        gold = (-1.5, -0.2)
        AG.config.system.set_centre_of_rotation(*gold)
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed rotated detector x A")

        gold = (1.5, 0.2)
        AG.config.system.set_centre_of_rotation(*gold)
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed rotated detector x B")

        AG = AcquisitionGeometry.create_Cone3D(source_position=[0,-500,0], detector_position=[0.,500.,0], detector_direction_y=[0.0,-0.5,0.5],rotation_axis_position=[0.5,0.,0.])
        gold = (-1.5, -0.2)
        AG.config.system.set_centre_of_rotation(*gold)
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed tilted detector x A")

        gold = (1.5, 0.2)
        AG.config.system.set_centre_of_rotation(*gold)
        out = AG.config.system.calculate_centre_of_rotation()
        np.testing.assert_allclose(out, gold, err_msg="Failed tilted detector x B")

class TestSubset(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()
    def tearDown(self) -> None:
        return super().tearDown()

    def test_partition_indices_int(self):
        par = Partitioner()

        num_batches = 4
        indices = 9
        ret = par._partition_indices(num_batches, indices, stagger=False)
        gold = [[0, 1, 2], [3, 4] ,[5, 6], [7, 8]]

        self.assertListEqual(ret, gold)

        ret = par._partition_indices(num_batches, indices, stagger=True)
        gold = [[0, 4, 8], [1, 5], [2, 6], [3, 7]]

        self.assertListEqual(ret, gold)

    def test_partition_indices_list(self):
        par = Partitioner()

        num_batches = 4
        num_indices = 9
        indices = list(range(num_indices))
        ret = par._partition_indices(num_batches, indices, stagger=False)
        gold = [[0, 1, 2], [3, 4] ,[5, 6], [7, 8]]

        self.assertListEqual(ret, gold)

        ret = par._partition_indices(num_batches, indices, stagger=True)
        gold = [[0, 4, 8], [1, 5], [2, 6], [3, 7]]

        self.assertListEqual(ret, gold)

    def test_AcquisitionData_split_to_BlockGeometry_and_BlockDataContainer(self):
        AG = AcquisitionGeometry.create_Parallel2D(detector_position=[0,10])\
            .set_panel(num_pixels=10)\
            .set_angles(angles=range(9))

        data = AG.allocate(None)
        for i in range(AG.num_projections):
            data.array[i] = i

        self.AcquisitionGeometry_split_to_BlockGeometry(data, 'sequential', 1)
        self.AcquisitionGeometry_split_to_BlockGeometry(data, 'staggered', 1)
        self.AcquisitionGeometry_split_to_BlockGeometry(data, Partitioner.RANDOM_PERMUTATION, 1)

    def test_AcquisitionData_split_to_BlockGeometry_and_BlockDataContainer_2D_order1(self):
        AG = AcquisitionGeometry.create_Parallel2D(detector_position=[0,10])\
            .set_panel(num_pixels=10)\
            .set_angles(angles=range(9))\
            .set_labels(['angle','horizontal'])

        data = AG.allocate(None)
        for i in range(AG.num_projections):
            data.array[i] = i

        self.AcquisitionGeometry_split_to_BlockGeometry(data, 'sequential', 1)
        self.AcquisitionGeometry_split_to_BlockGeometry(data, 'staggered', 1)
        self.AcquisitionGeometry_split_to_BlockGeometry(data, Partitioner.RANDOM_PERMUTATION, 1)
    
    def test_AcquisitionData_split_to_BlockGeometry_and_BlockDataContainer_2D_order2(self):
        
        AG = AcquisitionGeometry.create_Parallel2D(detector_position=[0,10])\
            .set_panel(num_pixels=10)\
            .set_angles(angles=range(9))\
            .set_labels(['horizontal', 'angle'])

        data = AG.allocate(None)
        for i in range(AG.num_projections):
            data.array[i] = i

        self.AcquisitionGeometry_split_to_BlockGeometry(data, 'sequential', 1)
        self.AcquisitionGeometry_split_to_BlockGeometry(data, 'staggered', 1)
        self.AcquisitionGeometry_split_to_BlockGeometry(data, Partitioner.RANDOM_PERMUTATION, 1)

    def AcquisitionGeometry_split_to_BlockGeometry(self, data, method, seed):
        num_batches = 4
        np.random.seed(seed)
        datasplit = data.partition(num_batches, method)
        bg = datasplit.geometry
        ag = data.geometry
        num_indices = ag.num_projections

        gold = [ np.zeros(num_indices, dtype=bool) for _ in range(num_batches) ]
        if method == Partitioner.SEQUENTIAL:
            gold = [[0, 1, 2], [3, 4], [5, 6], [7, 8]]
            
        elif method == Partitioner.STAGGERED:
            gold = [[0, 4, 8], [1, 5], [2, 6], [3, 7]]
            
        elif method == Partitioner.RANDOM_PERMUTATION:
            # with seed==1 
            gold = [[8, 2, 6], [7, 1], [0, 4], [3, 5]]
            

        for i, geo in enumerate(bg):
            np.testing.assert_allclose(geo.angles, np.asarray(gold[i]))
    
    def test_geometry_print_angles(self):
        
        AG = AcquisitionGeometry.create_Parallel2D(detector_position=[0,10])\
            .set_panel(num_pixels=10)\
            .set_angles(angles=range(90))\
            .set_labels(['horizontal', 'angle'])
        AD = AcquisitionData(np.zeros([10,90]), geometry=AG, deep_copy=False)
        
        # redirect print output
        capturedOutput = io.StringIO()                 
        sys.stdout = capturedOutput                    
        
        print(AD.geometry)
        angles = re.findall('Angles [\d]+-[\d]+ in degrees:\s+\[.*\]+', capturedOutput.getvalue(), re.MULTILINE)
        self.assertEqual(angles[0], 'Angles 0-9 in degrees: [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]')
        self.assertEqual(angles[1], 'Angles 80-89 in degrees: [80., 81., 82., 83., 84., 85., 86., 87., 88., 89.]')

        # test output when angles=31
        AG = AcquisitionGeometry.create_Parallel2D(detector_position=[0,10])\
            .set_panel(num_pixels=10)\
            .set_angles(angles=range(31))\
            .set_labels(['horizontal', 'angle'])
        AD = AcquisitionData(np.zeros([10,31]), geometry=AG, deep_copy=False)
        print(AD.geometry)
        angles = re.findall('Angles [\d]+-[\d]+ in degrees:\s+\[.*\]+', capturedOutput.getvalue(), re.MULTILINE)
        self.assertEqual(angles[2], 'Angles 0-9 in degrees: [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]')
        self.assertEqual(angles[3], 'Angles 21-30 in degrees: [21., 22., 23., 24., 25., 26., 27., 28., 29., 30.]')
        
        # test output when angles=30
        AG = AcquisitionGeometry.create_Parallel2D(detector_position=[0,10])\
            .set_panel(num_pixels=10)\
            .set_angles(angles=range(30))\
            .set_labels(['horizontal', 'angle'])
        AD = AcquisitionData(np.zeros([10,30]), geometry=AG, deep_copy=False)
        print(AD.geometry)
        angles = re.findall('Number of positions: 30\n\tAngles [\d]+-[\d]+ in degrees:\s+\[.*\n.*\]+', capturedOutput.getvalue(), re.MULTILINE)
        self.assertEqual(angles[0],\
                'Number of positions: 30\n\tAngles 0-29 in degrees: [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14.,\n 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29.]')
 
        # test no error occurs when angles<20
        AG = AcquisitionGeometry.create_Parallel2D(detector_position=[0,10])\
            .set_panel(num_pixels=10)\
            .set_angles(angles=range(17))\
            .set_labels(['horizontal', 'angle'])
        AD = AcquisitionData(np.zeros([10,17]), geometry=AG, deep_copy=False)
        print(AD.geometry)                                
                         
        # test no error occurs when angles<10
        AG = AcquisitionGeometry.create_Parallel2D(detector_position=[0,10])\
            .set_panel(num_pixels=10)\
            .set_angles(angles=range(9))\
            .set_labels(['horizontal', 'angle'])
        AD = AcquisitionData(np.zeros([10,9]), geometry=AG, deep_copy=False)
        print(AD.geometry)

        # test no error occurs when angle=1
        AG = AcquisitionGeometry.create_Parallel2D(detector_position=[0,10])\
            .set_panel(num_pixels=10)\
            .set_angles(angles=range(1))\
            .set_labels(['horizontal', 'angle'])
        AD = AcquisitionData(np.zeros([10,]), geometry=AG, deep_copy=False)
        print(AD.geometry)

        # return to standard print output
        sys.stdout = sys.__stdout__ 

