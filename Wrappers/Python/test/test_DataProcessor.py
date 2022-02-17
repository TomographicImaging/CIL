
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
from cil.framework import DataProcessor
from cil.framework import DataContainer
from cil.framework import ImageData
from cil.framework import AcquisitionData
from cil.framework import ImageGeometry
from cil.framework import AcquisitionGeometry
from cil.utilities import dataexample
from timeit import default_timer as timer

from cil.framework import AX, CastDataContainer, PixelByPixelDataProcessor

from cil.io import NEXUSDataReader
from cil.processors import CentreOfRotationCorrector, CofR_xcorrelation, CofR_image_sharpness
from cil.processors import TransmissionAbsorptionConverter, AbsorptionTransmissionConverter
from cil.processors import Slicer, Binner, MaskGenerator, Masker, Padder
import wget
import os

from utils import has_gpu_tigre, has_gpu_astra


try:
    import tigre
    has_tigre = True
except ModuleNotFoundError:
    print(  "This plugin requires the additional package TIGRE\n" +
            "Please install it via conda as tigre from the ccpi channel\n"+
            "Minimal version is 21.01")
    has_tigre = False
else:
    from cil.plugins.tigre import FBP as TigreFBP
    from cil.plugins.tigre import ProjectionOperator

try:
    import tomophantom
    has_tomophantom = True
except ModuleNotFoundError:
    print(  "This plugin requires the additional package tomophantom\n" +
            "Please install it via conda as tomophantom from the ccpi channel\n")
    has_tomophantom = False
else:
    from cil.plugins import TomoPhantom

try:
    import astra
    from cil.plugins.astra import FBP as AstraFBP
    has_astra = True
except ModuleNotFoundError:
    has_astra = False


class TestPadder(unittest.TestCase):
    def setUp(self):
        ray_direction = [0.1, 3.0]
        detector_position = [-1.3, 1000.0]
        detector_direction_row = [1.0, 0.2]
        rotation_axis_position = [0.1, 2.0]

        AG = AcquisitionGeometry.create_Parallel2D(ray_direction=ray_direction, 
                                                    detector_position=detector_position, 
                                                    detector_direction_x=detector_direction_row, 
                                                    rotation_axis_position=rotation_axis_position)

        # test int shortcut
        self.num_angles = 10
        angles = numpy.linspace(0, 360, self.num_angles, dtype=numpy.float32)

        self.num_channels = 10
        self.num_pixels = 5
        AG.set_channels(num_channels=self.num_channels)
        AG.set_angles(angles, initial_angle=10, angle_unit='radian')
        AG.set_panel(self.num_pixels, pixel_size=0.1)

        data = AG.allocate('random')
        
        self.data = data
        self.AG = AG

    def test_constant_with_int(self):
        data = self.data
        AG = self.AG
        num_pad = 5
        b = Padder.constant(pad_width=num_pad)
        b.set_input(data)
        data_padded = b.process()

        data_new = numpy.zeros((self.num_channels + 2*num_pad, self.num_angles, self.num_pixels + 2*num_pad), dtype=numpy.float32)
        data_new[5:15,:,5:10] = data.as_array()
        # new_angles = numpy.zeros((20,), dtype=numpy.float32)
        # new_angles[5:15] = angles

        geometry_padded = AG.copy()
        geometry_padded.set_channels(num_channels=20)
        geometry_padded.set_panel(15, pixel_size=0.1)
        # geometry_padded.set_angles(new_angles, initial_angle=10, angle_unit='radian')
                
        self.assertTrue(data_padded.geometry == geometry_padded)
        numpy.testing.assert_allclose(data_padded.as_array(), data_new, rtol=1E-6)
    
    def test_constant_with_tuple(self):
        data = self.data
        AG = self.AG
        # test tuple
        pad_tuple = (5,2)
        b = Padder.constant(pad_width=pad_tuple)
        b.set_input(data)
        data_padded = b.process()

        data_new = numpy.zeros((self.num_channels + sum(pad_tuple), self.num_angles, self.num_pixels + sum(pad_tuple)),\
                     dtype=numpy.float32)
        data_new[pad_tuple[0]:-pad_tuple[1],\
            :,\
                pad_tuple[0]:-pad_tuple[1]] = data.as_array()
        # new_angles = numpy.zeros((17,), dtype=numpy.float32)
        # new_angles[5:15] = angles

        geometry_padded = AG.copy()
        geometry_padded.set_channels(num_channels=self.num_channels+sum(pad_tuple))
        geometry_padded.set_panel(self.num_pixels+sum(pad_tuple), pixel_size=0.1)
        # geometry_padded.set_angles(new_angles, initial_angle=10, angle_unit='radian')
                
        self.assertTrue(data_padded.geometry == geometry_padded)
        numpy.testing.assert_allclose(data_padded.as_array(), data_new, rtol=1E-6)

    def test_constant_with_dictionary1(self):
        data = self.data
        AG = self.AG
        # test dictionary + constant values
        pad_tuple = (5,2)
        const = 5.0
        b = Padder.constant(pad_width={'channel':pad_tuple}, constant_values=const)
        b.set_input(data)
        data_padded = b.process()

        data_new = const * numpy.ones((self.num_channels + sum(pad_tuple), self.num_angles, self.num_pixels),\
                     dtype=numpy.float32)

        # data_new = 5*numpy.ones((10,10,5), dtype=numpy.float32)
        data_new[pad_tuple[0]:-pad_tuple[1],:,:] = data.as_array()
        geometry_padded = AG.copy()
        geometry_padded.set_channels(num_channels=17)

        self.assertTrue(data_padded.geometry == geometry_padded)
        numpy.testing.assert_allclose(data_padded.as_array(), data_new, rtol=1E-6)
    
    def test_constant_with_dictionary2(self):
        data = self.data
        AG = self.AG
        # test dictionary + constant values
        pad_tuple1 = (5,2)
        pad_tuple2 = (2,5)
        const = 5.0
        b = Padder.constant(pad_width={'channel':pad_tuple1, 'horizontal':pad_tuple2}, constant_values=const)
        b.set_input(data)
        data_padded = b.process()

        data_new = const * numpy.ones((self.num_channels + sum(pad_tuple1), self.num_angles, \
            self.num_pixels + sum(pad_tuple2)),\
                     dtype=numpy.float32)

        # data_new = 5*numpy.ones((10,10,5), dtype=numpy.float32)
        data_new[pad_tuple1[0]:-pad_tuple1[1],:,pad_tuple2[0]:-pad_tuple2[1]] = data.as_array()
        geometry_padded = AG.copy()
        geometry_padded.set_channels(num_channels=17)
        geometry_padded.set_panel(self.num_pixels + sum(pad_tuple2),pixel_size=0.1)
        
        self.assertTrue(data_padded.geometry == geometry_padded)
        numpy.testing.assert_allclose(data_padded.as_array(), data_new, rtol=1E-6)
    
    def test_edge_with_int(self):
        AG = self.AG
        value = -11.
        data = self.AG.allocate(value)
        
        num_pad = 5
        b = Padder.edge(pad_width=num_pad)
        b.set_input(data)
        data_padded = b.process()

        data_new = value * numpy.ones((self.num_channels + 2*num_pad, self.num_angles, self.num_pixels + 2*num_pad), dtype=numpy.float32)
        data_new[5:15,:,5:10] = data.as_array()

        # new_angles = numpy.zeros((20,), dtype=numpy.float32)
        # new_angles[5:15] = angles

        geometry_padded = AG.copy()
        geometry_padded.set_channels(num_channels=20)
        geometry_padded.set_panel(15, pixel_size=0.1)
        # geometry_padded.set_angles(new_angles, initial_angle=10, angle_unit='radian')
                
        self.assertTrue(data_padded.geometry == geometry_padded)
        numpy.testing.assert_allclose(data_padded.as_array(), data_new, rtol=1E-6)
    
    def test_edge_with_tuple(self):
        AG = self.AG
        value = -11.
        data = self.AG.allocate(value)
        # test tuple
        pad_tuple = (5,2)
        b = Padder.edge(pad_width=pad_tuple)
        b.set_input(data)
        data_padded = b.process()

        data_new = value * numpy.ones((self.num_channels + sum(pad_tuple), self.num_angles, self.num_pixels + sum(pad_tuple)),\
                     dtype=numpy.float32)
        data_new[pad_tuple[0]:-pad_tuple[1],\
            :,\
                pad_tuple[0]:-pad_tuple[1]] = data.as_array()
        # new_angles = numpy.zeros((17,), dtype=numpy.float32)
        # new_angles[5:15] = angles

        geometry_padded = AG.copy()
        geometry_padded.set_channels(num_channels=self.num_channels+sum(pad_tuple))
        geometry_padded.set_panel(self.num_pixels+sum(pad_tuple), pixel_size=0.1)
        # geometry_padded.set_angles(new_angles, initial_angle=10, angle_unit='radian')
                
        self.assertTrue(data_padded.geometry == geometry_padded)
        numpy.testing.assert_allclose(data_padded.as_array(), data_new, rtol=1E-6)

    def test_edge_with_dictionary(self):
        AG = self.AG
        value = -11.
        data = self.AG.allocate(value)
        # test dictionary + constant values
        pad_tuple = (5,2)
        b = Padder.edge(pad_width={'channel':pad_tuple})
        b.set_input(data)
        data_padded = b.process()

        data_new = value * numpy.ones((self.num_channels + sum(pad_tuple), self.num_angles, self.num_pixels),\
                     dtype=numpy.float32)

        # data_new = 5*numpy.ones((10,10,5), dtype=numpy.float32)
        data_new[pad_tuple[0]:-pad_tuple[1],:,:] = data.as_array()
        geometry_padded = AG.copy()
        geometry_padded.set_channels(num_channels=17)

        self.assertTrue(data_padded.geometry == geometry_padded)
        numpy.testing.assert_allclose(data_padded.as_array(), data_new, rtol=1E-6)

    def test_linear_ramp_with_int(self):
        AG = self.AG
        value = -11.
        data = self.AG.allocate(value)
        
        num_pad = 5
        b = Padder.linear_ramp(pad_width=num_pad)
        b.set_input(data)
        data_padded = b.process()

        data_new = 0 * numpy.ones((self.num_channels + 2*num_pad, self.num_angles, self.num_pixels + 2*num_pad), dtype=numpy.float32)
        data_new[5:15,:,5:10] = data.as_array()

        # new_angles = numpy.zeros((20,), dtype=numpy.float32)
        # new_angles[5:15] = angles

        geometry_padded = AG.copy()
        geometry_padded.set_channels(num_channels=20)
        geometry_padded.set_panel(15, pixel_size=0.1)
        # geometry_padded.set_angles(new_angles, initial_angle=10, angle_unit='radian')
                
        self.assertTrue(data_padded.geometry == geometry_padded)
        # numpy.testing.assert_allclose(data_padded.as_array(), data_new, rtol=1E-6)
        self.assertAlmostEqual(data_padded.as_array().ravel()[0] , 0.)
        self.assertAlmostEqual(data_padded.as_array().ravel()[-1] , 0.)
    
    def test_linear_ramp_with_tuple(self):
        AG = self.AG
        value = -11.
        data = self.AG.allocate(value)
        # test tuple
        pad_tuple = (5,2)
        b = Padder.edge(pad_width=pad_tuple)
        b.set_input(data)
        data_padded = b.process()

        data_new = value * numpy.ones((self.num_channels + sum(pad_tuple), self.num_angles, self.num_pixels + sum(pad_tuple)),\
                     dtype=numpy.float32)
        data_new[pad_tuple[0]:-pad_tuple[1],\
            :,\
                pad_tuple[0]:-pad_tuple[1]] = data.as_array()
        # new_angles = numpy.zeros((17,), dtype=numpy.float32)
        # new_angles[5:15] = angles

        geometry_padded = AG.copy()
        geometry_padded.set_channels(num_channels=self.num_channels+sum(pad_tuple))
        geometry_padded.set_panel(self.num_pixels+sum(pad_tuple), pixel_size=0.1)
        # geometry_padded.set_angles(new_angles, initial_angle=10, angle_unit='radian')
                
        self.assertTrue(data_padded.geometry == geometry_padded)
        numpy.testing.assert_allclose(data_padded.as_array(), data_new, rtol=1E-6)

    def test_linear_ramp_with_dictionary(self):
        AG = self.AG
        value = -11.
        data = self.AG.allocate(value)
        # test dictionary + constant values
        pad_tuple = (5,2)
        b = Padder.edge(pad_width={'channel':pad_tuple})
        b.set_input(data)
        data_padded = b.process()

        data_new = value * numpy.ones((self.num_channels + sum(pad_tuple), self.num_angles, self.num_pixels),\
                     dtype=numpy.float32)

        # data_new = 5*numpy.ones((10,10,5), dtype=numpy.float32)
        data_new[pad_tuple[0]:-pad_tuple[1],:,:] = data.as_array()
        geometry_padded = AG.copy()
        geometry_padded.set_channels(num_channels=17)

        self.assertTrue(data_padded.geometry == geometry_padded)
        numpy.testing.assert_allclose(data_padded.as_array(), data_new, rtol=1E-6)
    
    


has_astra = has_astra and has_gpu_astra()
has_tigre = has_tigre and has_gpu_tigre()


class TestBinner(unittest.TestCase):
    def test_Binner(self):
        #test parallel 2D case
        
        ray_direction = [0.1, 3.0]
        detector_position = [-1.3, 1000.0]
        detector_direction_row = [1.0, 0.2]
        rotation_axis_position = [0.1, 2.0]
        
        AG = AcquisitionGeometry.create_Parallel2D(ray_direction=ray_direction, 
                                                    detector_position=detector_position, 
                                                    detector_direction_x=detector_direction_row, 
                                                    rotation_axis_position=rotation_axis_position)
        
        angles = numpy.linspace(0, 360, 10, dtype=numpy.float32)
        
        AG.set_channels(num_channels=10)
        AG.set_angles(angles, initial_angle=10, angle_unit='radian')
        AG.set_panel(5, pixel_size=0.1)
        
        data = AG.allocate('random')
        
        b = Binner(roi={'channel': (1, -2, 3),
                        'angle': (2, 9, 2),
                        'horizontal': (2, -1)})
        
        b.set_input(data)
        data_binned = b.process()
        
        AG_binned = AG.clone()
        AG_binned.set_channels(num_channels=2)
        AG_binned.set_panel(2, pixel_size=0.1)
        angles_new = (angles[2:8:2] + angles[3:9:2])/2
        AG_binned.set_angles(angles_new, initial_angle=10, angle_unit='radian')
        
        data_new = (data.as_array()[1:6:3, :, :] + data.as_array()[2:7:3, :, :] + data.as_array()[3:8:3, :, :]) / 3
        data_new = (data_new[:, 2:8:2, :] + data_new[:, 3:9:2, :]) / 2
        data_new = data_new[:, :, 2:-1]
        
        self.assertTrue(data_binned.geometry == AG_binned)
        numpy.testing.assert_allclose(data_binned.as_array(), data_new, rtol=1E-6)
        
        #%%
        #test parallel 3D case
        
        ray_direction = [0.1, 3.0, 0.4]
        detector_position = [-1.3, 1000.0, 2]
        detector_direction_row = [1.0, 0.2, 0.0]
        detector_direction_col = [0.0 ,0.0, 1.0]
        rotation_axis_position = [0.1, 2.0, 0.5]
        rotation_axis_direction = [0.1, 2.0, 0.5]
        
        AG = AcquisitionGeometry.create_Parallel3D(ray_direction=ray_direction, 
                                                    detector_position=detector_position, 
                                                    detector_direction_x=detector_direction_row, 
                                                    detector_direction_y=detector_direction_col,
                                                    rotation_axis_position=rotation_axis_position,
                                                    rotation_axis_direction=rotation_axis_direction)
        
        angles = numpy.linspace(0, 360, 10, dtype=numpy.float32)
        
        AG.set_channels(num_channels=10)
        AG.set_angles(angles, initial_angle=10, angle_unit='radian')
        AG.set_panel((10, 5), pixel_size=(0.1, 0.2))
        AG.dimension_labels = ['vertical',\
                                'horizontal',\
                                'angle',\
                                'channel']
        
        data = AG.allocate('random')
        
        b = Binner(roi={'channel': (None, 1),
                        'angle': -1,
                        'horizontal': (1, None, 2),
                        'vertical': (0 , 4, 1)})
        b.set_input(data)
        data_binned = b.process()
        
        dimension_labels_binned = list(data.geometry.dimension_labels)
        dimension_labels_binned.remove('channel')
        
        AG_binned = AG.clone()
        AG_binned.dimension_labels = dimension_labels_binned
        AG_binned.set_channels(num_channels=1)
        AG_binned.set_panel([4, 4], pixel_size=(0.2, 0.2))
        
        data_new = data.as_array()[:4, :, :, 0]
        data_new = (data_new[:, 1:9:2, :] + data_new[:, 2:10:2, :]) / 2
        
        self.assertTrue(data_binned.geometry == AG_binned)
        numpy.testing.assert_allclose(data_binned.as_array(), data_new, rtol=1E-6)
        
        #%%
        #test cone 3D case
        
        source_position = [0.1, 3.0, 0.4]
        detector_position = [-1.3, 1000.0, 2]
        rotation_axis_position = [0.1, 2.0, 0.5]
        
        AG = AcquisitionGeometry.create_Cone3D(source_position=source_position, 
                                                detector_position=detector_position,
                                                rotation_axis_position=rotation_axis_position)
        
        angles = numpy.linspace(0, 360, 10, dtype=numpy.float32)
        
        AG.set_channels(num_channels=10)
        AG.set_angles(angles, initial_angle=10, angle_unit='radian')
        AG.set_panel((100, 50), pixel_size=(0.1, 0.2))
        AG.dimension_labels = ['vertical',\
                                'horizontal',\
                                'angle',\
                                'channel']
        
        data = AG.allocate('random')
        
        b = Binner(roi={'channel': (None, 1),
                        'angle': -1,
                        'horizontal': (10, None, 2),
                        'vertical': (24, 26, 2)})
        b.set_input(data)
        data_binned = b.process()
        
        dimension_labels_binned = list(data.geometry.dimension_labels)
        dimension_labels_binned.remove('channel')
        dimension_labels_binned.remove('vertical')
        
        AG_binned = AG.subset(vertical='centre')
        AG_binned = AG_binned.subset(channel=0)
        AG_binned.config.panel.num_pixels[0] = 45
        AG_binned.config.panel.pixel_size[0] = 0.2
        AG_binned.config.panel.pixel_size[1] = 0.4
        
        data_new = data.as_array()[:,:,:,0]
        data_new = (data_new[:, 10:99:2, :] + data_new[:, 11:100:2, :]) / 2
        data_new = (data_new[24, :, :] + data_new[25, :, :]) / 2
        
        self.assertTrue(data_binned.geometry == AG_binned)
        numpy.testing.assert_allclose(data_binned.as_array(), data_new, rtol=1E-6)
        
        
        #%% test ImageData
        IG = ImageGeometry(voxel_num_x=20,
                            voxel_num_y=30,
                            voxel_num_z=12,
                            voxel_size_x=0.1,
                            voxel_size_y=0.2,
                            voxel_size_z=0.3,
                            channels=10,
                            center_x=0.2,
                            center_y=0.4,
                            center_z=0.6,
                            dimension_labels = ['vertical',\
                                                'channel',\
                                                'horizontal_y',\
                                                'horizontal_x'])
        
        data = IG.allocate('random')
        
        b = Binner(roi={'channel': (None, None, 2),
                        'horizontal_x': -1,
                        'horizontal_y': (10, None, 2),
                        'vertical': (5, None, 3)})
        b.set_input(data)
        data_binned = b.process()
        
        IG_binned = IG.copy()
        IG_binned.voxel_num_y = 10
        IG_binned.voxel_size_y = 0.2 * 2
        IG_binned.voxel_num_z = 2
        IG_binned.voxel_size_z = 0.3 * 3
        IG_binned.channels = 5
        IG_binned.channel_spacing = 1 * 2.0
        
        data_new = (data.as_array()[:, :-1:2, :, :] + data.as_array()[:, 1::2, :, :]) / 2
        data_new = (data_new[5:-2:3, :, :, :] + data_new[6:-1:3, :, :, :] + data_new[7::3, :, :, :]) / 3
        data_new = (data_new[:, :, 10:-1:2, :] + data_new[:, :, 11::2, :]) / 2
        
        self.assertTrue(data_binned.geometry == IG_binned)
        numpy.testing.assert_allclose(data_binned.as_array(), data_new, rtol=1E-6)

class TestSlicer(unittest.TestCase):      
    def test_Slicer(self):
        
        #test parallel 2D case

        ray_direction = [0.1, 3.0]
        detector_position = [-1.3, 1000.0]
        detector_direction_row = [1.0, 0.2]
        rotation_axis_position = [0.1, 2.0]
        
        AG = AcquisitionGeometry.create_Parallel2D(ray_direction=ray_direction, 
                                                    detector_position=detector_position, 
                                                    detector_direction_x=detector_direction_row, 
                                                    rotation_axis_position=rotation_axis_position)
        
        angles = numpy.linspace(0, 360, 10, dtype=numpy.float32)
        
        AG.set_channels(num_channels=10)
        AG.set_angles(angles, initial_angle=10, angle_unit='radian')
        AG.set_panel(100, pixel_size=0.1)
        
        data = AG.allocate('random')
        
        s = Slicer(roi={'channel': (1, -2, 3),
                        'angle': (2, 9, 2),
                        'horizontal': (10, -11, 7)})
        s.set_input(data)
        data_sliced = s.process()
        
        AG_sliced = AG.clone()
        AG_sliced.set_channels(num_channels=numpy.arange(1, 10-2, 3).shape[0])
        AG_sliced.set_panel([numpy.arange(10, 100-11, 7).shape[0], 1], pixel_size=0.1)
        AG_sliced.set_angles(angles[2:9:2], initial_angle=10, angle_unit='radian')
        
        self.assertTrue(data_sliced.geometry == AG_sliced)
        numpy.testing.assert_allclose(data_sliced.as_array(), numpy.squeeze(data.as_array()[1:-2:3, 2:9:2, 10:-11:7]), rtol=1E-6)
        
        #%%
        #test parallel 3D case
        
        ray_direction = [0.1, 3.0, 0.4]
        detector_position = [-1.3, 1000.0, 2]
        detector_direction_row = [1.0, 0.2, 0.0]
        detector_direction_col = [0.0 ,0.0, 1.0]
        rotation_axis_position = [0.1, 2.0, 0.5]
        rotation_axis_direction = [0.1, 2.0, 0.5]
        
        AG = AcquisitionGeometry.create_Parallel3D(ray_direction=ray_direction, 
                                                    detector_position=detector_position, 
                                                    detector_direction_x=detector_direction_row, 
                                                    detector_direction_y=detector_direction_col,
                                                    rotation_axis_position=rotation_axis_position,
                                                    rotation_axis_direction=rotation_axis_direction)
        
        angles = numpy.linspace(0, 360, 10, dtype=numpy.float32)
        
        AG.set_channels(num_channels=10)
        AG.set_angles(angles, initial_angle=10, angle_unit='radian')
        AG.set_panel((100, 50), pixel_size=(0.1, 0.2))
        AG.dimension_labels = ['vertical',\
                                'horizontal',\
                                'angle',\
                                'channel']
        
        data = AG.allocate('random')
        
        s = Slicer(roi={'channel': (None, 1),
                        'angle': -1,
                        'horizontal': (10, None, 2),
                        'vertical': (10, 12, 1)})
        s.set_input(data)
        data_sliced = s.process()
        
        dimension_labels_sliced = list(data.geometry.dimension_labels)
        dimension_labels_sliced.remove('channel')
        dimension_labels_sliced.remove('vertical')
        
        AG_sliced = AG.clone()
        AG_sliced.dimension_labels = dimension_labels_sliced
        AG_sliced.set_channels(num_channels=1)
        AG_sliced.set_panel([numpy.arange(10, 100, 2).shape[0], numpy.arange(10, 12, 1).shape[0]], pixel_size=(0.1, 0.2))
        
        self.assertTrue(data_sliced.geometry == AG_sliced)
        numpy.testing.assert_allclose(data_sliced.as_array(), numpy.squeeze(data.as_array()[10:12:1, 10::2, :, :1]), rtol=1E-6)
        
        #%%
        #test cone 2D case
        
        source_position = [0.1, 3.0]
        detector_position = [-1.3, 1000.0]
        detector_direction_row = [1.0, 0.2]
        rotation_axis_position = [0.1, 2.0]
        
        AG = AcquisitionGeometry.create_Cone2D(source_position=source_position, 
                                                detector_position=detector_position, 
                                                detector_direction_x=detector_direction_row, 
                                                rotation_axis_position=rotation_axis_position)
        
        angles = numpy.linspace(0, 360, 10, dtype=numpy.float32)
        
        AG.set_channels(num_channels=10)
        AG.set_angles(angles, initial_angle=10, angle_unit='degree')
        AG.set_panel(100, pixel_size=0.1)
        
        data = AG.allocate('random')
        
        s = Slicer(roi={'channel': (1, None, 4),
                        'angle': (2, 9, 2),
                        'horizontal': (10, -10, 5)})
        s.set_input(data)
        data_sliced = s.process()
        
        AG_sliced = AG.clone()
        AG_sliced.set_channels(num_channels=numpy.arange(1,10,4).shape[0])
        AG_sliced.set_angles(AG.config.angles.angle_data[2:9:2], angle_unit='degree', initial_angle=10)
        AG_sliced.set_panel(numpy.arange(10,90,5).shape[0], pixel_size=0.1)
        
        self.assertTrue(data_sliced.geometry == AG_sliced)
        numpy.testing.assert_allclose(data_sliced.as_array(), numpy.squeeze(data.as_array()[1::4, 2:9:2, 10:-10:5]), rtol=1E-6)
        
        #%%
        #test cone 3D case
        
        source_position = [0.1, 3.0, 0.4]
        detector_position = [-1.3, 1000.0, 2]
        rotation_axis_position = [0.1, 2.0, 0.5]
        
        AG = AcquisitionGeometry.create_Cone3D(source_position=source_position, 
                                                detector_position=detector_position,
                                                rotation_axis_position=rotation_axis_position)
        
        angles = numpy.linspace(0, 360, 10, dtype=numpy.float32)
        
        AG.set_channels(num_channels=10)
        AG.set_angles(angles, initial_angle=10, angle_unit='radian')
        AG.set_panel((100, 50), pixel_size=(0.1, 0.2))
        AG.dimension_labels = ['vertical',\
                                'horizontal',\
                                'angle',\
                                'channel']
        
        data = AG.allocate('random')
        
        s = Slicer(roi={'channel': (None, 1),
                        'angle': -1,
                        'horizontal': (10, None, 2),
                        'vertical': (10, -10, 2)})
        s.set_input(data)
        data_sliced = s.process()
        
        dimension_labels_sliced = list(data.geometry.dimension_labels)
        dimension_labels_sliced.remove('channel')
        
        AG_sliced = AG.clone()
        AG_sliced.dimension_labels = dimension_labels_sliced
        AG_sliced.set_channels(num_channels=1)
        AG_sliced.set_panel([numpy.arange(10, 100, 2).shape[0], numpy.arange(10, 50-10, 2).shape[0]], pixel_size=(0.1, 0.2))
        self.assertTrue(data_sliced.geometry == AG_sliced)
        
        numpy.testing.assert_allclose(data_sliced.as_array(), numpy.squeeze(data.as_array()[10:-10:2, 10::2, :, :1]), rtol=1E-6)
        
        #%% test cone 3D - central slice
        s = Slicer(roi={'channel': (None, 1),
                        'angle': -1,
                        'horizontal': (10, None, 2),
                        'vertical': (25, 26)})
        s.set_input(data)
        data_sliced = s.process()
        
        dimension_labels_sliced = list(data.geometry.dimension_labels)
        dimension_labels_sliced.remove('channel')
        dimension_labels_sliced.remove('vertical')
        
        AG_sliced = AG.subset(vertical='centre')
        AG_sliced = AG_sliced.subset(channel=1)
        AG_sliced.config.panel.num_pixels[0] = numpy.arange(10,100,2).shape[0]
        
        self.assertTrue(data_sliced.geometry == AG_sliced)
        numpy.testing.assert_allclose(data_sliced.as_array(), numpy.squeeze(data.as_array()[25:26, 10::2, :, :1]), rtol=1E-6)
        
        
        #%% test ImageData
        IG = ImageGeometry(voxel_num_x=20,
                            voxel_num_y=30,
                            voxel_num_z=12,
                            voxel_size_x=0.1,
                            voxel_size_y=0.2,
                            voxel_size_z=0.3,
                            channels=10,
                            center_x=0.2,
                            center_y=0.4,
                            center_z=0.6,
                            dimension_labels = ['vertical',\
                                                'channel',\
                                                'horizontal_y',\
                                                'horizontal_x'])
        
        data = IG.allocate('random')
        
        s = Slicer(roi={'channel': (None, None, 2),
                        'horizontal_x': -1,
                        'horizontal_y': (10, None, 2),
                        'vertical': (5, None, 3)})
        s.set_input(data)
        data_sliced = s.process()
        
        IG_sliced = IG.copy()
        IG_sliced.voxel_num_y = numpy.arange(10, 30, 2).shape[0]
        IG_sliced.voxel_num_z = numpy.arange(5, 12, 3).shape[0]
        IG_sliced.channels = numpy.arange(0, 10, 2).shape[0]
        
        self.assertTrue(data_sliced.geometry == IG_sliced)
        numpy.testing.assert_allclose(data_sliced.as_array(), numpy.squeeze(data.as_array()[5:12:3, ::2, 10:30:2, :]), rtol=1E-6)


class TestCentreOfRotation_parallel(unittest.TestCase):
    
    def setUp(self):
        data_raw = dataexample.SYNCHROTRON_PARALLEL_BEAM_DATA.get()
        self.data_DLS = data_raw.log()
        self.data_DLS *= -1

    def test_CofR_xcorrelation(self):       

        corr = CofR_xcorrelation(slice_index='centre', projection_index=0, ang_tol=0.1)
        corr.set_input(self.data_DLS.clone())
        ad_out = corr.get_output()
        self.assertAlmostEqual(6.33, ad_out.geometry.config.system.rotation_axis.position[0],places=2)     
        
        corr = CofR_xcorrelation(slice_index=67, projection_index=0, ang_tol=0.1)
        corr.set_input(self.data_DLS.clone())
        ad_out = corr.get_output()
        self.assertAlmostEqual(6.33, ad_out.geometry.config.system.rotation_axis.position[0],places=2)              

    @unittest.skipUnless(has_astra, "ASTRA not installed")
    def test_CofR_image_sharpness_astra(self):
        corr = CofR_image_sharpness(search_range=20, FBP=AstraFBP)
        corr.set_input(self.data_DLS.clone())
        ad_out = corr.get_output()
        self.assertAlmostEqual(6.33, ad_out.geometry.config.system.rotation_axis.position[0],places=1)     


    @unittest.skipUnless(False, "TIGRE not installed")
    def skiptest_test_CofR_image_sharpness_tigre(self): #currently not avaliable for parallel beam
        corr = CofR_image_sharpness(search_range=20, FBP=TigreFBP)
        corr.set_input(self.data_DLS.clone())
        ad_out = corr.get_output()
        self.assertAlmostEqual(6.33, ad_out.geometry.config.system.rotation_axis.position[0],places=2)     

    def test_CenterOfRotationCorrector(self):       
        corr = CentreOfRotationCorrector.xcorrelation(slice_index='centre', projection_index=0, ang_tol=0.1)
        corr.set_input(self.data_DLS.clone())
        ad_out = corr.get_output()
        self.assertAlmostEqual(6.33, ad_out.geometry.config.system.rotation_axis.position[0],places=2)     
        
        corr = CentreOfRotationCorrector.xcorrelation(slice_index=67, projection_index=0, ang_tol=0.1)
        corr.set_input(self.data_DLS.clone())
        ad_out = corr.get_output()
        self.assertAlmostEqual(6.33, ad_out.geometry.config.system.rotation_axis.position[0],places=2)              


class TestCentreOfRotation_conebeam(unittest.TestCase):

    def setUp(self):
        angles = numpy.linspace(0, 360, 180, endpoint=False)

        ag_orig = AcquisitionGeometry.create_Cone2D([0,-100],[0,100])\
            .set_panel(64,0.2)\
            .set_angles(angles)\
            .set_labels(['angle', 'horizontal'])

        ig = ag_orig.get_ImageGeometry()
        phantom = TomoPhantom.get_ImageData(12, ig)

        Op = ProjectionOperator(ig, ag_orig, direct_method='Siddon')
        self.data_0 = Op.direct(phantom)

        ag_offset = AcquisitionGeometry.create_Cone2D([0,-100],[0,100],rotation_axis_position=(-0.150,0))\
            .set_panel(64,0.2)\
            .set_angles(angles)\
            .set_labels(['angle', 'horizontal'])

        Op = ProjectionOperator(ig, ag_offset, direct_method='Siddon')
        self.data_offset = Op.direct(phantom)
        self.data_offset.geometry = ag_orig

    @unittest.skipUnless(has_tomophantom and has_astra, "Tomophantom or ASTRA not installed")
    def test_CofR_image_sharpness_astra(self):
        corr = CofR_image_sharpness(FBP=AstraFBP)
        ad_out = corr(self.data_0)
        self.assertAlmostEqual(0.000, ad_out.geometry.config.system.rotation_axis.position[0],places=3)     

        corr = CofR_image_sharpness(FBP=AstraFBP)
        ad_out = corr(self.data_offset)
        self.assertAlmostEqual(-0.150, ad_out.geometry.config.system.rotation_axis.position[0],places=3)     

    @unittest.skipUnless(has_tomophantom and has_tigre, "Tomophantom or TIGRE not installed")
    def test_CofR_image_sharpness_tigre(self): #currently not avaliable for parallel beam
        corr = CofR_image_sharpness(FBP=TigreFBP)
        ad_out = corr(self.data_0)
        self.assertAlmostEqual(0.000, ad_out.geometry.config.system.rotation_axis.position[0],places=3)     

        corr = CofR_image_sharpness(FBP=TigreFBP)
        ad_out = corr(self.data_offset)
        self.assertAlmostEqual(-0.150, ad_out.geometry.config.system.rotation_axis.position[0],places=3)     

class TestDataProcessor(unittest.TestCase):
   
    def test_DataProcessorBasic(self):

        dc_in = DataContainer(numpy.arange(10), True)
        dc_out = dc_in.copy()

        ax = AX()
        ax.scalar = 2
        ax.set_input(dc_in)

        #check results with out
        out_gold = dc_in*2
        ax.get_output(out=dc_out)
        numpy.testing.assert_array_equal(dc_out.as_array(), out_gold.as_array())

        #check results with return
        dc_out2 = ax.get_output()
        numpy.testing.assert_array_equal(dc_out2.as_array(), out_gold.as_array())

        #check call method
        dc_out2 = ax(dc_in)
        numpy.testing.assert_array_equal(dc_out2.as_array(), out_gold.as_array())

        #check storage mode
        self.assertFalse(ax.store_output)
        self.assertTrue(ax.output == None)
        ax.store_output = True
        self.assertTrue(ax.store_output)

        #check storing a copy and not a reference
        ax.set_input(dc_in)
        dc_out = ax.get_output()
        numpy.testing.assert_array_equal(ax.output.as_array(), out_gold.as_array())
        self.assertFalse(id(ax.output.as_array()) == id(dc_out.as_array()))

        #check recalculation on argument change
        ax.scalar = 3
        out_gold = dc_in*3
        ax.get_output(out=dc_out)
        numpy.testing.assert_array_equal(dc_out.as_array(), out_gold.as_array())

        #check recalculation on input change
        dc_in2 = dc_in.copy()
        dc_in2 *=2
        out_gold = dc_in2*3
        ax.set_input(dc_in2)
        ax.get_output(out=dc_out)
        numpy.testing.assert_array_equal(dc_out.as_array(), out_gold.as_array())

        #check recalculation on input modified (won't pass)
        dc_in2 *= 2
        out_gold = dc_in2*3
        ax.get_output(out=dc_out)
        #numpy.testing.assert_array_equal(dc_out.as_array(), out_gold.as_array())


    def test_DataProcessorChaining(self):
        shape = (2,3,4,5)
        size = shape[0]
        for i in range(1, len(shape)):
            size = size * shape[i]
        #print("a refcount " , sys.getrefcount(a))
        a = numpy.asarray([i for i in range( size )])
        a = numpy.reshape(a, shape)
        ds = DataContainer(a, False, ['X', 'Y','Z' ,'W'])
        c = ds.subset(Y=0)
        c = c.subset(['Z','W','X'])
        arr = c.as_array()
        #[ 0 60  1 61  2 62  3 63  4 64  5 65  6 66  7 67  8 68  9 69 10 70 11 71
        # 12 72 13 73 14 74 15 75 16 76 17 77 18 78 19 79]
        
        #print(arr)
    
        ax = AX()
        ax.scalar = 2

        numpy.testing.assert_array_equal(ax(c).as_array(), arr*2)

        ax.set_input(c)
        numpy.testing.assert_array_equal(ax.get_output().as_array(), arr*2)

        cast = CastDataContainer(dtype=numpy.float32)
        cast.set_input(c)
        out = cast.get_output()
        self.assertTrue(out.as_array().dtype == numpy.float32)
        out *= 0 
        axm = AX()
        axm.scalar = 0.5
        axm.set_input(c)
        axm.get_output(out)
        numpy.testing.assert_array_equal(out.as_array(), arr*0.5)
        
        #print("check call method of DataProcessor")
        numpy.testing.assert_array_equal(axm(c).as_array(), arr*0.5)        
    
        
        # check out in DataSetProcessor
        #a = numpy.asarray([i for i in range( size )])
           
        # create a PixelByPixelDataProcessor
        
        #define a python function which will take only one input (the pixel value)
        pyfunc = lambda x: -x if x > 20 else x
        clip = PixelByPixelDataProcessor()
        clip.pyfunc = pyfunc 
        clip.set_input(c)    
        #clip.apply()
        v = clip.get_output().as_array()
        
        self.assertTrue(v.max() == 19)
        self.assertTrue(v.min() == -79)
        
        #print ("clip in {0} out {1}".format(c.as_array(), clip.get_output().as_array()))
        
        #dsp = DataProcessor()
        #dsp.set_input(ds)
        #dsp.input = a
        # pipeline
    
        chain = AX()
        chain.scalar = 0.5
        chain.set_input_processor(ax)
        #print ("chain in {0} out {1}".format(ax.get_output().as_array(), chain.get_output().as_array()))
        numpy.testing.assert_array_equal(chain.get_output().as_array(), arr)
        
        #print("check call method of DataProcessor")
        numpy.testing.assert_array_equal(ax(chain(c)).as_array(), arr)        

class TestMaskGenerator(unittest.TestCase):       

    def test_MaskGenerator(self): 
    
        IG = ImageGeometry(voxel_num_x=10,
                        voxel_num_y=10)
        
        data = IG.allocate('random')
        
        data.as_array()[2,3] = float('inf')
        data.as_array()[4,5] = float('nan')
        
        # check special values - default
        m = MaskGenerator.special_values()
        m.set_input(data)
        mask = m.process()
        
        mask_manual = numpy.ones((10,10), dtype=numpy.bool)
        mask_manual[2,3] = 0
        mask_manual[4,5] = 0
        
        numpy.testing.assert_array_equal(mask.as_array(), mask_manual)
        
        # check nan
        m = MaskGenerator.special_values(inf=False)
        m.set_input(data)
        mask = m.process()
        
        mask_manual = numpy.ones((10,10), dtype=numpy.bool)
        mask_manual[4,5] = 0
        
        numpy.testing.assert_array_equal(mask.as_array(), mask_manual)
        
        # check inf
        m = MaskGenerator.special_values(nan=False)
        m.set_input(data)
        mask = m.process()
        
        mask_manual = numpy.ones((10,10), dtype=numpy.bool)
        mask_manual[2,3] = 0
        
        numpy.testing.assert_array_equal(mask.as_array(), mask_manual)
        
        # check threshold
        data = IG.allocate('random')
        data.as_array()[6,8] = 100
        data.as_array()[1,3] = 80
        
        m = MaskGenerator.threshold(None, 70)
        m.set_input(data)
        mask = m.process()
        
        mask_manual = numpy.ones((10,10), dtype=numpy.bool)
        mask_manual[6,8] = 0
        mask_manual[1,3] = 0
        
        numpy.testing.assert_array_equal(mask.as_array(), mask_manual)
        
        m = MaskGenerator.threshold(None, 80)
        m.set_input(data)
        mask = m.process()
        
        mask_manual = numpy.ones((10,10), dtype=numpy.bool)
        mask_manual[6,8] = 0
        
        numpy.testing.assert_array_equal(mask.as_array(), mask_manual)
        
        # check quantile
        data = IG.allocate('random')
        data.as_array()[6,8] = 100
        data.as_array()[1,3] = 80
        
        m = MaskGenerator.quantile(None, 0.98)
        m.set_input(data)
        mask = m.process()
        
        mask_manual = numpy.ones((10,10), dtype=numpy.bool)
        mask_manual[6,8] = 0
        mask_manual[1,3] = 0
        
        numpy.testing.assert_array_equal(mask.as_array(), mask_manual)
        
        m = MaskGenerator.quantile(None, 0.99)
        m.set_input(data)
        mask = m.process()
        
        mask_manual = numpy.ones((10,10), dtype=numpy.bool)
        mask_manual[6,8] = 0
        
        numpy.testing.assert_array_equal(mask.as_array(), mask_manual)
        
        # check mean
        IG = ImageGeometry(voxel_num_x=200,
                            voxel_num_y=200)
        #data = IG.allocate('random', seed=10)
        data = IG.allocate()
        numpy.random.seed(10)
        data.fill(numpy.random.rand(200,200))
        data.as_array()[7,4] += 10 * numpy.std(data.as_array()[7,:])
        
        m = MaskGenerator.mean(axis='horizontal_x')
        m.set_input(data)
        mask = m.process()
        
        mask_manual = numpy.ones((200,200), dtype=numpy.bool)
        mask_manual[7,4] = 0
        
        numpy.testing.assert_array_equal(mask.as_array(), mask_manual)
        
        m = MaskGenerator.mean(window=5)
        m.set_input(data)
        mask = m.process()
        
        mask_manual = numpy.ones((200,200), dtype=numpy.bool)
        mask_manual[7,4] = 0
        
        numpy.testing.assert_array_equal(mask.as_array(), mask_manual)
        
        # check median
        m = MaskGenerator.median(axis='horizontal_x')
        m.set_input(data)
        mask = m.process()
        
        mask_manual = numpy.ones((200,200), dtype=numpy.bool)
        mask_manual[7,4] = 0
        
        numpy.testing.assert_array_equal(mask.as_array(), mask_manual)
        
        m = MaskGenerator.median()
        m.set_input(data)
        mask = m.process()
        
        mask_manual = numpy.ones((200,200), dtype=numpy.bool)
        mask_manual[7,4] = 0
        numpy.testing.assert_array_equal(mask.as_array(), mask_manual)
        
        # check movmean
        m = MaskGenerator.mean(window=10)
        m.set_input(data)
        mask = m.process()
        
        mask_manual = numpy.ones((200,200), dtype=numpy.bool)
        mask_manual[7,4] = 0
        numpy.testing.assert_array_equal(mask.as_array(), mask_manual)
        
        #
        m = MaskGenerator.mean(window=20, axis='horizontal_y')
        m.set_input(data)
        mask = m.process()
        
        mask_manual = numpy.ones((200,200), dtype=numpy.bool)
        mask_manual[7,4] = 0
        numpy.testing.assert_array_equal(mask.as_array(), mask_manual)
        
        m = MaskGenerator.mean(window=10, threshold_factor=10)
        m.set_input(data)
        mask = m.process()
        
        mask_manual = numpy.ones((200,200), dtype=numpy.bool)
        numpy.testing.assert_array_equal(mask.as_array(), mask_manual)
        
        # check movmedian
        m = MaskGenerator.median(window=20)
        m.set_input(data)
        mask = m.process()
        
        mask_manual = numpy.ones((200,200), dtype=numpy.bool)
        mask_manual[7,4] = 0
        numpy.testing.assert_array_equal(mask.as_array(), mask_manual)
        
        # check movmedian
        m = MaskGenerator.median(window=40)
        m.set_input(data)
        mask = m.process()
        
        mask_manual = numpy.ones((200,200), dtype=numpy.bool)
        mask_manual[7,4] = 0
        numpy.testing.assert_array_equal(mask.as_array(), mask_manual)

class TestTransmissionAbsorptionConverter(unittest.TestCase):

    def test_TransmissionAbsorptionConverter(self):
            
        ray_direction = [0.1, 3.0, 0.4]
        detector_position = [-1.3, 1000.0, 2]
        detector_direction_row = [1.0, 0.2, 0.0]
        detector_direction_col = [0.0 ,0.0, 1.0]
        rotation_axis_position = [0.1, 2.0, 0.5]
        rotation_axis_direction = [0.1, 2.0, 0.5]
        
        AG = AcquisitionGeometry.create_Parallel3D(ray_direction=ray_direction, 
                                                    detector_position=detector_position, 
                                                    detector_direction_x=detector_direction_row, 
                                                    detector_direction_y=detector_direction_col,
                                                    rotation_axis_position=rotation_axis_position,
                                                    rotation_axis_direction=rotation_axis_direction)
        
        angles = numpy.linspace(0, 360, 10, dtype=numpy.float32)
        
        AG.set_channels(num_channels=10)
        AG.set_angles(angles, initial_angle=10, angle_unit='radian')
        AG.set_panel((10, 5), pixel_size=(0.1, 0.2))
        AG.dimension_labels = ['vertical',\
                                'horizontal',\
                                'angle',\
                                'channel']
        
        ad = AG.allocate('random')
        
        s = TransmissionAbsorptionConverter(white_level=10, min_intensity=0.1)
        s.set_input(ad)
        data_exp = s.get_output()
        
        data_new = ad.as_array().copy()
        data_new /= 10
        data_new[data_new < 0.1] = 0.1
        data_new = -1 * numpy.log(data_new)
        
        self.assertTrue(data_exp.geometry == AG)
        numpy.testing.assert_allclose(data_exp.as_array(), data_new, rtol=1E-6)
        
        data_exp.fill(0)
        s.process(out=data_exp)
        
        self.assertTrue(data_exp.geometry == AG)
        numpy.testing.assert_allclose(data_exp.as_array(), data_new, rtol=1E-6)

class TestAbsorptionTransmissionConverter(unittest.TestCase):

    def test_AbsorptionTransmissionConverter(self):

        ray_direction = [0.1, 3.0, 0.4]
        detector_position = [-1.3, 1000.0, 2]
        detector_direction_row = [1.0, 0.2, 0.0]
        detector_direction_col = [0.0 ,0.0, 1.0]
        rotation_axis_position = [0.1, 2.0, 0.5]
        rotation_axis_direction = [0.1, 2.0, 0.5]
        
        AG = AcquisitionGeometry.create_Parallel3D(ray_direction=ray_direction, 
                                                    detector_position=detector_position, 
                                                    detector_direction_x=detector_direction_row, 
                                                    detector_direction_y=detector_direction_col,
                                                    rotation_axis_position=rotation_axis_position,
                                                    rotation_axis_direction=rotation_axis_direction)
        
        angles = numpy.linspace(0, 360, 10, dtype=numpy.float32)
        
        AG.set_channels(num_channels=10)
        AG.set_angles(angles, initial_angle=10, angle_unit='radian')
        AG.set_panel((10, 5), pixel_size=(0.1, 0.2))
        AG.dimension_labels = ['vertical',\
                                'horizontal',\
                                'angle',\
                                'channel']
        
        ad = AG.allocate('random')
        
        s = AbsorptionTransmissionConverter(white_level=10)
        s.set_input(ad)
        data_exp = s.get_output()
        
        self.assertTrue(data_exp.geometry == AG)
        numpy.testing.assert_allclose(data_exp.as_array(), numpy.exp(-ad.as_array())*10, rtol=1E-6)
        
        data_exp.fill(0)
        s.process(out=data_exp)
        
        self.assertTrue(data_exp.geometry == AG)
        numpy.testing.assert_allclose(data_exp.as_array(), numpy.exp(-ad.as_array())*10, rtol=1E-6)  


class TestMasker(unittest.TestCase):       

    def setUp(self):
        IG = ImageGeometry(voxel_num_x=10,
                        voxel_num_y=10)
        
        self.data_init = IG.allocate('random')
        
        self.data = self.data_init.copy()
        
        self.data.as_array()[2,3] = float('inf')
        self.data.as_array()[4,5] = float('nan')
        
        mask_manual = numpy.ones((10,10), dtype=numpy.bool)
        mask_manual[2,3] = 0
        mask_manual[4,5] = 0
        
        self.mask_manual = DataContainer(mask_manual, dimension_labels=self.data.dimension_labels) 
        self.mask_generated = MaskGenerator.special_values()(self.data)

    def test_Masker_Manual(self):
        self.Masker_check(self.mask_manual, self.data, self.data_init)

    def test_Masker_generated(self):
        self.Masker_check(self.mask_generated, self.data, self.data_init)

    def Masker_check(self, mask, data, data_init): 

        # test vaue mode
        m = Masker.value(mask=mask, value=10)
        m.set_input(data)
        res = m.process()
        
        data_test = data.copy().as_array()
        data_test[2,3] = 10
        data_test[4,5] = 10
        
        numpy.testing.assert_allclose(res.as_array(), data_test, rtol=1E-6)     
        
        # test mean mode
        m = Masker.mean(mask=mask)
        m.set_input(data)
        res = m.process()
        
        data_test = data.copy().as_array()
        tmp = numpy.sum(data_init.as_array())-(data_init.as_array()[2,3]+data_init.as_array()[4,5])
        tmp /= 98
        data_test[2,3] = tmp
        data_test[4,5] = tmp
        
        numpy.testing.assert_allclose(res.as_array(), data_test, rtol=1E-6)  
        
        # test median mode
        m = Masker.median(mask=mask)
        m.set_input(data)
        res = m.process()
        
        data_test = data.copy().as_array()
        tmp = data.as_array()[numpy.isfinite(data.as_array())]
        data_test[2,3] = numpy.median(tmp)
        data_test[4,5] = numpy.median(tmp)
        
        numpy.testing.assert_allclose(res.as_array(), data_test, rtol=1E-6)
        
        # test axis int
        m = Masker.median(mask=mask, axis=0)
        m.set_input(data)
        res = m.process()
        
        data_test = data.copy().as_array()
        tmp1 = data.as_array()[2,:][numpy.isfinite(data.as_array()[2,:])]
        tmp2 = data.as_array()[4,:][numpy.isfinite(data.as_array()[4,:])]
        data_test[2,3] = numpy.median(tmp1)
        data_test[4,5] = numpy.median(tmp2)
        
        numpy.testing.assert_allclose(res.as_array(), data_test, rtol=1E-6) 
        
        # test axis str
        m = Masker.mean(mask=mask, axis=data.dimension_labels[1])
        m.set_input(data)
        res = m.process()
        
        data_test = data.copy().as_array()
        tmp1 = data.as_array()[:,3][numpy.isfinite(data.as_array()[:,3])]
        tmp2 = data.as_array()[:,5][numpy.isfinite(data.as_array()[:,5])]
        data_test[2,3] = numpy.sum(tmp1) / 9
        data_test[4,5] = numpy.sum(tmp2) / 9
        
        numpy.testing.assert_allclose(res.as_array(), data_test, rtol=1E-6)
        
        # test inline
        data = data_init.copy()
        m = Masker.value(mask=mask, value=10)
        m.set_input(data)
        m.process(out=data)
        
        data_test = data_init.copy().as_array()
        data_test[2,3] = 10
        data_test[4,5] = 10
        
        numpy.testing.assert_allclose(data.as_array(), data_test, rtol=1E-6) 
        
        # test mask numpy 
        data = data_init.copy()
        m = Masker.value(mask=mask.as_array(), value=10)
        m.set_input(data)
        res = m.process()
        
        data_test = data.copy().as_array()
        data_test[2,3] = 10
        data_test[4,5] = 10
        
        numpy.testing.assert_allclose(res.as_array(), data_test, rtol=1E-6)  
        
        # test interpolate
        data = data_init.copy()
        m = Masker.interpolate(mask=mask, method='linear', axis='horizontal_y')
        m.set_input(data)
        res = m.process()
        
        data_test = data.copy().as_array()
        data_test[2,3] = (data_test[1,3] + data_test[3,3]) / 2
        data_test[4,5] = (data_test[3,5] + data_test[5,5]) / 2
        
        numpy.testing.assert_allclose(res.as_array(), data_test, rtol=1E-6)  
        

        
if __name__ == "__main__":
    
    d = TestDataProcessor()
    d.test_DataProcessorChaining()

