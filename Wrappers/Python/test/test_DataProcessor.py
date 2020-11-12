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
from cil.processors import CentreOfRotationCorrector, CofR_xcorr, Slicer
import wget
import os

class TestDataProcessor(unittest.TestCase):
    
    def setUp(self):

        data_raw = dataexample.SYNCHROTRON_PARALLEL_BEAM_DATA.get()

        self.data_DLS = data_raw.log()
        self.data_DLS *= -1
    
    def test_Slicer(self):
        
        ray_direction = [0.1, 3.0]
        detector_position = [-1.3, 1000.0]
        detector_direction_row = [1.0, 0.2]
        rotation_axis_position = [0.1, 2.0]
        
        AG = AcquisitionGeometry.create_Parallel2D(ray_direction=ray_direction, 
                                                   detector_position=detector_position, 
                                                   detector_direction_row=detector_direction_row, 
                                                   rotation_axis_position=rotation_axis_position)
        
        ray_direction = numpy.asarray(ray_direction)
        detector_direction_row = numpy.asarray(detector_direction_row)
        
        ray_direction /= numpy.sqrt((ray_direction**2).sum())
        detector_direction_row /= numpy.sqrt((detector_direction_row**2).sum())
        
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
        
        self.assertEqual(data_sliced.geometry.config.channels.num_channels, numpy.arange(1, 10-2, 3).shape[0] )
        self.assertEqual(data_sliced.geometry.config.panel.num_pixels, [numpy.arange(10, 100-11, 7).shape[0], 1])
        self.assertEqual(data_sliced.geometry.config.panel.pixel_size, [0.1, 0.1])
        self.assertEqual(data_sliced.geometry.config.angles.initial_angle, 10)
        self.assertEqual(data_sliced.geometry.config.angles.angle_unit, 'radian')
        self.assertEqual(data_sliced.geometry.config.angles.num_positions, numpy.arange(2, 9, 2).shape[0])
        self.assertEqual(data_sliced.geometry.config.system.dimension, '2D')
        self.assertEqual(data_sliced.geometry.config.system.geometry, 'parallel')
        self.assertEqual(data_sliced.geometry.dimension_labels, data.geometry.dimension_labels)
        
        numpy.testing.assert_allclose(data_sliced.as_array(), numpy.squeeze(data.as_array()[1:-2:3, 2:9:2, 10:-11:7]), rtol=1E-6)
        numpy.testing.assert_allclose(data_sliced.geometry.config.angles.angle_data, angles[2:9:2], rtol=1E-6)
        numpy.testing.assert_allclose(data_sliced.geometry.config.system.ray.direction, ray_direction, rtol=1E-6)
        numpy.testing.assert_allclose(data_sliced.geometry.config.system.detector.position, detector_position, rtol=1E-6)
        numpy.testing.assert_allclose(data_sliced.geometry.config.system.detector.direction_row, detector_direction_row, rtol=1E-6)
        numpy.testing.assert_allclose(data_sliced.geometry.config.system.rotation_axis.position, rotation_axis_position, rtol=1E-6)
        
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
                                                   detector_direction_row=detector_direction_row, 
                                                   detector_direction_col=detector_direction_col,
                                                   rotation_axis_position=rotation_axis_position,
                                                   rotation_axis_direction=rotation_axis_direction)
        
        ray_direction = numpy.asarray(ray_direction)
        rotation_axis_direction = numpy.asarray(ray_direction)
        detector_direction_row = numpy.asarray(detector_direction_row)
        detector_direction_col = numpy.asarray(detector_direction_col)
        
        ray_direction /= numpy.sqrt((ray_direction**2).sum())
        rotation_axis_direction /= numpy.sqrt((rotation_axis_direction**2).sum())
        detector_direction_row /= numpy.sqrt((detector_direction_row**2).sum())
        detector_direction_col /= numpy.sqrt((detector_direction_col**2).sum())
        
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
                        'vertical': (10, 12, 2)})
        s.set_input(data)
        data_sliced = s.process()
        
        dimension_labels_sliced = list(data.geometry.dimension_labels)
        dimension_labels_sliced.remove('channel')
        dimension_labels_sliced.remove('vertical')
        
        self.assertEqual(data_sliced.geometry.config.channels.num_channels, numpy.arange(0, 1, 1).shape[0])
        self.assertEqual(data_sliced.geometry.config.panel.num_pixels, [numpy.arange(10, 100, 2).shape[0], numpy.arange(10, 12, 2).shape[0]])
        self.assertEqual(data_sliced.geometry.config.panel.pixel_size, [0.1, 0.1])
        self.assertEqual(data_sliced.geometry.config.angles.initial_angle, 10)
        self.assertEqual(data_sliced.geometry.config.angles.angle_unit, 'radian')
        self.assertEqual(data_sliced.geometry.config.angles.num_positions, angles.shape[0])
        self.assertEqual(data_sliced.geometry.config.system.dimension, '2D')
        self.assertEqual(data_sliced.geometry.config.system.geometry, 'parallel')
        self.assertEqual(list(data_sliced.geometry.dimension_labels), dimension_labels_sliced)
        
        ray_direction = ray_direction[:2]
        detector_direction_row = detector_direction_row[:2]
        
        ray_direction = numpy.asarray(ray_direction)
        detector_direction_row = numpy.asarray(detector_direction_row)
        
        ray_direction /= numpy.sqrt((ray_direction**2).sum())
        detector_direction_row /= numpy.sqrt((detector_direction_row**2).sum())
        
        numpy.testing.assert_allclose(data_sliced.as_array(), numpy.squeeze(data.as_array()[10:12:2, 10::2, :, :1]), rtol=1E-6)
        numpy.testing.assert_allclose(data_sliced.geometry.config.angles.angle_data, angles, rtol=1E-6)
        numpy.testing.assert_allclose(data_sliced.geometry.config.system.ray.direction, ray_direction, rtol=1E-6)
        numpy.testing.assert_allclose(data_sliced.geometry.config.system.detector.position, detector_position[:2], rtol=1E-6)
        numpy.testing.assert_allclose(data_sliced.geometry.config.system.detector.direction_row, detector_direction_row, rtol=1E-6)
        numpy.testing.assert_allclose(data_sliced.geometry.config.system.rotation_axis.position, rotation_axis_position[:2], rtol=1E-6)
        
        #%%
        #test cone 2D case
        
        source_position = [0.1, 3.0]
        detector_position = [-1.3, 1000.0]
        detector_direction_row = [1.0, 0.2]
        rotation_axis_position = [0.1, 2.0]
        
        AG = AcquisitionGeometry.create_Cone2D(source_position=source_position, 
                                               detector_position=detector_position, 
                                               detector_direction_row=detector_direction_row, 
                                               rotation_axis_position=rotation_axis_position)
        
        ray_direction = numpy.asarray(ray_direction)
        detector_direction_row = numpy.asarray(detector_direction_row)
        
        ray_direction /= numpy.sqrt((ray_direction**2).sum())
        detector_direction_row /= numpy.sqrt((detector_direction_row**2).sum())
        
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
        
        self.assertEqual(data_sliced.geometry.config.channels.num_channels, numpy.arange(1, 10, 4).shape[0] )
        self.assertEqual(data_sliced.geometry.config.panel.num_pixels, [numpy.arange(10, 100-10, 5).shape[0], 1])
        self.assertEqual(data_sliced.geometry.config.panel.pixel_size, [0.1, 0.1])
        self.assertEqual(data_sliced.geometry.config.angles.initial_angle, 10)
        self.assertEqual(data_sliced.geometry.config.angles.angle_unit, 'degree')
        self.assertEqual(data_sliced.geometry.config.angles.num_positions, numpy.arange(2, 9, 2).shape[0])
        self.assertEqual(data_sliced.geometry.config.system.dimension, '2D')
        self.assertEqual(data_sliced.geometry.config.system.geometry, 'cone')
        self.assertEqual(data_sliced.geometry.dimension_labels, data.geometry.dimension_labels)
        
        numpy.testing.assert_allclose(data_sliced.as_array(), numpy.squeeze(data.as_array()[1::4, 2:9:2, 10:-10:5]), rtol=1E-6)
        numpy.testing.assert_allclose(data_sliced.geometry.config.angles.angle_data, angles[2:9:2], rtol=1E-6)
        numpy.testing.assert_allclose(data_sliced.geometry.config.system.source.position, source_position, rtol=1E-6)
        numpy.testing.assert_allclose(data_sliced.geometry.config.system.detector.position, detector_position, rtol=1E-6)
        numpy.testing.assert_allclose(data_sliced.geometry.config.system.detector.direction_row, detector_direction_row, rtol=1E-6)
        numpy.testing.assert_allclose(data_sliced.geometry.config.system.rotation_axis.position, rotation_axis_position, rtol=1E-6)
        
        #%%
        #test cone 3D case
        
        source_position = [0.1, 3.0, 0.4]
        detector_position = [-1.3, 1000.0, 2]
        detector_direction_row = [1.0, 0.2, 0.0]
        detector_direction_col = [0.0 ,0.0, 1.0]
        rotation_axis_position = [0.1, 2.0, 0.5]
        rotation_axis_direction = [0.1, 2.0, 0.5]
        
        AG = AcquisitionGeometry.create_Cone3D(source_position=source_position, 
                                               detector_position=detector_position, 
                                               detector_direction_row=detector_direction_row, 
                                               detector_direction_col=detector_direction_col,
                                               rotation_axis_position=rotation_axis_position,
                                               rotation_axis_direction=rotation_axis_direction)
        
        ray_direction = numpy.asarray(ray_direction)
        rotation_axis_direction = numpy.asarray(rotation_axis_direction)
        detector_direction_row = numpy.asarray(detector_direction_row)
        detector_direction_col = numpy.asarray(detector_direction_col)
        
        ray_direction /= numpy.sqrt((ray_direction**2).sum())
        rotation_axis_direction /= numpy.sqrt((rotation_axis_direction**2).sum())
        detector_direction_row /= numpy.sqrt((detector_direction_row**2).sum())
        detector_direction_col /= numpy.sqrt((detector_direction_col**2).sum())
        
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
        
        self.assertEqual(data_sliced.geometry.config.channels.num_channels, numpy.arange(0, 1, 1).shape[0])
        self.assertEqual(data_sliced.geometry.config.panel.num_pixels, [numpy.arange(10, 100, 2).shape[0], numpy.arange(10, 50-10, 2).shape[0]])
        self.assertEqual(data_sliced.geometry.config.panel.pixel_size, [0.1, 0.2])
        self.assertEqual(data_sliced.geometry.config.angles.initial_angle, 10)
        self.assertEqual(data_sliced.geometry.config.angles.angle_unit, 'radian')
        self.assertEqual(data_sliced.geometry.config.angles.num_positions, angles.shape[0])
        self.assertEqual(data_sliced.geometry.config.system.dimension, '3D')
        self.assertEqual(data_sliced.geometry.config.system.geometry, 'cone')
        self.assertEqual(list(data_sliced.geometry.dimension_labels), dimension_labels_sliced)
        
        numpy.testing.assert_allclose(data_sliced.as_array(), numpy.squeeze(data.as_array()[10:-10:2, 10::2, :, :1]), rtol=1E-6)
        numpy.testing.assert_allclose(data_sliced.geometry.config.angles.angle_data, angles, rtol=1E-6)
        numpy.testing.assert_allclose(data_sliced.geometry.config.system.source.position, source_position, rtol=1E-6)
        numpy.testing.assert_allclose(data_sliced.geometry.config.system.detector.position, detector_position, rtol=1E-6)
        numpy.testing.assert_allclose(data_sliced.geometry.config.system.rotation_axis.position, rotation_axis_position, rtol=1E-6)
        numpy.testing.assert_allclose(data_sliced.geometry.config.system.rotation_axis.direction, rotation_axis_direction, rtol=1E-6)
        numpy.testing.assert_allclose(data_sliced.geometry.config.system.detector.direction_row, detector_direction_row, rtol=1E-6)
        numpy.testing.assert_allclose(data_sliced.geometry.config.system.detector.direction_col, detector_direction_col, rtol=1E-6)
        
        #%% test cone 3D - centre slice
        s = Slicer(roi={'channel': (None, 1),
                        'angle': -1,
                        'horizontal': (10, None, 2),
                        'vertical': (25, 26)})
        s.set_input(data)
        data_sliced = s.process()
        
        dimension_labels_sliced = list(data.geometry.dimension_labels)
        dimension_labels_sliced.remove('channel')
        dimension_labels_sliced.remove('vertical')
        
        self.assertEqual(data_sliced.geometry.config.channels.num_channels, numpy.arange(0, 1, 1).shape[0])
        self.assertEqual(data_sliced.geometry.config.panel.num_pixels, [numpy.arange(10, 100, 2).shape[0], numpy.arange(25,26).shape[0]])
        self.assertEqual(data_sliced.geometry.config.panel.pixel_size, [0.1, 0.1])
        self.assertEqual(data_sliced.geometry.config.angles.initial_angle, 10)
        self.assertEqual(data_sliced.geometry.config.angles.angle_unit, 'radian')
        self.assertEqual(data_sliced.geometry.config.angles.num_positions, angles.shape[0])
        self.assertEqual(data_sliced.geometry.config.system.dimension, '2D')
        self.assertEqual(data_sliced.geometry.config.system.geometry, 'cone')
        self.assertEqual(list(data_sliced.geometry.dimension_labels), dimension_labels_sliced)
        
        detector_direction_row = detector_direction_row[:2]
        detector_direction_row = numpy.asarray(detector_direction_row)
        detector_direction_row /= numpy.sqrt((detector_direction_row**2).sum())
        
        numpy.testing.assert_allclose(data_sliced.as_array(), numpy.squeeze(data.as_array()[25:26, 10::2, :, :1]), rtol=1E-6)
        numpy.testing.assert_allclose(data_sliced.geometry.config.angles.angle_data, angles, rtol=1E-6)
        numpy.testing.assert_allclose(data_sliced.geometry.config.system.source.position, source_position[:2], rtol=1E-6)
        numpy.testing.assert_allclose(data_sliced.geometry.config.system.detector.position, detector_position[:2], rtol=1E-6)
        numpy.testing.assert_allclose(data_sliced.geometry.config.system.detector.direction_row, detector_direction_row, rtol=1E-6)
        numpy.testing.assert_allclose(data_sliced.geometry.config.system.rotation_axis.position, rotation_axis_position[:2], rtol=1E-6)
        
        #%% test cone 3D single slice, not central
        s = Slicer(roi={'channel': (None, 1),
                        'angle': -1,
                        'horizontal': (10, None, 2),
                        'vertical': (1, 2)},
                    force=True)
        s.set_input(data)
        data_sliced = s.process()
        
        dimension_labels_sliced = {i:data.dimension_labels[i] for i in data.dimension_labels if data.dimension_labels[i]!='channel' or data.dimension_labels[i]!='vertical'}
        
        self.assertEqual(data_sliced.geometry, None)
        self.assertEqual(data_sliced.dimension_labels, {0: 'horizontal', 1: 'angle'})
        numpy.testing.assert_allclose(data_sliced.as_array(), numpy.squeeze(data.as_array()[1:2, 10::2, :, :1]), rtol=1E-6)
        
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
        
        self.assertEqual(data_sliced.geometry.voxel_num_x, IG.voxel_num_x)
        self.assertEqual(data_sliced.geometry.voxel_num_y, numpy.arange(10, 30, 2).shape[0])
        self.assertEqual(data_sliced.geometry.voxel_num_z, numpy.arange(5, 12, 3).shape[0])
        self.assertEqual(data_sliced.geometry.channels, numpy.arange(0, 10, 2).shape[0])
        self.assertEqual(data_sliced.geometry.voxel_size_x, IG.voxel_size_x)
        self.assertEqual(data_sliced.geometry.voxel_size_y, IG.voxel_size_y)
        self.assertEqual(data_sliced.geometry.voxel_size_z, IG.voxel_size_z)
        self.assertEqual(data_sliced.geometry.center_x, IG.center_x)
        self.assertEqual(data_sliced.geometry.center_y, IG.center_y)
        self.assertEqual(data_sliced.geometry.center_z, IG.center_z)
        self.assertEqual(data_sliced.dimension_labels, data.dimension_labels)
        
        numpy.testing.assert_allclose(data_sliced.as_array(), numpy.squeeze(data.as_array()[5:12:3, ::2, 10:30:2, :]), rtol=1E-6)


    def test_CofR_xcorr(self):       

        corr = CofR_xcorr(slice_index='centre', projection_index=0, ang_tol=0.1)
        corr.set_input(self.data_DLS.clone())
        ad_out = corr.get_output()
        self.assertAlmostEqual(6.33, ad_out.geometry.config.system.rotation_axis.position[0],places=2)     
        
        corr = CofR_xcorr(slice_index=67, projection_index=0, ang_tol=0.1)
        corr.set_input(self.data_DLS.clone())
        ad_out = corr.get_output()
        self.assertAlmostEqual(6.33, ad_out.geometry.config.system.rotation_axis.position[0],places=2)              

    def test_CenterOfRotationCorrector(self):       
        corr = CentreOfRotationCorrector.xcorr(slice_index='centre', projection_index=0, ang_tol=0.1)
        corr.set_input(self.data_DLS.clone())
        ad_out = corr.get_output()
        self.assertAlmostEqual(6.33, ad_out.geometry.config.system.rotation_axis.position[0],places=2)     
        
        corr = CentreOfRotationCorrector.xcorr(slice_index=67, projection_index=0, ang_tol=0.1)
        corr.set_input(self.data_DLS.clone())
        ad_out = corr.get_output()
        self.assertAlmostEqual(6.33, ad_out.geometry.config.system.rotation_axis.position[0],places=2)              

    def test_Normalizer(self):
        pass         
        
    def test_DataProcessorChaining(self):
        shape = (2,3,4,5)
        size = shape[0]
        for i in range(1, len(shape)):
            size = size * shape[i]
        #print("a refcount " , sys.getrefcount(a))
        a = numpy.asarray([i for i in range( size )])
        a = numpy.reshape(a, shape)
        ds = DataContainer(a, False, ['X', 'Y','Z' ,'W'])
        c = ds.subset(['Z','W','X'])
        arr = c.as_array()
        #[ 0 60  1 61  2 62  3 63  4 64  5 65  6 66  7 67  8 68  9 69 10 70 11 71
        # 12 72 13 73 14 74 15 75 16 76 17 77 18 78 19 79]
        
        print(arr)
    
        ax = AX()
        ax.scalar = 2
        ax.set_input(c)
        #ax.apply()
        print ("ax  in {0} out {1}".format(c.as_array().flatten(),
               ax.get_output().as_array().flatten()))
        
        numpy.testing.assert_array_equal(ax.get_output().as_array(), arr*2)
                
        
        print("check call method of DataProcessor")
        numpy.testing.assert_array_equal(ax(c).as_array(), arr*2)
        
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
        
        print("check call method of DataProcessor")
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
        
        print ("clip in {0} out {1}".format(c.as_array(), clip.get_output().as_array()))
        
        #dsp = DataProcessor()
        #dsp.set_input(ds)
        #dsp.input = a
        # pipeline
    
        chain = AX()
        chain.scalar = 0.5
        chain.set_input_processor(ax)
        print ("chain in {0} out {1}".format(ax.get_output().as_array(), chain.get_output().as_array()))
        numpy.testing.assert_array_equal(chain.get_output().as_array(), arr)
        
        print("check call method of DataProcessor")
        numpy.testing.assert_array_equal(ax(chain(c)).as_array(), arr)        

        
        
if __name__ == "__main__":
    
    d = TestDataProcessor()
    d.test_DataProcessorChaining()