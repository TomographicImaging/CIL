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
from ccpi.framework import DataProcessor
from ccpi.framework import DataContainer
from ccpi.framework import ImageData
from ccpi.framework import AcquisitionData
from ccpi.framework import ImageGeometry
from ccpi.framework import AcquisitionGeometry
from timeit import default_timer as timer

from ccpi.framework import AX, CastDataContainer, PixelByPixelDataProcessor

from ccpi.io.reader import NexusReader
from ccpi.processors import CenterOfRotationFinder, Binner, Slicer, Padder
import wget
import os

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        wget.download('https://github.com/DiamondLightSource/Savu/raw/master/test_data/data/24737_fd.nxs')
        self.filename = '24737_fd.nxs'

    def tearDown(self):
        os.remove(self.filename)

    def test_CenterOfRotation(self):
        reader = NexusReader(self.filename)
        data = reader.get_acquisition_data_whole()

        ad = data.clone()
        print (ad)
        cf = CenterOfRotationFinder()
        cf.set_input(ad)
        print ("Center of rotation", cf.get_output())
        self.assertAlmostEqual(86.25, cf.get_output())

        ad = data.clone()
        ad = ad.subset(['vertical','angle','horizontal'])
        print (ad)
        cf = CenterOfRotationFinder()
        cf.set_input(ad)
        print ("Center of rotation", cf.get_output())
        self.assertAlmostEqual(86.25, cf.get_output())

        ad = data.clone()
        ad = ad.subset(vertical=67)
        print (ad)
        cf = CenterOfRotationFinder()
        cf.set_input(ad)
        print ("Center of rotation", cf.get_output())
        self.assertAlmostEqual(86.25, cf.get_output())

        ad = data.clone()
        print (ad)
        cf = CenterOfRotationFinder()
        cf.set_input(ad)
        cf.set_slice(80)
        print ("Center of rotation", cf.get_output())
        self.assertAlmostEqual(86.25, cf.get_output())
        cf.set_slice()
        print ("Center of rotation", cf.get_output())
        self.assertAlmostEqual(86.25, cf.get_output())       
        cf.set_slice('centre')
        print ("Center of rotation", cf.get_output())
        self.assertAlmostEqual(86.25, cf.get_output())
    
    def test_Binner(self):
        reader = NexusReader(self.filename)
        data = reader.get_acquisition_data_whole()
        ad = data.clone()
        print(ad.geometry)
        
        resizer = Binner(roi = {'vertical': (10,124),
                                 'horizontal': (None,None,2)})
        resizer.input = data
        data_resized = resizer.process()
        
        print(data_resized.geometry)
        
        self.assertTrue(80 == data_resized.geometry.pixel_num_h)
        self.assertTrue(114 == data_resized.geometry.pixel_num_v)
        self.assertTrue(data.geometry.pixel_size_v == data_resized.geometry.pixel_size_v)
        self.assertTrue(data.geometry.pixel_size_h * 2 == data_resized.geometry.pixel_size_h)
        
        resizer = Binner(roi = {'horizontal': (10,20), 
                                 'vertical': (None,None,5)})
        resizer.input = data
        data_resized = resizer.process()
        
        print(data_resized.geometry)
        
        self.assertTrue(10 == data_resized.geometry.pixel_num_h)
        self.assertTrue(26 == data_resized.geometry.pixel_num_v)
        self.assertTrue(data.geometry.pixel_size_v * 5 == data_resized.geometry.pixel_size_v)
        self.assertTrue(data.geometry.pixel_size_h == data_resized.geometry.pixel_size_h)
        
        resizer = Binner(roi = {'horizontal': (10,100,4), 
                                 'vertical': (10,100,5)})
        resizer.input = data
        data_resized = resizer.process()
        
        print(data_resized.geometry)
        
        self.assertTrue(22 == data_resized.geometry.pixel_num_h)
        self.assertTrue(18 == data_resized.geometry.pixel_num_v)
        self.assertTrue(5 == data_resized.geometry.pixel_size_v)
        self.assertTrue(4 == data_resized.geometry.pixel_size_h)
        
        ig = ImageGeometry(voxel_num_x=40,
                           voxel_num_y=50,
                           voxel_num_z=60,
                           voxel_size_x=1, 
                           voxel_size_y=2,
                           voxel_size_z=3,
                           channels=10)
        
        image = ig.allocate()
        
        resizer = Binner(roi = {'channel': (None, None, 2), 
                                 'horizontal_x': (20, None),
                                 'horizontal_y': (-40,-10),
                                 'vertical': (None,None,4)})
        resizer.input = image
        image_resized = resizer.process()
        
        print(image_resized.geometry)
        
        self.assertTrue(5 == image_resized.geometry.channels)
        self.assertTrue(20 == image_resized.geometry.voxel_num_x)
        self.assertTrue(30 == image_resized.geometry.voxel_num_y)
        self.assertTrue(15 == image_resized.geometry.voxel_num_z)
        self.assertTrue(image.geometry.voxel_size_x == image_resized.geometry.voxel_size_x)
        self.assertTrue(image.geometry.voxel_size_y == image_resized.geometry.voxel_size_y)
        self.assertTrue(image.geometry.voxel_size_z * 4 == image_resized.geometry.voxel_size_z)
    
    def test_Slicer(self):
        
        reader = NexusReader(self.filename)
        data = reader.get_acquisition_data_whole()
        ad = data.clone()
        print(ad.geometry)
        
        resizer = Slicer(roi = {'vertical': (10,124),
                                 'horizontal': (None,None,2)})
        resizer.input = data
        data_resized = resizer.process()
        
        print(data_resized.geometry)
        
        self.assertTrue(80 == data_resized.geometry.pixel_num_h)
        self.assertTrue(114 == data_resized.geometry.pixel_num_v)
        self.assertTrue(data.geometry.pixel_size_v == data_resized.geometry.pixel_size_v)
        self.assertTrue(data.geometry.pixel_size_h == data_resized.geometry.pixel_size_h)
        
        resizer = Slicer(roi = {'horizontal': (10,20), 
                                 'vertical': (None,None,5)})
        resizer.input = data
        data_resized = resizer.process()
        
        print(data_resized.geometry)
        
        self.assertTrue(10 == data_resized.geometry.pixel_num_h)
        self.assertTrue(27 == data_resized.geometry.pixel_num_v)
        self.assertTrue(data.geometry.pixel_size_v == data_resized.geometry.pixel_size_v)
        self.assertTrue(data.geometry.pixel_size_h == data_resized.geometry.pixel_size_h)
        
        resizer = Slicer(roi = {'horizontal': (10,100,4), 
                                 'vertical': (10,100,5)})
        resizer.input = data
        data_resized = resizer.process()
        
        self.assertTrue(data_resized.geometry)
        
        self.assertTrue(23 == data_resized.geometry.pixel_num_h)
        self.assertTrue(18 == data_resized.geometry.pixel_num_v)
        self.assertTrue(data.geometry.pixel_size_v == data_resized.geometry.pixel_size_v)
        self.assertTrue(data.geometry.pixel_size_h == data_resized.geometry.pixel_size_h)
        
        ig = ImageGeometry(voxel_num_x=40,
                           voxel_num_y=50,
                           voxel_num_z=60,
                           voxel_size_x=1, 
                           voxel_size_y=2,
                           voxel_size_z=3,
                           channels=10)
        
        image = ig.allocate()
        
        resizer = Slicer(roi = {'channel': (None, None, 2), 
                                 'horizontal_x': (20, None),
                                 'horizontal_y': (-40,-10),
                                 'vertical': (None,None,4)})
        resizer.input = image
        image_resized = resizer.process()
        
        print(image_resized.geometry)
        
        self.assertTrue(5 == image_resized.geometry.channels)
        self.assertTrue(20 == image_resized.geometry.voxel_num_x)
        self.assertTrue(30 == image_resized.geometry.voxel_num_y)
        self.assertTrue(15 == image_resized.geometry.voxel_num_z)
        self.assertTrue(image.geometry.voxel_size_x == image_resized.geometry.voxel_size_x)
        self.assertTrue(image.geometry.voxel_size_y == image_resized.geometry.voxel_size_y)
        self.assertTrue(image.geometry.voxel_size_z == image_resized.geometry.voxel_size_z)
    
    def test_Padder(self):
        
        reader = NexusReader(self.filename)
        data = reader.get_acquisition_data_whole()
        ad = data.clone()
        print(ad.geometry)
        
        padder = Padder(pad_width = 10,
                        mode = 'constant',
                        constant_values = {'vertical': (0,100), 'horizontal': 1000})
        
        padder.input = data
        data_resized = padder.process()
        
        numpy.testing.assert_array_equal(data_resized.as_array(), numpy.pad(data.as_array(), 10, mode='constant', constant_values=((0,0), (0,100), (1000,1000))))
        self.assertTrue(data.geometry.pixel_num_h+20 == data_resized.geometry.pixel_num_h)
        self.assertTrue(data.geometry.pixel_num_v+20 == data_resized.geometry.pixel_num_v)
        self.assertTrue(data.geometry.pixel_size_v == data_resized.geometry.pixel_size_v)
        self.assertTrue(data.geometry.pixel_size_h == data_resized.geometry.pixel_size_h)
        
        padder = Padder(pad_width = {'vertical': 10, 'horizontal': 20, 'angle': 30},
                        mode = 'reflect',
                        reflect_type = 'odd')
        
        padder.input = data
        data_resized = padder.process()
        
        numpy.testing.assert_array_equal(data_resized.as_array(), numpy.pad(data.as_array(), ((30,30), (10,10), (20,20)), mode ='reflect',reflect_type='odd'))
        self.assertTrue(data.geometry.pixel_num_h+40 == data_resized.geometry.pixel_num_h)
        self.assertTrue(data.geometry.pixel_num_v+20 == data_resized.geometry.pixel_num_v)
        self.assertTrue(data.geometry.pixel_size_v == data_resized.geometry.pixel_size_v)
        self.assertTrue(data.geometry.pixel_size_h == data_resized.geometry.pixel_size_h)
        
        ig = ImageGeometry(voxel_num_x=40,
                           voxel_num_y=50,
                           voxel_num_z=60,
                           voxel_size_x=1, 
                           voxel_size_y=2,
                           voxel_size_z=3,
                           channels=10)
        
        image = ig.allocate()
        
        resizer = Padder(pad_width = {'vertical': 1, 'horizontal_x': 2, 'channel': 3},
                         mode='linear_ramp',
                         end_values = {'vertical': 1, 'horizontal_x': 2, 'channel': 3})
        resizer.input = image
        image_resized = resizer.process()
        
        print(image_resized.geometry)
        
        numpy.testing.assert_array_equal(image_resized.as_array(), numpy.pad(image.as_array(), ((3,3),(1,1),(0,0),(2,2)), mode ='linear_ramp', end_values=((3,3),(1,1),(0,0),(2,2))))
        self.assertTrue(image.geometry.channels+6 == image_resized.geometry.channels)
        self.assertTrue(image.geometry.voxel_num_x+4 == image_resized.geometry.voxel_num_x)
        self.assertTrue(image.geometry.voxel_num_y == image_resized.geometry.voxel_num_y)
        self.assertTrue(image.geometry.voxel_num_z+2 == image_resized.geometry.voxel_num_z)
        self.assertTrue(image.geometry.voxel_size_x == image_resized.geometry.voxel_size_x)
        self.assertTrue(image.geometry.voxel_size_y == image_resized.geometry.voxel_size_y)
        self.assertTrue(image.geometry.voxel_size_z == image_resized.geometry.voxel_size_z)
    
        
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
    
        ax = AX()
        ax.scalar = 2
        ax.set_input(c)
        #ax.apply()
        print ("ax  in {0} out {1}".format(c.as_array().flatten(),
               ax.get_output().as_array().flatten()))
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