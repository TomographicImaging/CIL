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
from cil.processors import CentreOfRotationCorrector, CofR_xcorr
import wget
import os

class TestDataProcessor(unittest.TestCase):
    
    def setUp(self):

        data_raw = dataexample.SYNCHROTRON_PARALLEL_BEAM_DATA.get()

        self.data_DLS = data_raw.log()
        self.data_DLS *= -1
    
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
        
        s = TransmissionAbsorptionConverter(white_level=10, threshold=0.1, value=0.5)
        s.set_input(ad)
        data_exp = s.get_output()
        
        data_new = ad.as_array().copy()
        data_new /= 10
        data_new[data_new < 0.1] = 0.5
        data_new = -1 * numpy.log(data_new)
        
        self.assertTrue(data_exp.geometry == AG)
        numpy.testing.assert_allclose(data_exp.as_array(), data_new, rtol=1E-6)
        
        s.process(out=ad)
        
        self.assertTrue(ad.geometry == AG)
        numpy.testing.assert_allclose(data_exp.as_array(), ad.as_array(), rtol=1E-6)
        
    
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
        
        s.process(out=ad)
        
        self.assertTrue(ad.geometry == AG)
        numpy.testing.assert_allclose(data_exp.as_array(), ad.as_array(), rtol=1E-6)



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

    def test_Normaliser(self):
        pass         

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