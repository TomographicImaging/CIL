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

from ccpi.utilities.dataexample import data_dir
from timeit import default_timer as timer

from ccpi.framework import AX, CastDataContainer, PixelByPixelDataProcessor

from ccpi.io import NEXUSDataReader
from ccpi.processors import CenterOfRotationFinder
from ccpi.processors import CentreOfRotationCorrector, CofR_xcorr
import wget
import os

class TestDataProcessor(unittest.TestCase):
    
    def setUp(self):

        path = os.path.join(data_dir,'24737_fd_normalised.nxs')
        myreader = NEXUSDataReader(nexus_file=path)
        data_raw = myreader.load_data()

        self.data_DLS = data_raw.log()
        self.data_DLS *= -1

    def test_CenterOfRotation(self):

        ad = self.data_DLS.clone()
        print (ad)
        cf = CenterOfRotationFinder()
        cf.set_input(ad)
        print ("Center of rotation", cf.get_output())
        self.assertAlmostEqual(86.25, cf.get_output())
        
        print("check call method of DataProcessor")
        self.assertAlmostEqual(86.25, cf(ad))

        ad = self.data_DLS.clone()
        ad = ad.subset(['vertical','angle','horizontal'])
        print (ad)
        cf = CenterOfRotationFinder()
        cf.set_input(ad)
        print ("Center of rotation", cf.get_output())
        self.assertAlmostEqual(86.25, cf.get_output())
        
        print("check call method of DataProcessor")
        self.assertAlmostEqual(86.25, cf(ad))

        ad = self.data_DLS.clone()
        ad = ad.subset(vertical=67)
        print (ad)
        cf = CenterOfRotationFinder()
        cf.set_input(ad)
        print ("Center of rotation", cf.get_output())
        self.assertAlmostEqual(86.25, cf.get_output())
        print("check call method of DataProcessor")
        self.assertAlmostEqual(86.25, cf(ad))        

        ad = self.data_DLS.clone()
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