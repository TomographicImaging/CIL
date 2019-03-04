# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library developed by
#   Visual Analytics and Imaging System Group of the Science Technology
#   Facilities Council, STFC

#   Copyright 2018 Jakob Jorgensen, Daniil Kazantsev, Edoardo Pasca and Srikanth Nagella

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

'''
Unit tests for Readers

@author: Mr. Srikanth Nagella
'''
import unittest

import numpy.testing
import wget
import os
from ccpi.io.reader import NexusReader
from ccpi.io.reader import XTEKReader
#@unittest.skip
class TestNexusReader(unittest.TestCase):

    def setUp(self):
        wget.download('https://github.com/DiamondLightSource/Savu/raw/master/test_data/data/24737_fd.nxs')
        self.filename = '24737_fd.nxs'

    def tearDown(self):
        os.remove(self.filename)


    def testGetDimensions(self):
        nr = NexusReader(self.filename)
        self.assertEqual(nr.getSinogramDimensions(), (135, 91, 160), "Sinogram dimensions are not correct")
        
    def testGetProjectionDimensions(self):
        nr = NexusReader(self.filename)
        self.assertEqual(nr.getProjectionDimensions(), (91,135,160), "Projection dimensions are not correct")        
        
    def testLoadProjectionWithoutDimensions(self):
        nr = NexusReader(self.filename)
        projections = nr.loadProjection()        
        self.assertEqual(projections.shape, (91,135,160), "Loaded projection data dimensions are not correct")        

    def testLoadProjectionWithDimensions(self):
        nr = NexusReader(self.filename)
        projections = nr.loadProjection((slice(0,1), slice(0,135), slice(0,160)))        
        self.assertEqual(projections.shape, (1,135,160), "Loaded projection data dimensions are not correct")        
            
    def testLoadProjectionCompareSingle(self):
        nr = NexusReader(self.filename)
        projections_full = nr.loadProjection()
        projections_part = nr.loadProjection((slice(0,1), slice(0,135), slice(0,160))) 
        numpy.testing.assert_array_equal(projections_part, projections_full[0:1,:,:])
        
    def testLoadProjectionCompareMulti(self):
        nr = NexusReader(self.filename)
        projections_full = nr.loadProjection()
        projections_part = nr.loadProjection((slice(0,3), slice(0,135), slice(0,160))) 
        numpy.testing.assert_array_equal(projections_part, projections_full[0:3,:,:])        
             
    def testLoadProjectionCompareRandom(self):
        nr = NexusReader(self.filename)
        projections_full = nr.loadProjection()
        projections_part = nr.loadProjection((slice(1,8), slice(5,10), slice(8,20))) 
        numpy.testing.assert_array_equal(projections_part, projections_full[1:8,5:10,8:20])                

    def testLoadProjectionCompareFull(self):
        nr = NexusReader(self.filename)
        projections_full = nr.loadProjection()
        projections_part = nr.loadProjection((slice(None,None), slice(None,None), slice(None,None))) 
        numpy.testing.assert_array_equal(projections_part, projections_full[:,:,:])
        
    def testLoadFlatCompareFull(self):
        nr = NexusReader(self.filename)
        flats_full = nr.loadFlat()
        flats_part = nr.loadFlat((slice(None,None), slice(None,None), slice(None,None))) 
        numpy.testing.assert_array_equal(flats_part, flats_full[:,:,:])
              
    def testLoadDarkCompareFull(self):
        nr = NexusReader(self.filename)
        darks_full = nr.loadDark()
        darks_part = nr.loadDark((slice(None,None), slice(None,None), slice(None,None))) 
        numpy.testing.assert_array_equal(darks_part, darks_full[:,:,:])
        
    def testProjectionAngles(self):
        nr = NexusReader(self.filename)
        angles = nr.getProjectionAngles()
        self.assertEqual(angles.shape, (91,), "Loaded projection number of angles are not correct")        
        
class TestXTEKReader(unittest.TestCase):
    
    def setUp(self):
        testpath, filename = os.path.split(os.path.realpath(__file__))
        testpath = os.path.normpath(os.path.join(testpath, '..','..','..'))
        self.filename = os.path.join(testpath,'data','SophiaBeads','SophiaBeads_64_averaged.xtekct')
                           
    def tearDown(self):
        pass
        
    def testLoad(self):
        xtekReader = XTEKReader(self.filename)
        self.assertEqual(xtekReader.geometry.pixel_num_h, 500, "Detector pixel X size is not correct")
        self.assertEqual(xtekReader.geometry.pixel_num_v, 500, "Detector pixel Y size is not correct")
        self.assertEqual(xtekReader.geometry.dist_source_center, -80.6392412185669, "Distance from source to center is not correct")
        self.assertEqual(xtekReader.geometry.dist_center_detector, (1007.006 - 80.6392412185669), "Distance from center to detector is not correct")
        
    def testReadAngles(self):    
        xtekReader = XTEKReader(self.filename)
        angles = xtekReader.readAngles()
        self.assertEqual(angles.shape, (63,), "Angles doesn't match")
        self.assertAlmostEqual(angles[46], -085.717, 3, "46th Angles doesn't match")
        
    def testLoadProjection(self):
        xtekReader = XTEKReader(self.filename)
        pixels = xtekReader.loadProjection()
        self.assertEqual(pixels.shape, (63, 500, 500), "projections shape doesn't match")
        
        
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'TestNexusReader.testLoad']
    unittest.main()