# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 15:05:00 2019

@author: ofn77899
"""

import unittest
import wget
import os
from ccpi.io.reader import NexusReader
import numpy


class TestNexusReader(unittest.TestCase):

    def setUp(self):
        wget.download('https://github.com/DiamondLightSource/Savu/raw/master/test_data/data/24737_fd.nxs')
        self.filename = '24737_fd.nxs'

    def tearDown(self):
        os.remove(self.filename)


    def testGetDimensions(self):
        nr = NexusReader(self.filename)
        self.assertEqual(nr.get_sinogram_dimensions(), (135, 91, 160), "Sinogram dimensions are not correct")
        
    def testGetProjectionDimensions(self):
        nr = NexusReader(self.filename)
        self.assertEqual(nr.get_projection_dimensions(), (91,135,160), "Projection dimensions are not correct")        
        
    def testLoadProjectionWithoutDimensions(self):
        nr = NexusReader(self.filename)
        projections = nr.load_projection()        
        self.assertEqual(projections.shape, (91,135,160), "Loaded projection data dimensions are not correct")        

    def testLoadProjectionWithDimensions(self):
        nr = NexusReader(self.filename)
        projections = nr.load_projection((slice(0,1), slice(0,135), slice(0,160)))        
        self.assertEqual(projections.shape, (1,135,160), "Loaded projection data dimensions are not correct")        
            
    def testLoadProjectionCompareSingle(self):
        nr = NexusReader(self.filename)
        projections_full = nr.load_projection()
        projections_part = nr.load_projection((slice(0,1), slice(0,135), slice(0,160))) 
        numpy.testing.assert_array_equal(projections_part, projections_full[0:1,:,:])
        
    def testLoadProjectionCompareMulti(self):
        nr = NexusReader(self.filename)
        projections_full = nr.load_projection()
        projections_part = nr.load_projection((slice(0,3), slice(0,135), slice(0,160))) 
        numpy.testing.assert_array_equal(projections_part, projections_full[0:3,:,:])
        
    def testLoadProjectionCompareRandom(self):
        nr = NexusReader(self.filename)
        projections_full = nr.load_projection()
        projections_part = nr.load_projection((slice(1,8), slice(5,10), slice(8,20))) 
        numpy.testing.assert_array_equal(projections_part, projections_full[1:8,5:10,8:20])                
        
    def testLoadProjectionCompareFull(self):
        nr = NexusReader(self.filename)
        projections_full = nr.load_projection()
        projections_part = nr.load_projection((slice(None,None), slice(None,None), slice(None,None))) 
        numpy.testing.assert_array_equal(projections_part, projections_full[:,:,:])
        
    def testLoadFlatCompareFull(self):
        nr = NexusReader(self.filename)
        flats_full = nr.load_flat()
        flats_part = nr.load_flat((slice(None,None), slice(None,None), slice(None,None))) 
        numpy.testing.assert_array_equal(flats_part, flats_full[:,:,:])
        
    def testLoadDarkCompareFull(self):
        nr = NexusReader(self.filename)
        darks_full = nr.load_dark()
        darks_part = nr.load_dark((slice(None,None), slice(None,None), slice(None,None))) 
        numpy.testing.assert_array_equal(darks_part, darks_full[:,:,:])
        
    def testProjectionAngles(self):
        nr = NexusReader(self.filename)
        angles = nr.get_projection_angles()
        self.assertEqual(angles.shape, (91,), "Loaded projection number of angles are not correct")        
        
    def test_get_acquisition_data_subset(self):
        nr = NexusReader(self.filename)
        key = nr.get_image_keys()
        sl = nr.get_acquisition_data_subset(0,10)
        data = nr.get_acquisition_data().subset(['vertical','horizontal'])
        
        self.assertTrue(sl.shape , (10,data.shape[1]))



if __name__ == '__main__':
    unittest.main()
    
