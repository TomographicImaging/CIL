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

import unittest
has_wget = True
try:
    import wget
except ImportError as ie:
    has_wget = False
import os
from ccpi.io.reader import NexusReader
import numpy


class TestNexusReader(unittest.TestCase):

    def setUp(self):
        if has_wget:
            wget.download('https://github.com/DiamondLightSource/Savu/raw/master/test_data/data/24737_fd.nxs')
            self.filename = '24737_fd.nxs'

    def tearDown(self):
        if has_wget:
            os.remove(self.filename)

    def testAll(self):
        if has_wget:
        # def testGetDimensions(self):
            nr = NexusReader(self.filename)
            self.assertEqual(nr.get_sinogram_dimensions(), (135, 91, 160), "Sinogram dimensions are not correct")

        # def testGetProjectionDimensions(self):
            nr = NexusReader(self.filename)
            self.assertEqual(nr.get_projection_dimensions(), (91,135,160), "Projection dimensions are not correct")        

        # def testLoadProjectionWithoutDimensions(self):
            nr = NexusReader(self.filename)
            projections = nr.load_projection()        
            self.assertEqual(projections.shape, (91,135,160), "Loaded projection data dimensions are not correct")        

        # def testLoadProjectionWithDimensions(self):
            nr = NexusReader(self.filename)
            projections = nr.load_projection((slice(0,1), slice(0,135), slice(0,160)))        
            self.assertEqual(projections.shape, (1,135,160), "Loaded projection data dimensions are not correct")        

        # def testLoadProjectionCompareSingle(self):
            nr = NexusReader(self.filename)
            projections_full = nr.load_projection()
            projections_part = nr.load_projection((slice(0,1), slice(0,135), slice(0,160))) 
            numpy.testing.assert_array_equal(projections_part, projections_full[0:1,:,:])

        # def testLoadProjectionCompareMulti(self):
            nr = NexusReader(self.filename)
            projections_full = nr.load_projection()
            projections_part = nr.load_projection((slice(0,3), slice(0,135), slice(0,160))) 
            numpy.testing.assert_array_equal(projections_part, projections_full[0:3,:,:])

        # def testLoadProjectionCompareRandom(self):
            nr = NexusReader(self.filename)
            projections_full = nr.load_projection()
            projections_part = nr.load_projection((slice(1,8), slice(5,10), slice(8,20))) 
            numpy.testing.assert_array_equal(projections_part, projections_full[1:8,5:10,8:20])                

        # def testLoadProjectionCompareFull(self):
            nr = NexusReader(self.filename)
            projections_full = nr.load_projection()
            projections_part = nr.load_projection((slice(None,None), slice(None,None), slice(None,None))) 
            numpy.testing.assert_array_equal(projections_part, projections_full[:,:,:])

        # def testLoadFlatCompareFull(self):
            nr = NexusReader(self.filename)
            flats_full = nr.load_flat()
            flats_part = nr.load_flat((slice(None,None), slice(None,None), slice(None,None))) 
            numpy.testing.assert_array_equal(flats_part, flats_full[:,:,:])

        # def testLoadDarkCompareFull(self):
            nr = NexusReader(self.filename)
            darks_full = nr.load_dark()
            darks_part = nr.load_dark((slice(None,None), slice(None,None), slice(None,None))) 
            numpy.testing.assert_array_equal(darks_part, darks_full[:,:,:])

        # def testProjectionAngles(self):
            nr = NexusReader(self.filename)
            angles = nr.get_projection_angles()
            self.assertEqual(angles.shape, (91,), "Loaded projection number of angles are not correct")        

        # def test_get_acquisition_data_subset(self):
            nr = NexusReader(self.filename)
            key = nr.get_image_keys()
            sl = nr.get_acquisition_data_subset(0,10)
            data = nr.get_acquisition_data()
            print (data.geometry)
            print (data.geometry.dimension_labels)
            print (data.dimension_labels)
            rdata = data.subset(channel=0)
            
            #
            
            self.assertTrue(sl.shape , (10,rdata.shape[1]))

            try:
                data.subset(['vertical','horizontal'])
                self.assertTrue(False)
            except ValueError as ve:
                print ("Exception catched", ve)
                self.assertTrue(True)
        else:
            # skips all tests if module wget is not present
            self.assertFalse(has_wget)


if __name__ == '__main__':
    unittest.main()
    
