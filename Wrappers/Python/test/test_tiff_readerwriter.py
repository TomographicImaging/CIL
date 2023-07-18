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
from cil.io import TIFFWriter
from cil.io import TIFFStackReader

from cil.framework import ImageGeometry, AcquisitionGeometry
import os
import numpy as np
import shutil, glob

initialise_tests()

class TIFFReadWriter(unittest.TestCase):
    def setUp(self):
        ig = ImageGeometry(10,11,12, channels=3)

        data = ig.allocate(0)
        arr = data.as_array()
        k = 0
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                arr[i][j] += k
                k += 1
        data.fill(arr)
        self.cwd = os.getcwd()
        # filename contains brackets to test if previous problems 
        # with use of glob in the tiff reader have been resolved:
        fname = os.path.join(self.cwd, 'test_tiff [0]','myfile.tif')
        self.data_dir = os.path.dirname(fname)
        writer = TIFFWriter(data=data, file_name=fname, counter_offset=0)
        writer.write()
        
        self.ig = ig
        self.data = data


    def tearDown(self):
        shutil.rmtree(self.data_dir)


    def test_write_expected_num_files(self):
        data = self.data
        files = glob.glob(os.path.join(glob.escape(self.data_dir), '*'))
        assert len(files) == data.shape[0]*data.shape[1]


    def test_read1(self):
        data = self.data
        reader = TIFFStackReader(file_name = self.data_dir)
        read = reader.read()
        np.testing.assert_array_equal(read.flatten(), data.as_array().flatten())


    def test_read_as_ImageData1(self):
        reader = TIFFStackReader(file_name = self.data_dir)
        
        img = reader.read_as_ImageData(self.ig)
        np.testing.assert_array_equal(img.as_array(), self.data.as_array())
    

    def test_read_as_ImageData_Exceptions(self):
        igs = [ ImageGeometry(10,11,12, channels=5) ]
        igs.append( ImageGeometry(12,32) )
        reader = TIFFStackReader(file_name = self.data_dir)
        
        for geom in igs:
            with self.assertRaises(ValueError):
                img = reader.read_as_ImageData(geom)
                

    def test_read_as_AcquisitionData1(self):
        ag = AcquisitionGeometry.create_Parallel3D()
        ag.set_panel([10,11])
        ag.set_angles([i for i in range(12)])
        ag.set_channels(3)
        print (ag.shape)
        # print (glob.glob(os.path.join(self.data_dir, '*')))
        reader = TIFFStackReader(file_name = self.data_dir)
        acq = reader.read_as_AcquisitionData(ag)

        np.testing.assert_array_equal(acq.as_array(), self.data.as_array())


    def test_read_as_AcquisitionData2(self):
        # with this data will be scrambled but reshape is possible
        ag = AcquisitionGeometry.create_Parallel3D()
        ag.set_panel([11,10])
        ag.set_angles([i for i in range(12)])
        ag.set_channels(3)

        reader = TIFFStackReader(file_name = self.data_dir)
        acq = reader.read_as_AcquisitionData(ag)

        np.testing.assert_array_equal(acq.as_array().flatten(), self.data.as_array().flatten())


    def test_read_as_AcquisitionData_Exceptions1(self):

            ag = AcquisitionGeometry.create_Parallel3D()
            ag.set_panel([11,12])
            ag.set_angles([i for i in range(12)])
            ag.set_channels(3)
            reader = TIFFStackReader(file_name = self.data_dir)
            with self.assertRaises(ValueError):
                acq = reader.read_as_AcquisitionData(ag)
                

    def test_read_as_AcquisitionData_Exceptions2(self):
            ag = AcquisitionGeometry.create_Parallel3D()
            ag.set_panel([11,12])
            ag.set_angles([i for i in range(12)])
            ag.set_channels(3)
            reader = TIFFStackReader(file_name = self.data_dir)

            with self.assertRaises(TypeError):
                class Fake(object):
                    pass
                fake = Fake()
                fake.shape = (36,11,10)
                acq = reader.read_as_ImageData(fake)
                