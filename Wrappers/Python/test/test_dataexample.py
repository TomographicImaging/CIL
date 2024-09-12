#  Copyright 2019 United Kingdom Research and Innovation
#  Copyright 2019 The University of Manchester
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
from cil.framework import ImageGeometry, AcquisitionGeometry
from cil.utilities import dataexample
from cil.utilities import noise
import os, sys, shutil
from testclass import CCPiTestClass
import platform
import numpy as np
from unittest.mock import patch 
from zipfile import ZipFile
from io import StringIO
import uuid

initialise_tests()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class TestTestData(CCPiTestClass):
    def test_noise_gaussian(self):
        camera = dataexample.CAMERA.get()
        noisy_camera = noise.gaussian(camera, seed=1)
        norm = (camera - noisy_camera).norm()
        decimal = 4
        if platform.system() == 'Darwin':
            decimal = 2
        self.assertAlmostEqual(norm, 48.881268, places=decimal)


    def check_load(self, example):
        try:
            image = example.get()
        except FileNotFoundError:
            self.assertFalse(msg="File not found")
        except:
            self.assertFalse(msg="Failed to load file")

        return image


    def test_load_CAMERA(self):

        image = self.check_load(dataexample.CAMERA)

        ig_expected = ImageGeometry(512,512)
        self.assertEqual(ig_expected,image.geometry,msg="Image geometry mismatch")


    def test_load_BOAT(self):

        image = self.check_load(dataexample.BOAT)

        ig_expected = ImageGeometry(512,512)
        self.assertEqual(ig_expected,image.geometry,msg="Image geometry mismatch")


    def test_load_PEPPERS(self):
        image = self.check_load(dataexample.PEPPERS)

        ig_expected = ImageGeometry(512,512,channels=3,dimension_labels=['channel', 'horizontal_y', 'horizontal_x'])
        self.assertEqual(ig_expected,image.geometry,msg="Image geometry mismatch")


    def test_load_RAINBOW(self):

        image = self.check_load(dataexample.RAINBOW)

        ig_expected = ImageGeometry(1194,1353,channels=3,dimension_labels=['channel', 'horizontal_y', 'horizontal_x'])
        self.assertEqual(ig_expected,image.geometry,msg="Image geometry mismatch")


    def test_load_RESOLUTION_CHART(self):

        image = self.check_load(dataexample.RESOLUTION_CHART)

        ig_expected = ImageGeometry(256,256)
        self.assertEqual(ig_expected,image.geometry,msg="Image geometry mismatch")


    def test_load_SIMPLE_PHANTOM_2D(self):

        image = self.check_load(dataexample.SIMPLE_PHANTOM_2D)

        ig_expected = ImageGeometry(512,512)
        self.assertEqual(ig_expected,image.geometry,msg="Image geometry mismatch")


    def test_load_SHAPES(self):

        image = self.check_load(dataexample.SHAPES)

        ig_expected = ImageGeometry(300,200)
        self.assertEqual(ig_expected,image.geometry,msg="Image geometry mismatch")


    def test_load_SYNCHROTRON_PARALLEL_BEAM_DATA(self):

        image = self.check_load(dataexample.SYNCHROTRON_PARALLEL_BEAM_DATA)

        ag_expected = AcquisitionGeometry.create_Parallel3D()\
                                         .set_panel((160,135),(1,1))\
                                         .set_angles(np.linspace(-88.2,91.8,91))

        self.assertEqual(ag_expected.shape,image.geometry.shape,msg="Image geometry mismatch")
        np.testing.assert_allclose(ag_expected.angles, image.geometry.angles,atol=0.05)


    def test_load_SIMULATED_SPHERE_VOLUME(self):

        image = self.check_load(dataexample.SIMULATED_SPHERE_VOLUME)

        ig_expected = ImageGeometry(128,128,128,16,16,16)

        self.assertEqual(ig_expected,image.geometry,msg="Image geometry mismatch")


    def test_load_SIMULATED_PARALLEL_BEAM_DATA(self):

        image = self.check_load(dataexample.SIMULATED_PARALLEL_BEAM_DATA)

        ag_expected = AcquisitionGeometry.create_Parallel3D()\
                                         .set_panel((128,128),(16,16))\
                                         .set_angles(np.linspace(0,360,300,False))

        self.assertEqual(ag_expected,image.geometry,msg="Acquisition geometry mismatch")


    def test_load_SIMULATED_CONE_BEAM_DATA(self):

        image = self.check_load(dataexample.SIMULATED_CONE_BEAM_DATA)

        ag_expected = AcquisitionGeometry.create_Cone3D([0,-20000,0],[0,60000,0])\
                                         .set_panel((128,128),(64,64))\
                                         .set_angles(np.linspace(0,360,300,False))

        self.assertEqual(ag_expected,image.geometry,msg="Acquisition geometry mismatch")   

class TestRemoteData(unittest.TestCase):

    def setUp(self):
        self.data_list = ['WALNUT','USB','KORN','SANDSTONE']


    def mock_zenodo_get(*args):
        # mock zenodo_get by making a zip file containing the shapes test data when the function is called
        shapes_path = os.path.join(dataexample.CILDATA.data_dir, dataexample.TestData.SHAPES)
        with ZipFile(os.path.join(args[0][4], args[0][2]), mode='w') as zip_file:
            zip_file.write(shapes_path, arcname=dataexample.TestData.SHAPES)

            
    @patch('cil.utilities.dataexample.input', return_value='y')
    @patch('cil.utilities.dataexample.zenodo_get', side_effect=mock_zenodo_get)
    def test_download_data_input_y(self, mock_zenodo_get, input):
        '''
        Test the download_data function, when the user input is 'y' to 'are you sure you want to download data'
        The user input to confirm the download is mocked as 'y'
        The zip file download is mocked by creating a zip file locally
        Test the download_data function correctly extracts files from the zip file
        '''        
        # create a temporary folder in the CIL data directory
        tmp_dir = os.path.join(dataexample.CILDATA.data_dir, str(uuid.uuid4()))
        os.makedirs(tmp_dir)
        # redirect print output
        capturedOutput = StringIO()                
        sys.stdout = capturedOutput
        for data in self.data_list:
            test_func = getattr(dataexample, data)
            test_func.download_data(tmp_dir)
            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, getattr(test_func, 'FOLDER'), dataexample.TestData.SHAPES)), 
                            msg = "Download data test failed with dataset " + data)
        # return to standard print output
        sys.stdout = sys.__stdout__
        shutil.rmtree(tmp_dir)


    @patch('cil.utilities.dataexample.input', return_value='n')
    @patch('cil.utilities.dataexample.zenodo_get', side_effect=mock_zenodo_get)   
    def test_download_data_input_n(self, mock_zenodo_get, input):
        '''
        Test the download_data function, when the user input is 'n' to 'are you sure you want to download data'
        '''
        # create a temporary folder in the CIL data directory
        tmp_dir = os.path.join(dataexample.CILDATA.data_dir, str(uuid.uuid4()))
        os.makedirs(tmp_dir)
        for data in self.data_list:
            # redirect print output
            capturedOutput = StringIO()
            sys.stdout = capturedOutput
            test_func = getattr(dataexample, data)
            test_func.download_data(tmp_dir)
            self.assertFalse(os.path.isfile(os.path.join(tmp_dir, getattr(test_func, 'FOLDER'), dataexample.TestData.SHAPES)), 
                             msg = "Download dataset test failed with dataset " + data)
            self.assertEqual(capturedOutput.getvalue(),'Download cancelled\n', 
                             msg = "Download dataset test failed with dataset " + data)
            # return to standard print output
            sys.stdout = sys.__stdout__ 

        shutil.rmtree(tmp_dir)


    @patch('cil.utilities.dataexample.input', return_value='y')
    def test_download_data_empty(self, input):
        '''
        Test an error is raised when download_data is used on an empty Zenodo record
        '''
        remote_data = dataexample.REMOTEDATA
        remote_data.ZENODO_RECORD = 'empty'
        remote_data.FOLDER = 'empty'
        
        with self.assertRaises(ValueError):
            remote_data.download_data('.')