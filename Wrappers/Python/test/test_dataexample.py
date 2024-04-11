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
from cil.framework.framework import ImageGeometry,AcquisitionGeometry
from cil.utilities import dataexample
from cil.utilities import noise
import os, sys, shutil
from testclass import CCPiTestClass
import platform
import numpy as np
from unittest.mock import patch, MagicMock 
from urllib import request
from zipfile import ZipFile
from io import StringIO
from tempfile import NamedTemporaryFile

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
        self.shapes_path = os.path.join(dataexample.CILDATA.data_dir, dataexample.TestData.SHAPES)

    def mock_urlopen(self, mock_urlopen, zipped_bytes):
        mock_response = MagicMock()
        mock_response.read.return_value = zipped_bytes
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

    @unittest.skipIf(platform.system() == 'Windows', "Skip on Windows")   
    @patch('cil.utilities.dataexample.urlopen')
    def test_unzip_remote_data(self, mock_urlopen):
        '''
        Test the _download_and_extract_data_from_url function correctly extracts files from a byte string
        The zipped byte string is mocked using a temporary local zip file
        '''
        
        # create a temporary zip file to test the function
        with NamedTemporaryFile(suffix = '.zip') as tf:
            tmp_path = os.path.dirname(tf.name)
            tmp_dir = os.path.splitext(os.path.basename(tf.name))[0]
            with ZipFile(tf.name, mode='w') as zip_file:
                zip_file.write(self.shapes_path, arcname=dataexample.TestData.SHAPES)
                
            with open(tf.name, 'rb') as zip_file:
                zipped_bytes = zip_file.read()
        
        self.mock_urlopen(mock_urlopen, zipped_bytes)
        dataexample.REMOTEDATA._download_and_extract_from_url(os.path.join(tmp_path, tmp_dir))

        self.assertTrue(os.path.isfile(os.path.join(tmp_path, tmp_dir, dataexample.TestData.SHAPES)))

        if os.path.exists(os.path.join(tmp_path,tmp_dir)):
            shutil.rmtree(os.path.join(tmp_path,tmp_dir)) 
        
    @unittest.skipIf(platform.system() == 'Windows', "Skip on Windows")   
    @patch('cil.utilities.dataexample.input', return_value='n')    
    @patch('cil.utilities.dataexample.urlopen')
    def test_download_data_input_n(self, mock_urlopen, input):
        '''
        Test the download_data function, when the user input is 'n' to 'are you sure you want to download data'
        The zipped byte string is mocked using a temporary local zip file
        '''
        
        # create a temporary zip file to test the function
        with NamedTemporaryFile(suffix = '.zip') as tf:
            tmp_path = os.path.dirname(tf.name)
            tmp_dir = os.path.splitext(os.path.basename(tf.name))[0]
            with ZipFile(tf.name, mode='w') as zip_file:
                zip_file.write(self.shapes_path, arcname=dataexample.TestData.SHAPES)
                
            with open(tf.name, 'rb') as zip_file:
                    zipped_bytes = zip_file.read()

        self.mock_urlopen(mock_urlopen, zipped_bytes)

        for data in self.data_list:
            # redirect print output
            capturedOutput = StringIO()                 
            sys.stdout = capturedOutput 
            test_func = getattr(dataexample, data)
            test_func.download_data(os.path.join(tmp_path, tmp_dir))
            self.assertFalse(os.path.isfile(os.path.join(tmp_path, tmp_dir, test_func.FOLDER, dataexample.TestData.SHAPES)), msg = "Failed with dataset " + data)
            self.assertEqual(capturedOutput.getvalue(),'Download cancelled\n', msg = "Failed with dataset " + data)
            # return to standard print output
            sys.stdout = sys.__stdout__ 

        if os.path.exists(os.path.join(tmp_path,tmp_dir)):
            shutil.rmtree(os.path.join(tmp_path,tmp_dir)) 

    @unittest.skipIf(platform.system() == 'Windows', "Skip on Windows")   
    @patch('cil.utilities.dataexample.input', return_value='y')    
    @patch('cil.utilities.dataexample.urlopen')
    def test_download_data_input_y(self, mock_urlopen, input):
        '''
        Test the download_data function, when the user input is 'y' to 'are you sure you want to download data'
        The zipped byte string is mocked using a temporary local zip file
        '''
        
        with NamedTemporaryFile(suffix = '.zip') as tf:
            tmp_path = os.path.dirname(tf.name)
            tmp_dir = os.path.splitext(os.path.basename(tf.name))[0]
            with ZipFile(tf.name, mode='w') as zip_file:
                zip_file.write(self.shapes_path, arcname=dataexample.TestData.SHAPES)
                
            with open(tf.name, 'rb') as zip_file:
                    zipped_bytes = zip_file.read()

        self.mock_urlopen(mock_urlopen, zipped_bytes)

        # redirect print output
        capturedOutput = StringIO()                 
        sys.stdout = capturedOutput         

        for data in self.data_list:
            test_func = getattr(dataexample, data)
            test_func.download_data(os.path.join(tmp_path, tmp_dir))
            self.assertTrue(os.path.isfile(os.path.join(tmp_path, tmp_dir, test_func.FOLDER, dataexample.TestData.SHAPES)), msg = "Failed with dataset " + data)
        
        # return to standard print output
        sys.stdout = sys.__stdout__ 
        
        if os.path.exists(os.path.join(tmp_path,tmp_dir)):
            shutil.rmtree(os.path.join(tmp_path,tmp_dir))


    def test_download_data_bad_URL(self):
        '''
        Test an error is raised when _download_and_extract_from_url has an empty URL
        '''
        with self.assertRaises(ValueError):
            dataexample.REMOTEDATA._download_and_extract_from_url('.')
