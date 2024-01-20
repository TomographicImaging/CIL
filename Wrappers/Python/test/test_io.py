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
from unittest.mock import patch, mock_open, Mock
from utils import initialise_tests
from cil.framework import AcquisitionGeometry, AcquisitionData
import numpy as np
import os
from cil.framework import ImageGeometry
from cil.io import TXRMDataReader, NEXUSDataReader, NikonDataReader, ZEISSDataReader
from cil.io.ReaderABC import Reader
from cil.io import TIFFWriter, TIFFStackReader
from cil.io.utilities import HDF5_utilities
from cil.processors import Slicer
from utils import has_astra, has_nvidia
from cil.utilities.dataexample import data_dir
from cil.utilities.quality_measures import mse
from cil.utilities import dataexample
import shutil
import logging
import glob
import json
from cil.io import utilities
from cil.io import RAWFileWriter
import configparser
import tempfile
from abc import ABC, abstractmethod
from cil.utilities.display import show2D

initialise_tests()

has_dxchange = True
try:
    import dxchange
except ImportError as ie:
    has_dxchange = False
has_olefile = True
try:
    import olefile
except ImportError as ie:
    has_olefile = False
has_wget = True
try:
    import wget
except ImportError as ie:
    has_wget = False
if has_astra:
    from cil.plugins.astra import FBP


# change basedir to point to the location of the walnut dataset which can
# be downloaded from https://zenodo.org/record/4822516
# basedir = os.path.abspath('/home/edo/scratch/Data/Walnut/valnut_2014-03-21_643_28/tomo-A/')
basedir = data_dir
filename = os.path.join(basedir, "valnut_tomo-A.txrm")
has_file = os.path.isfile(filename)


has_prerequisites = has_olefile and has_dxchange and has_astra and has_nvidia and has_file \
    and has_wget

# Change the level of the logger to WARNING (or whichever you want) to see more information
logging.basicConfig(level=logging.WARNING)

logging.info ("has_astra {}".format(has_astra))
logging.info ("has_wget {}".format(has_wget))
logging.info ("has_olefile {}".format(has_olefile))
logging.info ("has_dxchange {}".format(has_dxchange))
logging.info ("has_file {}".format(has_file))

if not has_file:
    logging.info("This unittest requires the walnut Zeiss dataset saved in {}".format(data_dir))


class ReaderConcreteClass(Reader):
    """
    A concrete implementation of the Reader base class in order to test full functionality of readerABC
    """

    supported_extensions= ['txt']


    def __init__(self, file_name):

        super().__init__(file_name)

        # for the test
        self.data = np.random.random_integers(200,65535,self.geometry.shape)
        self.flat_field = np.ones((self.geometry.shape[1::]),dtype=np.int32)*60000
        self.dark_field = np.zeros((self.geometry.shape[1::]),dtype=np.int32)


    def _read_metadata(self):
        self._metadata = {'num_pixels_x':4,'num_pixels_y':8,
                          'size_pixels_x':0.2, 'size_pixels_y':0.2,
                          'num_projections':3,'angles':[0,45,90]}


    def _create_geometry(self):
        self._acquisition_geometry = AcquisitionGeometry.create_Parallel3D()
        self._acquisition_geometry.set_angles(self._metadata['angles'])
        self._acquisition_geometry.set_panel((self._metadata['num_pixels_x'],self._metadata['num_pixels_y']),(self._metadata['size_pixels_x'],self._metadata['size_pixels_y']))
        self._acquisition_geometry.set_labels(labels=['angle', 'vertical', 'horizontal'])


    def get_raw_flatfield(self):
        return self.flat_field


    def get_raw_darkfield(self):
        return self.dark_field


    def get_raw_data(self):
        return self.data


    def _create_normalisation_correction(self):
        darkfield = self.get_raw_darkfield().astype(np.float32)
        flatfield = self.get_raw_flatfield().astype(np.float32)

        self._normalisation = (darkfield, 1/(flatfield-darkfield))


    def _apply_normalisation(self, data_array):
        data_array -= self._normalisation[0][tuple(self._panel_crop)]
        data_array *= self._normalisation[1][tuple(self._panel_crop)]


    def _get_data(self, proj_slice=None):

        if proj_slice is None:
            selection = (slice(None),*self._panel_crop)
        else:
            selection = (proj_slice,*self._panel_crop)


        return self.data[selection].astype(np.float32)


class TestReaderBaseClass(unittest.TestCase):

    @patch('os.path.isfile')
    def create_reader(self, mock_isfile):
        mock_isfile.return_value = True
        self.mock_path = '/mock/path/to/file.txt'

        reader = ReaderConcreteClass(self.mock_path)
        return reader


    def test_get_normalised_data(self):
        reader = self.create_reader()
        data = reader._get_normalised_data(projs=None)

        norm = reader.flat_field - reader.dark_field
        data_gold = (reader.data.astype(np.float32) - reader.dark_field.astype(np.float32)) / norm.astype(np.float32)

        np.testing.assert_allclose(data, data_gold,rtol=1e-6)


        #  test with binning
            # all
            # list
            # slice

        # with no binning
            # all
            # list
            # slice


    def test_set_panel_roi(self):

        reader = self.create_reader()

        self.assertTupleEqual(reader._panel_crop, (slice(None),slice(None)))
        self.assertFalse(reader._bin)
        self.assertTupleEqual(reader._bin_roi, (1,1))

        reader.set_panel_roi(vertical=(1,-2,2),horizontal=(None,7,3))

        # check roi
        self.assertTupleEqual(reader._panel_crop, (slice(1,-2),slice(None,7)))
        self.assertTrue(reader._bin)
        self.assertTupleEqual(reader._bin_roi, (2,3))

        reader.reset()

        self.assertTupleEqual(reader._panel_crop, (slice(None),slice(None)))
        self.assertFalse(reader._bin)
        self.assertTupleEqual(reader._bin_roi, (1,1))


    def test_set_projections(self):

        reader = self.create_reader()

        self.assertIsNone(reader._angle_indices)

        # index
        reader.set_projections(1)
        self.assertListEqual(reader._angle_indices, [1])

        reader.set_projections(-1)
        self.assertListEqual(reader._angle_indices, [-1])

        with self.assertRaises(ValueError):
            reader.set_projections(3)

        # slice
        reader.set_projections((None,None,2))
        self.assertEqual(reader._angle_indices, slice(None,None,2))

        reader.set_projections((1,None,1))
        self.assertEqual(reader._angle_indices, slice(1,None,1))

        with self.assertRaises(ValueError):
            reader.set_projections((5,7,1))

        # list
        reader.set_projections([0])
        self.assertListEqual(reader._angle_indices, [0])

        reader.set_projections([0,2])
        self.assertListEqual(reader._angle_indices, [0,2])

        with self.assertRaises(ValueError):
            reader.set_projections([0,3])

        reader.reset()
        self.assertIsNone(reader._angle_indices)


    def test_preview(self):

        reader = self.create_reader()

        with patch.object(show2D,'_show2D__show2D') as mock_show:
            reader.preview()

        mock_show.assert_called_once()


    def test_get_geometry(self):

        # reflects roi changes
        reader = self.create_reader()

        ag = AcquisitionGeometry.create_Parallel3D().set_panel([4,8],[0.2,0.2]).set_angles([0,45,90])

        self.assertEquals(reader.geometry, ag)
        self.assertEquals(reader.get_geometry(), ag)
        
        reader.set_panel_roi(horizontal=(2,4,2),vertical=(1,-2,2))
        reader.set_projections(2)

        ag = AcquisitionGeometry.create_Parallel3D(detector_position=[ 0.2, 0. ,-0.2]).set_panel([1,2],[0.4,0.4]).set_angles([90])
        self.assertEquals(reader.get_geometry(), ag)


    def test_read(self):

        #reflects roi changes
        reader = self.create_reader()

        acquisition_data = reader.read()

        self.assertIsInstance(acquisition_data, AcquisitionData)

        ag = AcquisitionGeometry.create_Parallel3D().set_panel([4,8],[0.2,0.2]).set_angles([0,45,90])
        self.assertEquals(acquisition_data.geometry, ag)

        norm = reader.flat_field - reader.dark_field
        data_gold = (reader.data.astype(np.float32) - reader.dark_field.astype(np.float32)) / norm.astype(np.float32)

        np.testing.assert_allclose(acquisition_data.array, data_gold,rtol=1e-6)
        self.assertEqual(acquisition_data.array.dtype, np.float32)



        # test with roi and binning
        reader.set_panel_roi(horizontal=(2,4,2),vertical=(1,-2,2))
        reader.set_projections(2)

        acquisition_data = reader.read()
        self.assertIsInstance(acquisition_data, AcquisitionData)

        ag = AcquisitionGeometry.create_Parallel3D(detector_position=[ 0.2, 0. ,-0.2]).set_panel([1,2],[0.4,0.4]).set_angles([90])
        self.assertEquals(acquisition_data.geometry, ag)

        norm = reader.flat_field - reader.dark_field
        data_byhand = (reader.data.astype(np.float32) - reader.dark_field.astype(np.float32)) / norm.astype(np.float32)
        data_byhand = data_byhand[2,1:-3, 2:4].mean(-1).reshape(2,2).mean(-1)
        np.testing.assert_allclose(acquisition_data.array, data_byhand,rtol=1e-6)
        self.assertEqual(acquisition_data.array.dtype, np.float32)



        # test with roi and binning
        reader.reset()
        reader.set_projections(2)

        acquisition_data = reader.read()
        self.assertIsInstance(acquisition_data, AcquisitionData)

        ag = AcquisitionGeometry.create_Parallel3D().set_panel([4,8],[0.2,0.2]).set_angles([90])
        self.assertEquals(acquisition_data.geometry, ag)

        norm = reader.flat_field - reader.dark_field
        data_byhand = (reader.data.astype(np.float32) - reader.dark_field.astype(np.float32)) / norm.astype(np.float32)
        data_byhand = data_byhand[2]
        np.testing.assert_allclose(acquisition_data.array, data_byhand,rtol=1e-6)
        self.assertEqual(acquisition_data.array.dtype, np.float32)




class TestNikonDataReader(unittest.TestCase):

    def setUp(self) -> None:
        self.mock_path = '/mock/path/to/file.xtekct'


    @patch('os.path.isfile')
    def create_reader(self,mock_isfile):
        mock_isfile.return_value = True

        mock_contents = """
        Projections=2
        WhiteLevel=50000
        DetectorPixelsY=1024
        DetectorPixelsX=2048
        DetectorPixelSizeX=0.1
        DetectorPixelSizeY=0.2
        SrcToObject=50
        SrcToDetector=300
        AngularStep=0.2
        ObjectOffsetX=-0.03
        ObjectRoll=0.01
        ObjectTilt=30
        InitialAngle=-3
        ObjectOffsetX=-0.03
        ObjectRoll=0.01
        ObjectTilt=30
        InputFolderName=directory
        """


        with patch('builtins.open', mock_open(read_data=mock_contents)):
            reader = NikonDataReader(self.mock_path)
        return reader


    @patch('os.path.isfile')
    def test_setup_errors(self, mock_isfile):

        mock_isfile.return_value = False
        with self.assertRaises(FileNotFoundError):
            NikonDataReader(self.mock_path)

        mock_isfile.return_value = True
        with self.assertRaises(TypeError):
            NikonDataReader('/path/to/non-existing/file.txt')

    @patch('os.path.isfile')    
    def test_setup_mocked(self, mock_isfile):
        mock_isfile.return_value = True

        mock_read_metadata = Mock()
        mock_create_geometry = Mock()

        with patch.object(NikonDataReader, '_read_metadata', mock_read_metadata),\
             patch.object(NikonDataReader, '_create_geometry', mock_create_geometry):

            reader = NikonDataReader(self.mock_path)

        # Assert that the mocked methods were called by the constructor call
        self.assertEqual(reader._file_name, self.mock_path)
        mock_read_metadata.assert_called_once()
        mock_create_geometry.assert_called_once()


    def test_read_metadata(self):
        reader = self.create_reader()

        metadata = {
            'ObjectTilt':30.0, 
            'ObjectRoll': 0.01,
            'ObjectOffsetX':-0.03,
            'CentreOfRotationTop':0,
            'CentreOfRotationBottom':0,
            'Projections':2,
            'WhiteLevel':50000,
            'DetectorPixelsY':1024,
            'DetectorPixelsX':2048,
            'DetectorPixelSizeX':0.1,
            'DetectorPixelSizeY':0.2,
            'SrcToObject':50.0,
            'SrcToDetector':300.0,
            'AngularStep':0.2,
            'InitialAngle':-3.0,
            'InputFolderName':'directory'
        }

        self.assertDictEqual(reader.metadata, metadata)


    def test_create_geometry(self):

        reader = self.create_reader()

        ag = AcquisitionGeometry.create_Cone3D(source_position=(0,0,0),
                                            rotation_axis_position=(0.03,50,0),
                                            rotation_axis_direction=(1.74532924e-04, 0.5, 0.866025391),
                                            detector_position=(0,300,0)
                                            )
        ag.set_panel([2048, 1024],[0.1, 0.2], 'top-right')
        ag.set_angles([183,182.8])

        self.assertTrue(reader.geometry == ag )
        self.assertTrue(reader.geometry == reader.get_geometry() )


    def test_get_raw_flatfield(self):
        reader = self.create_reader()
        self.assertIsNone(reader.get_raw_flatfield())


    def test_get_raw_darkfield(self):
        reader = self.create_reader()
        self.assertIsNone(reader.get_raw_darkfield())


    def test_get_raw_data(self):
        pass


    def test_create_normalisation_correction(self):
        reader = self.create_reader()
        reader._create_normalisation_correction()
        self.assertEqual(reader._normalisation, 1./50000.)



class TestTXRMDataReader(unittest.TestCase):
    

    def setUp(self):
        logging.info ("has_astra {}".format(has_astra))
        logging.info ("has_wget {}".format(has_wget))
        logging.info ("has_olefile {}".format(has_olefile))
        logging.info ("has_dxchange {}".format(has_dxchange))
        logging.info ("has_file {}".format(has_file))
        if has_file:
            self.reader = TXRMDataReader()
            angle_unit = AcquisitionGeometry.RADIAN
            
            self.reader.set_up(file_name=filename, 
                               angle_unit=angle_unit)
            data = self.reader.read()
            if data.geometry is None:
                raise AssertionError("WTF")
            # Choose the number of voxels to reconstruct onto as number of detector pixels
            N = data.geometry.pixel_num_h
            
            # Geometric magnification
            mag = (np.abs(data.geometry.dist_center_detector) + \
                np.abs(data.geometry.dist_source_center)) / \
                np.abs(data.geometry.dist_source_center)
                
            # Voxel size is detector pixel size divided by mag
            voxel_size_h = data.geometry.pixel_size_h / mag
            voxel_size_v = data.geometry.pixel_size_v / mag

            self.mag = mag
            self.N = N
            self.voxel_size_h = voxel_size_h
            self.voxel_size_v = voxel_size_v

            self.data = data


    def tearDown(self):
        pass


    def test_run_test(self):
        print("run test Zeiss Reader")
        self.assertTrue(True)
    

    @unittest.skipIf(not has_prerequisites, "Prerequisites not met")
    def test_read_and_reconstruct_2D(self):
        
        # get central slice
        data2d = self.data.subset(vertical='centre')
        # d512 = self.data.subset(vertical=512)
        # data2d.fill(d512.as_array())
        # neg log
        data2d.log(out=data2d)
        data2d *= -1

        ig2d = data2d.geometry.get_ImageGeometry()
        # Construct the appropriate ImageGeometry
        ig2d = ImageGeometry(voxel_num_x=self.N,
                            voxel_num_y=self.N,
                            voxel_size_x=self.voxel_size_h, 
                            voxel_size_y=self.voxel_size_h)
        if data2d.geometry is None:
            raise AssertionError('What? None?')
        fbpalg = FBP(ig2d,data2d.geometry)
        fbpalg.set_input(data2d)
        
        recfbp = fbpalg.get_output()
        
        wget.download('https://www.ccpi.ac.uk/sites/www.ccpi.ac.uk/files/walnut_slice512.nxs',
                      out=data_dir)
        fname = os.path.join(data_dir, 'walnut_slice512.nxs')
        reader = NEXUSDataReader()
        reader.set_up(file_name=fname)
        gt = reader.read()

        qm = mse(gt, recfbp)
        logging.info ("MSE {}".format(qm) )

        np.testing.assert_almost_equal(qm, 0, decimal=3)
        fname = os.path.join(data_dir, 'walnut_slice512.nxs')
        os.remove(fname)


class TestTIFF(unittest.TestCase):
    def setUp(self) -> None:
        self.TMP = tempfile.TemporaryDirectory()
        self.cwd = os.path.join(self.TMP.name, 'rawtest')

    def tearDown(self) -> None:
        self.TMP.cleanup()
        
    def get_slice_imagedata(self, data):
        '''Returns only 2 slices of data'''
        # data = dataexample.SIMULATED_SPHERE_VOLUME.get()
        data.dimension_labels[0]
        roi = {data.dimension_labels[0]: (0,2,1), 
               data.dimension_labels[1]: (None, None, None), 
               data.dimension_labels[2]: (None, None, None)}
        return Slicer(roi=roi)(data)


    def test_tiff_stack_ImageData(self):
        # data = dataexample.SIMULATED_SPHERE_VOLUME.get()
        data = self.get_slice_imagedata(
            dataexample.SIMULATED_SPHERE_VOLUME.get()
        )
        
        fname = os.path.join(self.cwd, "unittest")

        writer = TIFFWriter(data=data, file_name=fname)
        writer.write()

        reader = TIFFStackReader(file_name=self.cwd)
        read_array = reader.read()

        np.testing.assert_allclose(data.as_array(), read_array)

        read = reader.read_as_ImageData(data.geometry)
        np.testing.assert_allclose(data.as_array(), read.as_array())


    def test_tiff_stack_AcquisitionData(self):
        # data = dataexample.SIMULATED_CONE_BEAM_DATA.get()
        data = self.get_slice_imagedata(
            dataexample.SIMULATED_CONE_BEAM_DATA.get()
        )
        
        
        fname = os.path.join(self.cwd, "unittest")

        writer = TIFFWriter(data=data, file_name=fname)
        writer.write()

        reader = TIFFStackReader(file_name=self.cwd)
        read_array = reader.read()

        np.testing.assert_allclose(data.as_array(), read_array)

        read = reader.read_as_AcquisitionData(data.geometry)
        np.testing.assert_allclose(data.as_array(), read.as_array())


    def test_tiff_stack_ImageDataSlice(self):
        data = dataexample.SIMULATED_SPHERE_VOLUME.get()
        
        fname = os.path.join(self.cwd, "unittest")

        writer = TIFFWriter(data=data, file_name=fname)
        writer.write()

        roi = {'axis_0': -1, 'axis_1': -1, 'axis_2': (None, None, 2)}

        reader = TIFFStackReader(file_name=self.cwd, roi=roi, mode='slice')
        read_array = reader.read()

        shape = [el for el in data.shape]
        shape[2] /= 2

        np.testing.assert_allclose(shape, read_array.shape )
        
        roi = {'axis_0': (0, 2, None), 'axis_1': -1, 'axis_2': -1}

        reader = TIFFStackReader(file_name=self.cwd, roi=roi, mode='slice')
        read_array = reader.read()

        np.testing.assert_allclose(data.as_array()[:2], read_array)


    def test_tiff_stack_ImageData_wrong_file(self):
        # data = dataexample.SIMULATED_SPHERE_VOLUME.get()
        data = self.get_slice_imagedata(
            dataexample.SIMULATED_SPHERE_VOLUME.get()
        )
        
        fname = os.path.join(self.cwd, "unittest")

        writer = TIFFWriter(data=data, file_name=fname)
        writer.write()

        for el in glob.glob(os.path.join(self.cwd , "unittest*.tiff")):
            # print (f"modifying {el}")
            with open(el, 'w') as f:
                f.write('BOOM')
            break
                
        reader = TIFFStackReader(file_name=self.cwd)
        try:
            read_array = reader.read()
            assert False
        except:
            assert True

    def test_TIFF_compression3D_0(self):
        self.TIFF_compression_test(None)
    
    def test_TIFF_compression3D_1(self):
        self.TIFF_compression_test('uint8')

    def test_TIFF_compression3D_2(self):
        self.TIFF_compression_test('uint16')

    def test_TIFF_compression3D_3(self):
        with self.assertRaises(ValueError) as context:
            self.TIFF_compression_test('whatever_compression')
            
    def test_TIFF_compression4D_0(self):
        self.TIFF_compression_test(None,2)
        
    def test_TIFF_compression4D_1(self):
        self.TIFF_compression_test('uint8',2)

    def test_TIFF_compression4D_2(self):
        self.TIFF_compression_test('uint16',2)
    
    def test_TIFF_compression4D_3(self):
        with self.assertRaises(ValueError) as context:
            self.TIFF_compression_test('whatever_compression',2)

    def TIFF_compression_test(self, compression, channels=1):
        X=4
        Y=5
        Z=6
        C=channels
        if C == 1:
            ig = ImageGeometry(voxel_num_x=4, voxel_num_y=5, voxel_num_z=6)
        else:
            ig = ImageGeometry(voxel_num_x=4, voxel_num_y=5, voxel_num_z=6, channels=C)
        data = ig.allocate(0)
        data.fill(np.arange(X*Y*Z*C).reshape(ig.shape))

        compress = utilities.get_compress(compression)
        dtype = utilities.get_compressed_dtype(data.array, compression)
        scale, offset = utilities.get_compression_scale_offset(data.array, compression)
        if C > 1:
            assert data.ndim == 4
        fname = os.path.join(self.cwd, "unittest")
        writer = TIFFWriter(data=data, file_name=fname, compression=compression)
        writer.write()
        # force the reader to use the native TIFF dtype by setting dtype=None
        reader = TIFFStackReader(file_name=self.cwd, dtype=None)
        read_array = reader.read()
        if C > 1:
            read_array = reader.read_as_ImageData(ig).array

        
        if compress:
            tmp = data.array * scale + offset
            tmp = np.asarray(tmp, dtype=dtype)
            # test if the scale and offset are written to the json file
            with open(os.path.join(self.cwd, "scaleoffset.json"), 'r') as f:
                d = json.load(f)
            assert d['scale'] == scale
            assert d['offset'] == offset
            # test if the scale and offset are read from the json file
            sc, of = reader.read_scale_offset()
            assert sc == scale
            assert of == offset
            
            recovered_data = (read_array - of)/sc
            np.testing.assert_allclose(recovered_data, data.array, rtol=1e-1, atol=1e-2)

            # test read_rescaled
            approx = reader.read_rescaled()
            np.testing.assert_allclose(approx.ravel(), data.array.ravel(), rtol=1e-1, atol=1e-2)

            approx = reader.read_rescaled(sc, of)
            np.testing.assert_allclose(approx.ravel(), data.array.ravel(), rtol=1e-1, atol=1e-2)
        else:
            tmp = data.array
            # if the compression is None, the scale and offset should not be written to the json file
            with self.assertRaises(OSError) as context:
                sc, of = reader.read_scale_offset()
        
        assert tmp.dtype == read_array.dtype
        
        np.testing.assert_array_equal(tmp, read_array)

class TestRAW(unittest.TestCase):
    def setUp(self) -> None:
        self.TMP = tempfile.TemporaryDirectory()
        self.cwd = os.path.join(self.TMP.name, 'rawtest')

    def tearDown(self) -> None:
        self.TMP.cleanup()
        # pass

    def test_raw_nocompression_0(self):
        self.RAW_compression_test(None,1)
    
    def test_raw_compression_0(self):
        self.RAW_compression_test('uint8',1)

    def test_raw_compression_1(self):
        self.RAW_compression_test('uint16',1)

    def test_raw_nocompression_1(self):
        self.RAW_compression_test(None,1)
    
    def test_raw_compression_2(self):
        with self.assertRaises(ValueError) as context:
            self.RAW_compression_test(12,1)

    def RAW_compression_test(self, compression, channels=1):
        X=4
        Y=5
        Z=6
        C=channels
        if C == 1:
            ig = ImageGeometry(voxel_num_x=4, voxel_num_y=5, voxel_num_z=6)
        else:
            ig = ImageGeometry(voxel_num_x=4, voxel_num_y=5, voxel_num_z=6, channels=C)
        data = ig.allocate(0)
        data.fill(np.arange(X*Y*Z*C).reshape(ig.shape))

        compress = utilities.get_compress(compression)
        dtype = utilities.get_compressed_dtype(data.array, compression)
        scale, offset = utilities.get_compression_scale_offset(data.array, compression)
        if C > 1:
            assert data.ndim == 4
        raw = "unittest.raw"
        fname = os.path.join(self.cwd, raw)
        
        writer = RAWFileWriter(data=data, file_name=fname, compression=compression)
        writer.write()

        # read the data from the ini file
        ini = "unittest.ini"
        config = configparser.ConfigParser()
        inifname = os.path.join(self.cwd, ini)
        config.read(inifname)
        

        assert raw == config['MINIMAL INFO']['file_name']

        # read how to read the data from the ini file
        read_dtype = config['MINIMAL INFO']['data_type']
        read_array = np.fromfile(fname, dtype=read_dtype)
        read_shape = eval(config['MINIMAL INFO']['shape'])
        
        # reshape read in array
        read_array = read_array.reshape(read_shape)

        if compress:
            # rescale the dataset to the original data
            sc = float(config['COMPRESSION']['scale'])
            of = float(config['COMPRESSION']['offset'])
            assert sc == scale
            assert of == offset

            recovered_data = (read_array - of)/sc
            np.testing.assert_allclose(recovered_data, data.array, rtol=1e-1, atol=1e-2)

            # rescale the original data with scale and offset and compare with what saved
            tmp = data.array * scale + offset
            tmp = np.asarray(tmp, dtype=dtype)
        else:
            tmp = data.array
            
        assert tmp.dtype == read_array.dtype
        
        np.testing.assert_array_equal(tmp, read_array)

class Test_HDF5_utilities(unittest.TestCase):
    def setUp(self) -> None:
        self.path = os.path.join(os.path.abspath(data_dir), '24737_fd_normalised.nxs')

        
        self.dset_path ='/entry1/tomo_entry/data/data'


    def test_print_metadata(self):
        devnull = open(os.devnull, 'w') #suppress stdout
        with patch('sys.stdout', devnull):
            HDF5_utilities.print_metadata(self.path)    


    def test_get_dataset_metadata(self):
        dset_dict = HDF5_utilities.get_dataset_metadata(self.path, self.dset_path)

        dict_by_hand  ={'ndim': 3, 'shape': (91, 135, 160), 'size': 1965600, 'dtype': np.float32, 'compression': None, 'chunks': None, 'is_virtual': False}
        self.assertDictContainsSubset(dict_by_hand,dset_dict)


    def test_read(self):

        data_full = dataexample.SYNCHROTRON_PARALLEL_BEAM_DATA.get()

        # full dataset
        data_read_full = HDF5_utilities.read(self.path, self.dset_path)
        np.testing.assert_allclose(data_full.array,data_read_full)

        # subset of input
        subset = np.s_[44:45,70:90:2,80]
        data_read_subset = HDF5_utilities.read(self.path, self.dset_path, subset)
        self.assertTrue(data_read_subset.dtype == np.float32)
        np.testing.assert_allclose(data_full.array[subset],data_read_subset)

        # read as dtype
        subset = np.s_[44:45,70:90:2,80]
        data_read_dtype = HDF5_utilities.read(self.path, self.dset_path, subset, dtype=np.float64)
        self.assertTrue(data_read_dtype.dtype == np.float64)
        np.testing.assert_allclose(data_full.array[subset],data_read_dtype)


    def test_read_to(self):
        data_full = dataexample.SYNCHROTRON_PARALLEL_BEAM_DATA.get()

        # full dataset
        data_full_out = np.empty_like(data_full.array, dtype=np.float32)
        HDF5_utilities.read_to(self.path, self.dset_path, data_full_out)
        np.testing.assert_allclose(data_full.array,data_full_out)

        # subset of input, continuous output
        subset = np.s_[44:45,70:90:2,80]
        data_subset_out = np.empty((1,10), dtype=np.float32)
        HDF5_utilities.read_to(self.path, self.dset_path, data_subset_out, source_sel=subset)
        np.testing.assert_allclose(data_full.array[subset],data_subset_out)

        # subset of input, continuous output, change of dtype
        subset = np.s_[44:45,70:90:2,80]
        data_subset_out = np.empty((1,10), dtype=np.float64)
        HDF5_utilities.read_to(self.path, self.dset_path, data_subset_out, source_sel=subset)
        np.testing.assert_allclose(data_full.array[subset],data_subset_out)

        # subset of input written to subset of  output
        data_partial_by_hand = np.zeros_like(data_full.array, dtype=np.float32)
        data_partial = np.zeros_like(data_full.array, dtype=np.float32)

        data_partial_by_hand[subset] = data_full.array[subset]

        HDF5_utilities.read_to(self.path, self.dset_path, data_partial, source_sel=subset, dest_sel=subset)
        np.testing.assert_allclose(data_partial_by_hand,data_partial)


class TestZeissReader(unittest.TestCase):

    def test_setup(self):

        reader = ZEISSDataReader()
        self.assertEqual(reader.file_name, None)

        with self.assertRaises(FileNotFoundError):
            reader = ZEISSDataReader(file_name='no-file')
