# -*- coding: utf-8 -*-
#  Copyright 2018 - 2022 United Kingdom Research and Innovation
#  Copyright 2018 - 2022 The University of Manchester
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

import unittest
from utils import initialise_tests
from cil.framework import AcquisitionGeometry
import numpy as np
import os
from cil.framework import ImageGeometry
from cil.io import TXRMDataReader, NEXUSDataReader
from cil.io import TIFFWriter, TIFFStackReader
from cil.processors import Slicer
from utils import has_astra, has_nvidia
from cil.utilities.dataexample import data_dir
from cil.utilities.quality_measures import mae, mse, psnr
from cil.utilities import dataexample
import shutil
import logging
import glob

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


logging.info ("has_astra {}".format(has_astra))
logging.info ("has_wget {}".format(has_wget))
logging.info ("has_olefile {}".format(has_olefile))
logging.info ("has_dxchange {}".format(has_dxchange))
logging.info ("has_file {}".format(has_file))

if not has_file:
    logging.info("This unittest requires the walnut Zeiss dataset saved in {}".format(data_dir))


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
        # self.logger = logging.getLogger('cil.io')
        # self.logger.setLevel(logging.DEBUG)
        self.cwd = os.path.join(os.getcwd(), 'tifftest')
        os.mkdir(self.cwd)


    def tearDown(self) -> None:
        shutil.rmtree(self.cwd)
        

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
        self.TIFF_compression_test(0)
    
    def test_TIFF_compression3D_1(self):
        self.TIFF_compression_test(8)

    def test_TIFF_compression3D_2(self):
        self.TIFF_compression_test(16)

    def test_TIFF_compression3D_3(self):
        with self.assertRaises(ValueError) as context:
            self.TIFF_compression_test(12)
            
    def test_TIFF_compression4D_0(self):
        self.TIFF_compression_test(0,2)
        
    def test_TIFF_compression4D_1(self):
        self.TIFF_compression_test(8,2)

    def test_TIFF_compression4D_2(self):
        self.TIFF_compression_test(16,2)
    
    def test_TIFF_compression4D_3(self):
        with self.assertRaises(ValueError) as context:
            self.TIFF_compression_test(12,2)

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

        from cil.io import utilities
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
        else:
            tmp = data.array
        
        assert tmp.dtype == read_array.dtype
        
        np.testing.assert_array_equal(tmp, read_array)

