# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library (CIL) developed by CCPi 
#   (Collaborative Computational Project in Tomographic Imaging), with 
#   substantial contributions by UKRI-STFC and University of Manchester.

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
from cil.framework import AcquisitionData, AcquisitionGeometry
import numpy as np
import os
import olefile
from cil.framework import ImageGeometry
from cil.io import TXRMDataReader, NEXUSDataReader
has_astra = True
try:
    from cil.astra.processors import FBP
except ImportError as ie:
    has_astra = False
from cil.utilities.dataexample import data_dir
filename = os.path.join(data_dir, "valnut_tomo-A.txrm")
has_file = os.path.isfile(filename)

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
has_prerequisites = has_olefile and has_dxchange and has_astra and has_file \
    and has_wget
import wget


from cil.utilities.quality_measures import mae, mse, psnr

print ("has_astra",has_astra)
print ("has_wget",has_wget)
print ("has_olefile",has_olefile)
print ("has_dxchange",has_dxchange)
print ("has_file",has_file)

if not has_file:
    print("This unittest requires the walnut Zeiss dataset saved in {}".format(data_dir))

class TestTXRMDataReader(unittest.TestCase):
    
    def setUp(self):
        print ("has_astra",has_astra)
        print ("has_wget",has_wget)
        print ("has_olefile",has_olefile)
        print ("has_dxchange",has_dxchange)
        print ("has_file",has_file)
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
    
    @unittest.skipIf(True, 'skip test by default')
    def test_not_run_test(self):
        print("run test Zeiss Reader")
        self.assertTrue(True)

    @unittest.skipIf(not has_prerequisites, "Prerequisites not met")
    def test_read_and_reconstruct_2D(self):
        print (type(self.data))

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
        fbpalg = FDK(ig2d,data2d.geometry)
        fbpalg.set_input(data2d)
        
        recfbp = fbpalg.get_output()
        
        wget.download('https://www.ccpi.ac.uk/sites/www.ccpi.ac.uk/files/walnut_slice512.nxs',
                      out=data_dir)
        fname = os.path.join(data_dir, 'walnut_slice512.nxs')
        reader = NEXUSDataReader()
        reader.set_up(file_name=fname)
        gt = reader.read()

        qm = mse(gt, recfbp)
        print ("MSE" , qm )

        np.testing.assert_almost_equal(qm, 0, decimal=3)
        fname = os.path.join(data_dir, 'walnut_slice512.nxs')
        os.remove(fname)

