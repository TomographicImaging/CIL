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

from cil.framework.framework import ImageGeometry,AcquisitionGeometry
from cil.utilities import dataexample
from cil.utilities import noise
import os, sys

from numpy.testing._private.utils import assert_allclose, assert_array_equal, assert_equal
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from testclass import CCPiTestClass
import platform
import numpy as np

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

        shape_expected = (512,512)
        self.assertEquals(image.shape,shape_expected,msg="Image dimension mismatch")   


    def test_load_BOAT(self):

        image = self.check_load(dataexample.BOAT)

        shape_expected = (512,512)
        self.assertEquals(image.shape,shape_expected,msg="Image dimension mismatch")   


    def test_load_PEPPERS(self):
        image = self.check_load(dataexample.PEPPERS)

        shape_expected = (3,512,512)
        self.assertEquals(image.shape,shape_expected,msg="Image dimension mismatch")   


    def test_load_RAINBOW(self):

        image = self.check_load(dataexample.RAINBOW)

        shape_expected = (3,1353,1194)
        self.assertEquals(image.shape,shape_expected,msg="Image dimension mismatch")   


    def test_load_RESOLUTION_CHART(self):

        image = self.check_load(dataexample.RESOLUTION_CHART)

        shape_expected = (256,256)
        self.assertEquals(image.shape,shape_expected,msg="Image dimension mismatch")   


    def test_load_SIMPLE_PHANTOM_2D(self):

        image = self.check_load(dataexample.SIMPLE_PHANTOM_2D)

        shape_expected = (512,512)
        self.assertEquals(image.shape,shape_expected,msg="Image dimension mismatch")   


    def test_load_SHAPES(self):

        image = self.check_load(dataexample.SHAPES)

        shape_expected = (200,300)
        self.assertEquals(image.shape,shape_expected,msg="Image dimension mismatch")   


    def test_load_SYNCHROTRON_PARALLEL_BEAM_DATA(self):

        image = self.check_load(dataexample.SYNCHROTRON_PARALLEL_BEAM_DATA)

        shape_expected = (91,135,160)
        self.assertEquals(image.shape,shape_expected,msg="Image dimension mismatch")   


    def test_load_SIMULATED_SPHERE_VOLUME(self):

        image = self.check_load(dataexample.SIMULATED_SPHERE_VOLUME)

        ig_expected = ImageGeometry(128,128,128,16,16,16)

        self.assertEquals(ig_expected,image.geometry,msg="Image geometry mismatch")   


    def test_load_SIMULATED_PARALLEL_BEAM_DATA(self):

        image = self.check_load(dataexample.SIMULATED_PARALLEL_BEAM_DATA)

        ag_expected = AcquisitionGeometry.create_Parallel3D()\
                                         .set_panel((128,128),(16,16))\
                                         .set_angles(np.linspace(0,360,300,False))\

        self.assertEquals(ag_expected,image.geometry,msg="Acquisition geometry mismatch")   


    def test_load_SIMULATED_CONE_BEAM_DATA(self):

        image = self.check_load(dataexample.SIMULATED_CONE_BEAM_DATA)

        ag_expected = AcquisitionGeometry.create_Cone3D([0,-20000,0],[0,60000,0])\
                                         .set_panel((128,128),(64,64))\
                                         .set_angles(np.linspace(0,360,300,False))

        self.assertEquals(ag_expected,image.geometry,msg="Acquisition geometry mismatch")   
