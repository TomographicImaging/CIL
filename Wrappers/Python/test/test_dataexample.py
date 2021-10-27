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

import numpy
from cil.utilities import dataexample
from cil.utilities import noise
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from testclass import CCPiTestClass
import unittest
import platform

class TestTestData(CCPiTestClass):
    def test_noise_gaussian(self):
        camera = dataexample.CAMERA.get()
        noisy_camera = noise.gaussian(camera, seed=1)
        norm = (camera - noisy_camera).norm()
        decimal = 4
        if platform.system() == 'Darwin':
            decimal = 2
        self.assertAlmostEqual(norm, 48.881268, places=decimal)

    def test_load_CAMERA(self):

        
        res = False
        try:
            image = dataexample.CAMERA.get()
            if (image.shape[0] == 512) and (image.shape[1] == 512):
                res = True
            else:
                self.assertFalse(msg="Image dimension mismatch")
        except FileNotFoundError:
            self.assertFalse(msg="File not found")
        except:
            self.assertFalse(msg="Failed to load file")

        self.assertTrue(res)


    def test_load_BOAT(self):
        
        res = False
        try:
            image = dataexample.BOAT.get()
            if (image.shape[0] == 512) and (image.shape[1] == 512):
                res = True
            else:
                self.assertFalse(msg="Image dimension mismatch")
        except FileNotFoundError:
            self.assertFalse(msg="File not found")
        except:
            self.assertFalse(msg="Failed to load file")

        self.assertTrue(res)

    def test_load_PEPPERS(self):
        
        res = False
        try:
            image = dataexample.PEPPERS.get()
            if (image.shape[0] == 3) and (image.shape[1] == 512) and (image.shape[2] == 512):
                res = True
            else:
                self.assertFalse(msg="Image dimension mismatch")
        except FileNotFoundError:
            self.assertFalse(msg="File not found")
        except:
            self.assertFalse(msg="Failed to load file")

        self.assertTrue(res)

    def test_load_RAINBOW(self):
        
        res = False
        try:
            image = dataexample.RAINBOW.get()
            if (image.shape[0] == 3) and (image.shape[1] == 1353) and (image.shape[2] == 1194):
                res = True
            else:
                self.assertFalse(msg="Image dimension mismatch")
        except FileNotFoundError:
            self.assertFalse(msg="File not found")
        except:
            self.assertFalse(msg="Failed to load file")

        self.assertTrue(res)
    def test_load_RESOLUTION_CHART(self):
        
        res = False
        try:
            image = dataexample.RESOLUTION_CHART.get()
            if (image.shape[0] == 256) and (image.shape[1] == 256):
                res = True
            else:
                self.assertFalse(msg="Image dimension mismatch")
        except FileNotFoundError:
            self.assertFalse(msg="File not found")
        except:
            self.assertFalse(msg="Failed to load file")

        self.assertTrue(res)

    def test_load_SIMPLE_PHANTOM_2D(self):
        res = False
        try:
            image = dataexample.SIMPLE_PHANTOM_2D.get()
            if (image.shape[0] == 512) and (image.shape[1] == 512):
                res = True
            else:
                self.assertFalse(msg="Image dimension mismatch")
        except FileNotFoundError:
            self.assertFalse(msg="File not found")
        except:
            self.assertFalse(msg="Failed to load file")

        self.assertTrue(res)

    def test_load_SHAPES(self):
        res = False
        try:
            image = dataexample.SHAPES.get()
            if (image.shape[0] == 200) and (image.shape[1] == 300):
                res = True
            else:
                self.assertFalse(msg="Image dimension mismatch")
        except FileNotFoundError:
            self.assertFalse(msg="File not found")
        except:
            self.assertFalse(msg="Failed to load file")

        self.assertTrue(res)

    def test_load_SYNCHROTRON_PARALLEL_BEAM_DATA(self):
        res = False
        try:
            image = dataexample.SYNCHROTRON_PARALLEL_BEAM_DATA.get()
            if (image.shape[0] == 91) and (image.shape[1] == 135) and\
                (image.shape[2] == 160):
                res = True
            else:
                self.assertFalse(msg="Image dimension mismatch")
        except FileNotFoundError:
            self.assertFalse(msg="File not found")
        except:
            self.assertFalse(msg="Failed to load file")

        self.assertTrue(res)
        
    def test_load_SIMULATED_SPHERE_VOLUME(self):
        res = False
        try:
            image = dataexample.SIMULATED_SPHERE_VOLUME.get()
            if (image.shape[0] == 128) and (image.shape[1] == 128) and\
                (image.shape[2] == 128):
                res = True
            else:
                self.assertFalse(msg="Image dimension mismatch")
        except FileNotFoundError:
            self.assertFalse(msg="File not found")
        except:
            self.assertFalse(msg="Failed to load file")

        self.assertTrue(res)

    def test_load_SIMULATED_PARALLEL_BEAM_DATA(self):
        res = False
        try:
            image = dataexample.SIMULATED_PARALLEL_BEAM_DATA.get()
            if (image.shape[0] == 300) and (image.shape[1] == 128) and\
                (image.shape[2] == 128):
                res = True
            else:
                self.assertFalse(msg="Image dimension mismatch")
        except FileNotFoundError:
            self.assertFalse(msg="File not found")
        except:
            self.assertFalse(msg="Failed to load file")

        vol = dataexample.SIMULATED_SPHERE_VOLUME.get()
        self.assertTrue(vol.geometry == image.geometry.get_ImageGeometry())

        self.assertTrue(res)

    def test_load_SIMULATED_CONE_BEAM_DATA(self):
        res = False
        try:
            image = dataexample.SIMULATED_CONE_BEAM_DATA.get()
            if (image.shape[0] == 300) and (image.shape[1] == 128) and\
                (image.shape[2] == 128):
                res = True
            else:
                self.assertFalse(msg="Image dimension mismatch")
        except FileNotFoundError:
            self.assertFalse(msg="File not found")
        except:
            self.assertFalse(msg="Failed to load file")

        vol = dataexample.SIMULATED_SPHERE_VOLUME.get()
        self.assertTrue(vol.geometry == image.geometry.get_ImageGeometry())

        self.assertTrue(res)