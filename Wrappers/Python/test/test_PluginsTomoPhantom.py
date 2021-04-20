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

from cil.framework import ImageGeometry, AcquisitionGeometry
import numpy as np
import unittest
try:
    import tomophantom
    from cil.plugins import TomoPhantom
    has_tomophantom = True
except ImportError as ie:
    # raise ImportError(ie + "\n\n" + 
    #                   "This plugin requires the additional package ccpi-regularisation\n" +
    #                   "Please install it via conda as ccpi-regularisation from the ccpi channel\n"+
    #                   "Minimal version is 20.04")
    has_tomophantom = False


class TestTomoPhantom2D(unittest.TestCase):
    def setUp(self):
        
        N=128
        angles = np.linspace(0, 360, 50, True, dtype=np.float32)
        offset = 0.4
        ag = AcquisitionGeometry.create_Cone2D((offset,-100), (offset,100))
        ag.set_panel(N)
        
        ag.set_angles(angles, angle_unit=AcquisitionGeometry.DEGREE)
        ig = ag.get_ImageGeometry()
        self.ag = ag
        self.ig = ig
        self.N = N

    def tearDown(self):
        pass
    @unittest.skipUnless(has_tomophantom, 'Please install TomoPhantom')
    def test_2D(self):
        model = 1

        phantom = TomoPhantom.get_ImageData(model, self.ig)

        assert phantom.geometry.channels == 1
        assert phantom.shape == (self.N,self.N)

    @unittest.skipUnless(has_tomophantom, 'Please install TomoPhantom')
    def test_MC2D_wrong_model(self):
        model = 1
        ag = self.ag.copy()
        ag.set_channels(3)
        ig = ag.get_ImageGeometry()
        try:
            phantom = TomoPhantom.get_ImageData(model, ig)
            assert False
        except ValueError as ve:
            assert True
    @unittest.skipUnless(has_tomophantom, 'Please install TomoPhantom')
    def test_MC2D_wrong_amount_channels(self):
        model = 1
        ag = self.ag.copy()
        ag.set_channels(3)
        ig = ag.get_ImageGeometry()
        try:
            phantom = TomoPhantom.get_ImageData(model, ig)
            assert False
        except ValueError as ve:
            assert True
    @unittest.skipUnless(has_tomophantom, 'Please install TomoPhantom')
    def test_MC2D(self):
        model = 100
        self.ag.set_channels(3)
        phantom = TomoPhantom.get_ImageData(model, self.ig)
        assert phantom.geometry.channels == self.ig.channels
        assert phantom.shape == self.ig.shape
    @unittest.skipUnless(has_tomophantom, 'Please install TomoPhantom')
    def test_is_model_temporal(self):
        model = 100
        assert TomoPhantom.is_model_temporal(model, num_dims=2)
    @unittest.skipUnless(has_tomophantom, 'Please install TomoPhantom')
    def test_get_model_num_channels(self):
        model = 100
        N = TomoPhantom.is_model_temporal(model, num_dims=2)
        assert N >= 1

class TestTomoPhantom3D(unittest.TestCase):
    def setUp(self):
        
        N=128
        angles = np.linspace(0, 360, 50, True, dtype=np.float32)
        offset = 0.4
        ag = AcquisitionGeometry.create_Cone3D((offset,-100,0), (offset,100,0))
        ag.set_panel((N,N/2))
        
        ag.set_angles(angles, angle_unit=AcquisitionGeometry.DEGREE)
        ig = ag.get_ImageGeometry()
        self.ag = ag
        self.ig = ig
        self.N = N

    def tearDown(self):
        pass
    @unittest.skipUnless(has_tomophantom, 'Please install TomoPhantom')
    def test_3D(self):
        model = 1

        phantom = TomoPhantom.get_ImageData(model, self.ig)
        assert phantom.geometry.channels == 1
        assert phantom.shape == (self.N/2,self.N, self.N)

    @unittest.skipUnless(has_tomophantom, 'Please install TomoPhantom')
    def test_MC3D_wrong_model(self):
        model = 1
        ag = self.ag.copy()
        ag.set_channels(3)
        ig = ag.get_ImageGeometry()
        try:
            phantom = TomoPhantom.get_ImageData(model, ig)
            assert False
        except ValueError as ve:
            assert True
    @unittest.skipUnless(has_tomophantom, 'Please install TomoPhantom')
    def test_MC3D_wrong_amount_channels(self):
        model = 1
        ag = self.ag.copy()
        ag.set_channels(3)
        ig = ag.get_ImageGeometry()
        try:
            phantom = TomoPhantom.get_ImageData(model, ig)
            assert False
        except ValueError as ve:
            assert True
    @unittest.skipUnless(has_tomophantom, 'Please install TomoPhantom')
    def test_MC3D(self):
        model = 100
        ag = self.ag.copy()
        ag.set_channels(5)
        ig = ag.get_ImageGeometry()
        phantom = TomoPhantom.get_ImageData(model, ig)
        
        assert phantom.geometry.channels == ig.channels
        assert phantom.shape == ig.shape
    @unittest.skipUnless(has_tomophantom, 'Please install TomoPhantom')
    def test_is_model_temporal(self):
        model = 100
        assert TomoPhantom.is_model_temporal(model, num_dims=3)
    @unittest.skipUnless(has_tomophantom, 'Please install TomoPhantom')
    def test_get_model_num_channels(self):
        model = 100
        N = TomoPhantom.is_model_temporal(model, num_dims=3)
        assert N >= 1
