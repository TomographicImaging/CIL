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

from cil.framework import AcquisitionGeometry
from cil.reconstructors import FBP
import unittest
import numpy as np
from utils import has_gpu_tigre, has_gpu_astra

try:
    from cil.plugins.tigre import ProjectionOperator as ProjectionOperator
    from cil.plugins.tigre import FBP as FBP_tigre
    has_tigre = True
except ModuleNotFoundError:
    print(  "These reconstructors require the additional package TIGRE\n" +
            "Please install it via conda as tigre from the ccpi channel\n"+
            "Minimal version is 21.01")
    has_tigre = False


has_tigre = has_tigre and has_gpu_tigre()


class Test_Reconstructor(unittest.TestCase):

    def setUp(self):
        #%% Setup Geometry
        voxel_num_xy = 255
        voxel_num_z = 15
        cs_ind = (voxel_num_z-1)//2

        mag = 2
        src_to_obj = 50
        src_to_det = src_to_obj * mag

        pix_size = 0.2
        det_pix_x = voxel_num_xy
        det_pix_y = voxel_num_z

        num_projections = 1000
        angles = np.linspace(0, 360, num=num_projections, endpoint=False)

        self.ag = AcquisitionGeometry.create_Cone2D([0,-src_to_obj],[0,src_to_det-src_to_obj])\
                                           .set_angles(angles)\
                                           .set_panel(det_pix_x, pix_size)\
                                           .set_labels(['angle','horizontal'])

        self.ig = self.ag.get_ImageGeometry()

        self.ag3D = AcquisitionGeometry.create_Cone3D([0,-src_to_obj,0],[0,src_to_det-src_to_obj,0])\
                                     .set_angles(angles)\
                                     .set_panel((det_pix_x,det_pix_y), (pix_size,pix_size))\
                                     .set_labels(['angle','vertical','horizontal'])        
        self.ig3D = self.ag3D.get_ImageGeometry()

    def test_setup(self):
        ad3D = self.ag3D.allocate('random')
        ig3D = self.ag3D.get_ImageGeometry()
        fbp = FBP(ad3D)

        #test defaults
        self.assertEqual(id(fbp.input),id(ad3D))
        self.assertEqual(fbp.image_geometry,ig3D)
        self.assertEqual(fbp.backend, 'tigre')

        #test customisation
        ag3D_new = ad3D.copy()
        fbp.set_input(ag3D_new)
        self.assertEqual(id(fbp.input),id(ag3D_new))

        ag3D_new = ad3D.get_slice(vertical='centre')
        with self.assertRaises(ValueError):
            fbp.set_input(ag3D_new)
        
        ig3D.voxel_num_z = 1
        fbp.set_image_geometry(ig3D)
        self.assertEqual(fbp.image_geometry,ig3D)

        with self.assertRaises(ValueError):
            fbp.set_backend('gemma')

        #wrong input on initialisation
        ad3D.reorder('astra')
        with self.assertRaises(ValueError):
            fbp = FBP(ad3D)

        with self.assertRaises(TypeError):
            fbp = FBP(self.ag3D)

class Test_FBP(Test_Reconstructor, unittest.TestCase):
    
    def test_setup(self):
        ad3D = self.ag3D.allocate('random')
        ig3D = self.ag3D.get_ImageGeometry()
        fbp = FBP(ad3D)

        #test defaults
        self.assertEqual(fbp.filter, 'ram-lak')
        self.assertEqual(fbp.fft_order, 9)
        self.assertFalse(fbp.filter_inplace)

        filter = fbp.get_filter_array()
        self.assertEqual(type(filter), np.ndarray)
        self.assertEqual(len(filter), 2**9)
        self.assertEqual(filter[0], 0)
        self.assertEqual(filter[256],1.0)
        self.assertEqual(filter[1],filter[511])


        #test customisation
        filter_new =filter *0.5
        fbp.set_filter(filter_new)
        self.assertEqual(fbp.filter, 'custom')
        filter = fbp.get_filter_array()
        np.testing.assert_array_equal(filter,filter_new)

        fbp.set_fft_order(10)
        self.assertEqual(fbp.fft_order, 10)

        #change in filter length resets filter type
        self.assertEqual(fbp.filter, 'ram-lak')

        fbp.set_filter_inplace(True)
        self.assertTrue(fbp.filter_inplace)

        #test errors
        with self.assertRaises(ValueError):
            fbp.set_filter('gemma')
        with self.assertRaises(ValueError):
            fbp.set_filter(filter[1:-1])

        with self.assertRaises(TypeError):
            fbp.set_filter_inplace('gemma')

        with self.assertRaises(ValueError):
            fbp.set_fft_order(2)
