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
from cil.reconstructors import FBP, Reconstructor
import unittest
import numpy as np
from utils import has_gpu_tigre, has_ipp

try:
    from cil.plugins.tigre import ProjectionOperator as ProjectionOperator
    from cil.plugins.tigre import FBP as FBP_tigre
    from tigre.utilities.filtering import ramp_flat, filter
    has_tigre = True
except ModuleNotFoundError:
    print(  "These reconstructors require the additional package TIGRE\n" +
            "Please install it via conda as tigre from the ccpi channel\n"+
            "Minimal version is 21.01")
    has_tigre = False

has_tigre = has_tigre and has_gpu_tigre()

has_ipp = has_ipp()


class Test_Reconstructor(unittest.TestCase):

    def setUp(self):
        #%% Setup Geometry
        voxel_num_xy = 255
        voxel_num_z = 15

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


        reconstructor = Reconstructor(ad3D)

        #test defaults
        self.assertEqual(id(reconstructor.input),id(ad3D))
        self.assertEqual(reconstructor.image_geometry,ig3D)
        self.assertEqual(reconstructor.backend, 'tigre')

        #test customisation
        ag3D_new = ad3D.copy()
        reconstructor.set_input(ag3D_new)
        self.assertEqual(id(reconstructor.input),id(ag3D_new))

        ag3D_new = ad3D.get_slice(vertical='centre')
        with self.assertRaises(ValueError):
            reconstructor.set_input(ag3D_new)
        
        ig3D.voxel_num_z = 1
        reconstructor.set_image_geometry(ig3D)
        self.assertEqual(reconstructor.image_geometry,ig3D)

        with self.assertRaises(ValueError):
            reconstructor.set_backend('gemma')

        #wrong input on initialisation
        ad3D.reorder('astra')
        with self.assertRaises(ValueError):
            reconstructor = Reconstructor(ad3D)

        with self.assertRaises(TypeError):
            reconstructor = Reconstructor(self.ag3D)

class Test_FBP(Test_Reconstructor, unittest.TestCase):

    @unittest.skipUnless(has_ipp, "IPP not installed")
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

    @unittest.skipUnless(has_tigre and has_ipp, "TIGRE or IPP not installed")
    def test_results(self):
        #create phantom
        kernel_size = self.ag3D.pixel_num_h
        kernel_radius = (kernel_size - 1) // 2
        y, x = np.ogrid[-kernel_radius:kernel_radius+1, -kernel_radius:kernel_radius+1]

        circle1 = [5,0,0] #r,x,y
        dist1 = ((x - circle1[1])**2 + (y - circle1[2])**2)**0.5

        circle2 = [5,80,0] #r,x,y
        dist2 = ((x - circle2[1])**2 + (y - circle2[2])**2)**0.5

        circle3 = [25,0,80] #r,x,y
        dist3 = ((x - circle3[1])**2 + (y - circle3[2])**2)**0.5

        mask1 =(dist1 - circle1[0]).clip(0,1) 
        mask2 =(dist2 - circle2[0]).clip(0,1) 
        mask3 =(dist3 - circle3[0]).clip(0,1) 
        phantom = 1 - np.logical_and(np.logical_and(mask1, mask2),mask3)

        golden_data = self.ig3D.allocate(0)
        for i in range(4):
            golden_data.fill(array=phantom, vertical=7+i)

        Op = ProjectionOperator(self.ig3D, self.ag3D, direct_method='interpolated')
        data = Op.direct(golden_data)
        data2D = data.get_slice(vertical='centre')

        fbp_tigre = FBP_tigre(self.ig3D, self.ag3D)
        reco_tigre = fbp_tigre(data)

        fbp_cil = FBP(data)
        reco_cil = fbp_cil.run()

        #construct TIGRE's filter
        n = 2**fbp_cil.fft_order
        ramp = ramp_flat(n)
        filt = filter('ram_lak',ramp[0],n,1,False)
        fbp_cil.set_filter(filt)
        reco_cil_filter = fbp_cil.run()
        #check close to sim (aliasing makes the difference lareg at edges)
        diff = (golden_data-reco_cil).abs().max()
        self.assertLess(diff, 0.6)

        #check close to tigre
        diff = (reco_tigre-reco_cil).abs().max()
        self.assertLess(diff, 0.006)

        #very small difference expected with the same filter
        diff = (reco_cil_filter-reco_tigre).abs().max()
        self.assertLess(diff, 5e-6)

        #2D
        fbp_cil_2D = FBP(data2D)
        reco_cil_2D = fbp_cil_2D.run()
        diff = (reco_cil_2D - phantom).abs().max()
        self.assertLess(diff, 1)

        #test inplace run
        reco_cil_2D_b = reco_cil_2D.copy()
        reco_cil_2D_b.fill(0)
        fbp_cil_2D.run(out=reco_cil_2D_b)
        diff = (reco_cil_2D - reco_cil_2D_b).abs().max()
        self.assertLess(diff,1e-8)

        #test inplace filter
        data_filtered= data2D.copy()
        fbp_cil_filter_inplace = FBP(data_filtered)
        fbp_cil_filter_inplace.set_filter_inplace(True)
        fbp_cil_filter_inplace.run(out=reco_cil_2D)

        diff = (data_filtered - data2D).abs().mean()
        self.assertGreater(diff,0.5)