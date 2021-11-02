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
from cil.reconstructors import Reconstructor
from cil.reconstructors import FBP
from cil.utilities.dataexample import SIMULATED_CONE_BEAM_DATA, SIMULATED_SPHERE_VOLUME

import unittest
from scipy.fftpack  import fft, ifft
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

        self.ad3D = self.ag3D.allocate('random')
        self.ig3D = self.ag3D.get_ImageGeometry()
        
    def test_defaults(self):
        reconstructor = Reconstructor(self.ad3D)
        self.assertEqual(id(reconstructor.input),id(self.ad3D))
        self.assertEqual(reconstructor.image_geometry,self.ig3D)
        self.assertEqual(reconstructor.backend, 'tigre')

    def test_set_input(self):
        reconstructor = Reconstructor(self.ad3D)
        self.assertEqual(id(reconstructor.input),id(self.ad3D))

        ag3D_new = self.ad3D.copy()
        reconstructor.set_input(ag3D_new)
        self.assertEqual(id(reconstructor.input),id(ag3D_new))

        ag3D_new = self.ad3D.get_slice(vertical='centre')
        with self.assertRaises(ValueError):
            reconstructor.set_input(ag3D_new)

        with self.assertRaises(TypeError):
            reconstructor = Reconstructor(self.ag3D)

    def test_set_image_data(self):
        reconstructor = Reconstructor(self.ad3D)

        self.ig3D.voxel_num_z = 1
        reconstructor.set_image_geometry(self.ig3D)
        self.assertEqual(reconstructor.image_geometry,self.ig3D)

    def test_set_backend(self):
        reconstructor = Reconstructor(self.ad3D)

        with self.assertRaises(ValueError):
            reconstructor.set_backend('gemma')

        self.ad3D.reorder('astra')
        with self.assertRaises(ValueError):
            reconstructor = Reconstructor(self.ad3D)

class Test_FBP(unittest.TestCase):

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

        self.ad3D = self.ag3D.allocate('random')
        self.ig3D = self.ag3D.get_ImageGeometry()
        
    @unittest.skipUnless(has_ipp, "IPP not installed")
    def test_defaults(self):
        fbp = FBP(self.ad3D)

        self.assertEqual(fbp.filter, 'ram-lak')
        self.assertEqual(fbp.fft_order, 9)
        self.assertFalse(fbp.filter_inplace)

        filter = fbp.get_filter_array()
        self.assertEqual(type(filter), np.ndarray)
        self.assertEqual(len(filter), 2**9)
        self.assertEqual(filter[0], 0)
        self.assertEqual(filter[256],1.0)
        self.assertEqual(filter[1],filter[511])

    def test_set_filter(self):
        fbp = FBP(self.ad3D)

        with self.assertRaises(ValueError):
            fbp.set_filter("gemma")

        filter = fbp.get_filter_array()
        filter_new =filter *0.5
        fbp.set_filter(filter_new)
        self.assertEqual(fbp.filter, 'custom')
        filter = fbp.get_filter_array()
        np.testing.assert_array_equal(filter,filter_new)

        fbp.set_fft_order(10)
        self.assertEqual(fbp.filter, 'ram-lak')

        with self.assertRaises(ValueError):
            fbp.set_filter(filter[1:-1])

    def test_set_fft_order(self):
        fbp = FBP(self.ad3D)
        fbp.set_fft_order(10)
        self.assertEqual(fbp.fft_order, 10)

        with self.assertRaises(ValueError):
            fbp.set_fft_order(2)

    def test_set_filter_inplace(self):
        fbp = FBP(self.ad3D)
        fbp.set_filter_inplace(True)
        self.assertTrue(fbp.filter_inplace)

        with self.assertRaises(TypeError):
            fbp.set_filter_inplace('gemma')

    @unittest.skipUnless(has_ipp, "IPP not installed")
    def test_filtering(self):
        ag = AcquisitionGeometry.create_Cone3D([0,-1,0],[0,2,0])\
            .set_panel([64,3],[0.1,0.1])\
            .set_angles([0,90])

        ad = ag.allocate('random',seed=0)

        fbp = FBP(ad)
        out1 = ad.copy()
        fbp.pre_filtering(out1)

        #by hand
        filter = fbp.get_filter_array()
        weights = fbp.calculate_weights()
        pad0 = (len(filter)-ag.pixel_num_h)//2
        pad1 = len(filter)-ag.pixel_num_h-pad0

        out2 = ad.array.copy()
        out2*=weights
        for i in range(2):
            proj_padded = np.zeros((ag.pixel_num_v,len(filter)))
            proj_padded[:,pad0:-pad1] = out2[i]
            filtered_proj=fft(proj_padded,axis=-1)
            filtered_proj*=filter
            filtered_proj=ifft(filtered_proj,axis=-1)
            out2[i]=np.real(filtered_proj)[:,pad0:-pad1]

        diff = (out1-out2).abs().max()
        self.assertLess(diff, 1e-5)

    def test_weights(self):
        ag = AcquisitionGeometry.create_Cone3D([0,-1,0],[0,2,0])\
            .set_panel([3,4],[0.1,0.2])\
            .set_angles([0,90])
        ad = ag.allocate(0)

        fbp = FBP(ad)
        weights = fbp.calculate_weights()

        scaling =  7.5 * np.pi
        weights_new = np.ones_like(weights)

        det_size_x = ag.pixel_size_h*ag.pixel_num_h
        det_size_y = ag.pixel_size_v*ag.pixel_num_v

        ray_length_z = 3
        for j in range(4):
            ray_length_y = -det_size_y/2 +  ag.pixel_size_v * (j+0.5)
            for i in range(3):
                ray_length_x = -det_size_x/2 +  ag.pixel_size_h * (i+0.5)
                ray_length = (ray_length_x**2+ray_length_y**2+ray_length_z**2)**0.5
                weights_new[j,i] = scaling*ray_length_z/ray_length

        diff = np.max(np.abs(weights - weights_new))
        self.assertLess(diff, 1e-5)
  
class Test_FBP_results(unittest.TestCase):
    
    def setUp(self):

        self.acq_data = SIMULATED_CONE_BEAM_DATA.get()
        self.img_data = SIMULATED_SPHERE_VOLUME.get()

        self.acq_data=np.log(self.acq_data)
        self.acq_data*=-1.0

        self.ig = self.img_data.geometry
        self.ag = self.acq_data.geometry


    @unittest.skipUnless(has_tigre and has_ipp, "TIGRE or IPP not installed")
    def test_results_3D(self):

        fbp = FBP(self.acq_data)

        reco = fbp.run()
        np.testing.assert_allclose(reco.as_array(), self.img_data.as_array(),atol=1e-3)    

        reco2 = reco.copy()
        reco2.fill(0)
        fbp.run(out=reco2)
        np.testing.assert_array_equal(reco.as_array(), reco2.as_array())  


    @unittest.skipUnless(has_tigre and has_ipp, "TIGRE or IPP not installed")
    def test_results_2D(self):

        data2D = self.acq_data.get_slice(vertical='centre')
        img_data2D = self.img_data.get_slice(vertical='centre')

        fbp = FBP(data2D)
        reco = fbp.run()
        np.testing.assert_allclose(reco.as_array(), img_data2D.as_array(),atol=1e-3)    

        reco2 = reco.copy()
        reco2.fill(0)
        fbp.run(out=reco2)
        np.testing.assert_array_equal(reco.as_array(), reco2.as_array()) 


    @unittest.skipUnless(has_tigre and has_ipp, "TIGRE or IPP not installed")
    def test_results_with_tigre(self):

        fbp_tigre = FBP_tigre(self.ig, self.ag)
        reco_tigre = fbp_tigre(self.acq_data)
    
        #fbp CIL with TIGRE's filter
        fbp_cil = FBP(self.acq_data)
        n = 2**fbp_cil.fft_order
        ramp = ramp_flat(n)
        filt = filter('ram_lak',ramp[0],n,1,False)

        fbp_cil = FBP(self.acq_data)
        fbp_cil.set_filter(filt)
        reco_cil = fbp_cil.run()

        #with the same filter results should be virtually identical
        np.testing.assert_array_equal(reco_cil.as_array(), reco_tigre.as_array(),atol=1e-8) 


    @unittest.skipUnless(has_tigre and has_ipp, "TIGRE or IPP not installed")
    def test_results_inplace_filtering(self):

        fbp_cil_2D = FBP(self.acq_data)
        reco_cil_2D = fbp_cil_2D.run()

        data_filtered= self.acq_data.copy()
        fbp_cil_filter_inplace = FBP(data_filtered)
        fbp_cil_filter_inplace.set_filter_inplace(True)
        fbp_cil_filter_inplace.run(out=reco_cil_2D)

        diff = (data_filtered - self.acq_data).abs().mean()
        self.assertGreater(diff,0.8)