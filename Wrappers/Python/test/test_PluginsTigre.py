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

from cil.framework import AcquisitionGeometry, ImageGeometry
import unittest
import numpy as np

try:
    import tigre
    has_tigre = True
except ModuleNotFoundError:
    print(  "This plugin requires the additional package TIGRE\n" +
            "Please install it via conda as tigre from the ccpi channel\n"+
            "Minimal version is 21.01")
    has_tigre = False
else:
    from cil.plugins.tigre import CIL2TIGREGeometry
    from cil.plugins.tigre import ProjectionOperator
    from cil.plugins.tigre import FBP

try:
    import astra
    has_astra = True
except ModuleNotFoundError:
    has_astra = False
else:
    from cil.plugins.astra.operators import ProjectionOperator as AstraProjectionOperator
    from cil.plugins.astra.processors import FBP as AstraFBP

class Test_convert_geometry(unittest.TestCase):
    def setUp(self): 
        self.num_pixels_x = 128
        self.num_pixels_y = 3
        self.pixel_size_x = 0.1
        self.pixel_size_y = 0.2
        
        self.angles_deg = np.asarray([0,90.0,180.0], dtype='float32')
        self.angles_rad = self.angles_deg * np.pi /180.0

    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_cone2D(self):

        ag = AcquisitionGeometry.create_Cone2D(source_position=[0,-2], detector_position=[0,1])\
                                     .set_angles(self.angles_rad, angle_unit='radian')\
                                     .set_labels(['angle','horizontal'])\
                                     .set_panel(self.num_pixels_x, self.pixel_size_x)

        ig = ag.get_ImageGeometry()
        ig.voxel_num_y = 50
        ig.voxel_size_y /= 2

        angles_rad = np.array([-np.pi/2, -np.pi, -3 *np.pi/2])

        #2D cone
        tg_geometry, tg_angles = CIL2TIGREGeometry.getTIGREGeometry(ig, ag)
        np.testing.assert_allclose(tg_geometry.DSD, ag.dist_center_detector + ag.dist_source_center)
        np.testing.assert_allclose(tg_geometry.DSO, ag.dist_source_center)
        np.testing.assert_allclose(tg_angles, angles_rad)
        np.testing.assert_allclose(tg_geometry.dDetector, ag.config.panel.pixel_size[::-1])
        np.testing.assert_allclose(tg_geometry.nDetector, ag.config.panel.num_pixels[::-1])
        np.testing.assert_allclose(tg_geometry.sDetector, tg_geometry.dDetector * tg_geometry.nDetector)
        np.testing.assert_allclose(tg_geometry.rotDetector,0)
        np.testing.assert_allclose(tg_geometry.offDetector,0)
        np.testing.assert_allclose(tg_geometry.offOrigin,0)

        mag = ag.magnification
        np.testing.assert_allclose(tg_geometry.nVoxel, [1,50,128])
        np.testing.assert_allclose(tg_geometry.dVoxel, [0.1/mag,0.05/mag,0.1/mag])

    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_cone3D_simple(self):
        ag = AcquisitionGeometry.create_Cone3D(source_position=[0,-2,0], detector_position=[0,1,0])\
                                      .set_angles(self.angles_deg, angle_unit='degree')\
                                      .set_labels(['vertical', 'angle','horizontal'])\
                                      .set_panel((self.num_pixels_x,self.num_pixels_y), (self.pixel_size_x,self.pixel_size_y))

        ig = ag.get_ImageGeometry()
        ig.voxel_num_y = 50
        ig.voxel_size_y /= 2

        angles_rad = np.array([-np.pi/2, -np.pi, -3 *np.pi/2])

        tg_geometry, tg_angles = CIL2TIGREGeometry.getTIGREGeometry(ig, ag)

        np.testing.assert_allclose(tg_geometry.DSD, ag.dist_center_detector + ag.dist_source_center)
        np.testing.assert_allclose(tg_geometry.DSO, ag.dist_source_center)
        np.testing.assert_allclose(tg_angles, angles_rad)
        np.testing.assert_allclose(tg_geometry.dDetector, ag.config.panel.pixel_size[::-1])
        np.testing.assert_allclose(tg_geometry.nDetector, ag.config.panel.num_pixels[::-1])
        np.testing.assert_allclose(tg_geometry.sDetector, tg_geometry.dDetector * tg_geometry.nDetector)
        np.testing.assert_allclose(tg_geometry.rotDetector,0)
        np.testing.assert_allclose(tg_geometry.offDetector,0)
        np.testing.assert_allclose(tg_geometry.offOrigin,0)

        mag = ag.magnification
        np.testing.assert_allclose(tg_geometry.nVoxel, [3,50,128])
        np.testing.assert_allclose(tg_geometry.dVoxel, [0.2/mag,0.05/mag,0.1/mag])

    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_cone3D_cofr(self):

        #3, 4, 5 triangle for source + object
        ag = AcquisitionGeometry.create_Cone3D(source_position=[0,-4,0], detector_position=[0,4,0], rotation_axis_position=[3,0, 0])\
                                      .set_angles(self.angles_deg, angle_unit='degree')\
                                      .set_labels(['vertical', 'angle','horizontal'])\
                                      .set_panel((self.num_pixels_x,self.num_pixels_y), (self.pixel_size_x,self.pixel_size_y))
                                      

        ig = ag.get_ImageGeometry()
        ig.voxel_num_y = 50
        ig.voxel_size_y /= 2

        tg_geometry, tg_angles= CIL2TIGREGeometry.getTIGREGeometry(ig, ag)

        np.testing.assert_allclose(tg_geometry.DSO, ag.dist_source_center)

        yaw = np.arcsin(3./5.)
        det_rot = np.array([0,0,yaw])
        np.testing.assert_allclose(tg_geometry.rotDetector,det_rot)

        offset = 4 * 6 /5
        det_offset = np.array([0,-offset,0])
        np.testing.assert_allclose(tg_geometry.offDetector,det_offset)
    
        s2d = ag.dist_center_detector + ag.dist_source_center - 6 * 3 /5   
        np.testing.assert_allclose(tg_geometry.DSD, s2d)

        angles_rad = np.array([-np.pi/2, -np.pi, -3 *np.pi/2]) - yaw


        np.testing.assert_allclose(tg_angles, angles_rad)
        np.testing.assert_allclose(tg_geometry.dDetector, ag.config.panel.pixel_size[::-1])
        np.testing.assert_allclose(tg_geometry.nDetector, ag.config.panel.num_pixels[::-1])
        np.testing.assert_allclose(tg_geometry.sDetector, tg_geometry.dDetector * tg_geometry.nDetector)
        np.testing.assert_allclose(tg_geometry.offOrigin,0)

        mag = ag.magnification
        np.testing.assert_allclose(tg_geometry.nVoxel, [3,50,128])
        np.testing.assert_allclose(tg_geometry.dVoxel, [0.2/mag,0.05/mag,0.1/mag])

class Test_ProjectionOperator(unittest.TestCase):
    def setUp(self): 
        
        N = 3
        angles = np.linspace(0, np.pi, 2, dtype='float32')

        self.ag = AcquisitionGeometry.create_Cone2D([0,-100],[0,200])\
                                .set_angles(angles, angle_unit='radian')\
                                .set_panel(N, 0.1)\
                                .set_labels(['angle', 'horizontal'])
        
        self.ig = self.ag.get_ImageGeometry()
        self.Op = ProjectionOperator(self.ig, self.ag)

        self.ag3D = AcquisitionGeometry.create_Cone3D([0,-100,0],[0,200,0])\
                                .set_angles(angles, angle_unit='radian')\
                                .set_panel((N,N), (0.1,0.1))\
                                .set_labels(['angle', 'vertical', 'horizontal'])
        
        self.ig3D = self.ag3D.get_ImageGeometry()
        self.Op3D = ProjectionOperator(self.ig3D, self.ag3D)

    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_norm(self):
        n = self.Op.norm()
        self.assertAlmostEqual(n, 0.08165, places=3)

        n3D = self.Op3D.norm()
        self.assertAlmostEqual(n3D, 0.08165, places=3)

    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_bp(self):
        gold = np.zeros((3,3))
        gold.fill(2/30)

        ad = self.ag.allocate(1)
        res = self.Op.adjoint(ad)   
        self.assertEqual(res.shape, self.ig.shape)
        np.testing.assert_allclose(res.as_array(),gold,atol=1e-6)

        res = self.ig.allocate(None)
        self.Op.adjoint(ad, out=res)   
        self.assertEqual(res.shape, self.ig.shape)
        np.testing.assert_allclose(res.as_array(),gold,atol=1e-6)

    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_fp(self):
        gold = np.zeros((2,3))
        gold.fill(0.1000017)

        data = self.ig.allocate(1)
        res = self.Op.direct(data)
        self.assertEqual(res.shape, self.ag.shape)
        np.testing.assert_allclose(res.as_array(),gold,atol=1e-6)

        res = self.ag.allocate(None)
        self.Op.direct(data, out=res)
        self.assertEqual(res.shape, self.ag.shape)
        np.testing.assert_allclose(res.as_array(),gold,atol=1e-6)

class Test_results(unittest.TestCase):
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

        #%% Create phantom
        kernel_size = voxel_num_xy
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

        self.golden_data = self.ig3D.allocate(0)
        for i in range(4):
            self.golden_data.fill(array=phantom, vertical=7+i)

        self.golden_data_cs = self.golden_data.get_slice(vertical=cs_ind, force=True)

        self.Op = ProjectionOperator(self.ig3D, self.ag3D)
        self.fp = self.Op.direct(self.golden_data)

    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_FBP(self):
        reco_out = FBP(self.ig, self.ag)(self.fp.get_slice(vertical='centre'))
        mean_diff = (self.golden_data_cs-reco_out).abs().mean()
        self.assertLess(mean_diff, 0.01)
        np.testing.assert_allclose(self.golden_data_cs.as_array(),reco_out.as_array(),atol=1)

        reco_out3D = FBP(self.ig3D, self.ag3D)(self.fp)
        diff = (self.golden_data-reco_out3D).abs()
        self.assertLess(diff.mean(), 0.01)
        np.testing.assert_allclose(self.golden_data.as_array(),reco_out3D.as_array(),atol=1)


    @unittest.skipUnless(has_tigre and has_astra, "TIGRE or ASTRA not installed")
    def test_fp_with_Astra(self):
        AOp = AstraProjectionOperator(self.ig, self.ag)
        fp_ASTRA = AOp.direct(self.golden_data_cs)

        TOp = ProjectionOperator(self.ig, self.ag)
        fp_TIGRE = TOp.direct(self.golden_data_cs)

        mean_diff = (fp_ASTRA-fp_TIGRE).abs().mean()
        self.assertLess(mean_diff, 1e-2)
        np.testing.assert_allclose(fp_TIGRE.as_array(),fp_ASTRA.as_array(),atol=1)

        astra_ag3D = self.ag3D.copy()
        astra_ag3D.set_labels(['vertical','angle','horizontal'])

        AOp = AstraProjectionOperator(self.ig3D, astra_ag3D)
        fp_ASTRA = AOp.direct(self.golden_data)

        fp_ASTRA.reorder(['angle','vertical','horizontal'])
        mean_diff = (fp_ASTRA-self.fp).abs().mean()
        self.assertLess(mean_diff, 1)
        np.testing.assert_allclose(self.fp.as_array(),fp_ASTRA.as_array(),atol=5)

    @unittest.skipUnless(has_tigre and has_astra, "TIGRE or ASTRA not installed")
    def test_bp_with_Astra(self):
        AOp = AstraProjectionOperator(self.ig, self.ag)
        bp_ASTRA = AOp.adjoint(self.fp.get_slice(vertical='centre'))
        TOp = ProjectionOperator(self.ig, self.ag)
        bp_TIGRE = TOp.adjoint(self.fp.get_slice(vertical='centre'))
        mean_diff = (bp_ASTRA-bp_TIGRE).abs().mean()
        self.assertLess(mean_diff, 1)
        np.testing.assert_allclose(bp_ASTRA.as_array(),bp_TIGRE.as_array(),atol=10)


        astra_fp = self.fp.copy()
        astra_fp.reorder(['vertical','angle','horizontal'])

        AOp = AstraProjectionOperator(self.ig3D, astra_fp.geometry)
        bp_ASTRA = AOp.adjoint(astra_fp)
        
        bp_TIGRE = self.Op.adjoint(self.fp)
        mean_diff = (bp_ASTRA-bp_TIGRE).abs().mean()
        self.assertLess(mean_diff, 1)
        np.testing.assert_allclose(bp_ASTRA.as_array(),bp_TIGRE.as_array(),atol=5)

    @unittest.skipUnless(has_tigre and has_astra, "TIGRE or ASTRA not installed")
    def test_FBP_with_Astra(self):
        reco_ASTRA = AstraFBP(self.ig, self.ag)(self.fp.get_slice(vertical='centre'))
        reco_TIGRE = FBP(self.ig, self.ag)(self.fp.get_slice(vertical='centre'))
        mean_diff = (reco_ASTRA-reco_TIGRE).abs().mean()
        self.assertLess(mean_diff, 1e-4)
        np.testing.assert_allclose(reco_ASTRA.as_array(),reco_TIGRE.as_array(),atol=1e-2)

        astra_fp = self.fp.copy()
        astra_fp.reorder(['vertical','angle','horizontal'])
        reco_ASTRA3D = AstraFBP(self.ig3D, astra_fp.geometry)(astra_fp)
        reco_TIGRE3D = FBP(self.ig3D, self.ag3D)(self.fp)
        diff = (reco_ASTRA3D-reco_TIGRE3D).abs()
        self.assertLess(diff.mean(), 1e-4)
        np.testing.assert_allclose(reco_ASTRA3D.as_array(),reco_TIGRE3D.as_array(),atol=1e-2)
