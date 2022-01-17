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
from cil.framework.framework import ImageGeometry
from cil.utilities import dataexample
import unittest
import numpy as np
from utils import has_gpu_tigre, has_tigre
from cil.utilities.display import show2D

if has_tigre:
    from cil.plugins.tigre import CIL2TIGREGeometry
    from cil.plugins.tigre import ProjectionOperator
    from cil.plugins.tigre import FBP

gpu_test = has_gpu_tigre()
if  not gpu_test:
    print("Unable to run TIGRE tests")

has_tigre = has_tigre and gpu_test

class Test_convert_geometry(unittest.TestCase):
    def setUp(self): 
        self.num_pixels_x = 12
        self.num_pixels_y = 6
        self.pixel_size_x = 0.1
        self.pixel_size_y = 0.2

        self.ig = ImageGeometry(3,4,5,0.1,0.2,0.3)
        
        self.angles_deg = np.asarray([0,90.0,180.0], dtype='float32')
        self.angles_rad = self.angles_deg * np.pi /180.0

    def compare_angles(self,ang1,ang2,atol):

        diff = ang1 - ang2

        while diff < -np.pi:
            diff += 2 * np.pi
        while diff >= np.pi:
            diff -= 2 * np.pi

        self.assertLess(abs(diff),atol)

    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_cone2D(self):

        ag = AcquisitionGeometry.create_Cone2D(source_position=[0,-6], detector_position=[0,16])\
                                     .set_angles(self.angles_rad, angle_unit='radian')\
                                     .set_labels(['angle','horizontal'])\
                                     .set_panel(self.num_pixels_x, self.pixel_size_x)

        #2D cone
        tg_geometry, tg_angles = CIL2TIGREGeometry.getTIGREGeometry(self.ig, ag)

        for i, ang in enumerate(tg_angles):
            ang2 = -(self.angles_rad[i] + np.pi/2)
            self.compare_angles(ang,ang2,1e-6)

        self.assertTrue(tg_geometry.mode=='cone')
        np.testing.assert_allclose(tg_geometry.DSD, ag.dist_center_detector + ag.dist_source_center)
        np.testing.assert_allclose(tg_geometry.DSO, ag.dist_source_center)
        np.testing.assert_allclose(tg_geometry.dDetector, ag.config.panel.pixel_size[::-1])
        np.testing.assert_allclose(tg_geometry.nDetector, ag.config.panel.num_pixels[::-1])
        np.testing.assert_allclose(tg_geometry.sDetector, tg_geometry.dDetector * tg_geometry.nDetector)
        np.testing.assert_allclose(tg_geometry.rotDetector,0)
        np.testing.assert_allclose(tg_geometry.offDetector,0)
        np.testing.assert_allclose(tg_geometry.offOrigin,0)

        np.testing.assert_allclose(tg_geometry.nVoxel, [1,self.ig.voxel_num_y,self.ig.voxel_num_x])
        np.testing.assert_allclose(tg_geometry.dVoxel, [ag.config.panel.pixel_size[1]/ag.magnification,self.ig.voxel_size_y,self.ig.voxel_size_x])

    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_cone3D_simple(self):
        ag = AcquisitionGeometry.create_Cone3D(source_position=[0,-6,0], detector_position=[0,16,0])\
                                      .set_angles(self.angles_deg, angle_unit='degree')\
                                      .set_labels(['vertical', 'angle','horizontal'])\
                                      .set_panel((self.num_pixels_x,self.num_pixels_y), (self.pixel_size_x,self.pixel_size_y))

        self.assertTrue(ag.system_description=='simple')

        tg_geometry, tg_angles = CIL2TIGREGeometry.getTIGREGeometry(self.ig, ag)

        for i, ang in enumerate(tg_angles):
            ang2 = -(self.angles_rad[i] + np.pi/2)
            self.compare_angles(ang,ang2,1e-6)

        self.assertTrue(tg_geometry.mode=='cone')
        np.testing.assert_allclose(tg_geometry.DSD, ag.dist_center_detector + ag.dist_source_center)
        np.testing.assert_allclose(tg_geometry.DSO, ag.dist_source_center)
        np.testing.assert_allclose(tg_geometry.dDetector, ag.config.panel.pixel_size[::-1])
        np.testing.assert_allclose(tg_geometry.nDetector, ag.config.panel.num_pixels[::-1])
        np.testing.assert_allclose(tg_geometry.sDetector, tg_geometry.dDetector * tg_geometry.nDetector)
        np.testing.assert_allclose(tg_geometry.rotDetector,0)
        np.testing.assert_allclose(tg_geometry.offDetector,0)
        np.testing.assert_allclose(tg_geometry.offOrigin,0)

        np.testing.assert_allclose(tg_geometry.nVoxel, [self.ig.voxel_num_z,self.ig.voxel_num_y,self.ig.voxel_num_x])
        np.testing.assert_allclose(tg_geometry.dVoxel, [self.ig.voxel_size_z,self.ig.voxel_size_y,self.ig.voxel_size_x])

    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_cone3D_offset(self):

        #3, 4, 5 triangle for source + object
        ag = AcquisitionGeometry.create_Cone3D(source_position=[0,-4,0], detector_position=[0,4,0], rotation_axis_position=[3,0, 0])\
                                      .set_angles(self.angles_deg, angle_unit='degree')\
                                      .set_labels(['vertical', 'angle','horizontal'])\
                                      .set_panel((self.num_pixels_x,self.num_pixels_y), (self.pixel_size_x,self.pixel_size_y))
                                      
        self.assertTrue(ag.system_description=='offset')

        tg_geometry, tg_angles= CIL2TIGREGeometry.getTIGREGeometry(self.ig, ag)

        np.testing.assert_allclose(tg_geometry.DSO, ag.dist_source_center)

        yaw = np.arcsin(3./5.)
        det_rot = np.array([0,0,yaw])
        np.testing.assert_allclose(tg_geometry.rotDetector,det_rot)

        offset = 4 * 6 /5
        det_offset = np.array([0,-offset,0])
        np.testing.assert_allclose(tg_geometry.offDetector,det_offset)
    
        s2d = ag.dist_center_detector + ag.dist_source_center - 6 * 3 /5   
        np.testing.assert_allclose(tg_geometry.DSD, s2d)

        for i, ang in enumerate(tg_angles):
            ang2 = -(self.angles_rad[i] + np.pi/2 + yaw)
            self.compare_angles(ang,ang2,1e-6)

        self.assertTrue(tg_geometry.mode=='cone')
        np.testing.assert_allclose(tg_geometry.dDetector, ag.config.panel.pixel_size[::-1])
        np.testing.assert_allclose(tg_geometry.nDetector, ag.config.panel.num_pixels[::-1])
        np.testing.assert_allclose(tg_geometry.sDetector, tg_geometry.dDetector * tg_geometry.nDetector)
        np.testing.assert_allclose(tg_geometry.offOrigin,0)

        np.testing.assert_allclose(tg_geometry.nVoxel, [self.ig.voxel_num_z,self.ig.voxel_num_y,self.ig.voxel_num_x])
        np.testing.assert_allclose(tg_geometry.dVoxel, [self.ig.voxel_size_z,self.ig.voxel_size_y,self.ig.voxel_size_x])


    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_cone3D_advanced(self):

        ag = AcquisitionGeometry.create_Cone3D(source_position=[0,-10,0], detector_position=[0,10,0], rotation_axis_position=[0,0, 0],rotation_axis_direction=[0,-1,1])\
                                      .set_angles(self.angles_deg, angle_unit='degree')\
                                      .set_labels(['vertical', 'angle','horizontal'])\
                                      .set_panel((self.num_pixels_x,self.num_pixels_y), (self.pixel_size_x,self.pixel_size_y))
                                      
        self.assertTrue(ag.system_description=='advanced')

        tg_geometry, tg_angles= CIL2TIGREGeometry.getTIGREGeometry(self.ig, ag)

        self.assertAlmostEqual(tg_geometry.DSO, ag.dist_source_center*np.sin(np.pi/4),5)

        s2o = ag.dist_source_center * np.cos(np.pi/4)
        np.testing.assert_allclose(tg_geometry.DSO, s2o)

        s2d = (ag.dist_center_detector + ag.dist_source_center) * np.cos(np.pi/4)
        np.testing.assert_allclose(tg_geometry.DSD, s2d)

        det_rot = np.array([0,-np.pi/4,0])
        np.testing.assert_allclose(tg_geometry.rotDetector,det_rot)

        det_offset = np.array([-s2d,0,0])
        np.testing.assert_allclose(tg_geometry.offDetector,det_offset)
        
        for i, ang in enumerate(tg_angles):
            ang2 = -(self.angles_rad[i] + np.pi/2)
            self.compare_angles(ang,ang2,1e-6)

        self.assertTrue(tg_geometry.mode=='cone')
        np.testing.assert_allclose(tg_geometry.dDetector, ag.config.panel.pixel_size[::-1])
        np.testing.assert_allclose(tg_geometry.nDetector, ag.config.panel.num_pixels[::-1])
        np.testing.assert_allclose(tg_geometry.sDetector, tg_geometry.dDetector * tg_geometry.nDetector)


        height = 10 / np.sqrt(2)
        np.testing.assert_allclose(tg_geometry.offOrigin,[-height,0,0])

        np.testing.assert_allclose(tg_geometry.nVoxel, [self.ig.voxel_num_z,self.ig.voxel_num_y,self.ig.voxel_num_x])
        np.testing.assert_allclose(tg_geometry.dVoxel, [self.ig.voxel_size_z,self.ig.voxel_size_y,self.ig.voxel_size_x])


    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_parallel2D(self):

        ag = AcquisitionGeometry.create_Parallel2D()\
                                     .set_angles(self.angles_rad, angle_unit='radian')\
                                     .set_labels(['angle','horizontal'])\
                                     .set_panel(self.num_pixels_x, self.pixel_size_x)

        tg_geometry, tg_angles = CIL2TIGREGeometry.getTIGREGeometry(self.ig, ag)

        for i, ang in enumerate(tg_angles):
            ang2 = -(self.angles_rad[i] + np.pi/2)
            self.compare_angles(ang,ang2,1e-6)

        self.assertTrue(tg_geometry.mode=='parallel')
        np.testing.assert_allclose(tg_geometry.dDetector, ag.config.panel.pixel_size[::-1])
        np.testing.assert_allclose(tg_geometry.nDetector, ag.config.panel.num_pixels[::-1])
        np.testing.assert_allclose(tg_geometry.sDetector, tg_geometry.dDetector * tg_geometry.nDetector)
        np.testing.assert_allclose(tg_geometry.rotDetector,0)
        np.testing.assert_allclose(tg_geometry.offDetector,0)
        np.testing.assert_allclose(tg_geometry.offOrigin,0)

        np.testing.assert_allclose(tg_geometry.nVoxel, [1,self.ig.voxel_num_y,self.ig.voxel_num_x])
        np.testing.assert_allclose(tg_geometry.dVoxel, [ag.config.panel.pixel_size[1],self.ig.voxel_size_y,self.ig.voxel_size_x])


    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_parallel3D_simple(self):
        ag = AcquisitionGeometry.create_Parallel3D()\
                                      .set_angles(self.angles_deg, angle_unit='degree')\
                                      .set_labels(['vertical', 'angle','horizontal'])\
                                      .set_panel((self.num_pixels_x,self.num_pixels_y), (self.pixel_size_x,self.pixel_size_y))

        tg_geometry, tg_angles = CIL2TIGREGeometry.getTIGREGeometry(self.ig, ag)

        for i, ang in enumerate(tg_angles):
            ang2 = -(self.angles_rad[i] + np.pi/2)
            self.compare_angles(ang,ang2,1e-6)

        self.assertTrue(tg_geometry.mode=='parallel')
        np.testing.assert_allclose(tg_geometry.dDetector, ag.config.panel.pixel_size[::-1])
        np.testing.assert_allclose(tg_geometry.nDetector, ag.config.panel.num_pixels[::-1])
        np.testing.assert_allclose(tg_geometry.sDetector, tg_geometry.dDetector * tg_geometry.nDetector)
        np.testing.assert_allclose(tg_geometry.rotDetector,0)
        np.testing.assert_allclose(tg_geometry.offDetector,0)
        np.testing.assert_allclose(tg_geometry.offOrigin,0)

        np.testing.assert_allclose(tg_geometry.nVoxel, [self.ig.voxel_num_z,self.ig.voxel_num_y,self.ig.voxel_num_x])
        np.testing.assert_allclose(tg_geometry.dVoxel, [self.ig.voxel_size_z,self.ig.voxel_size_y,self.ig.voxel_size_x])


    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_parallel3D_offset(self):

        ag = AcquisitionGeometry.create_Parallel3D(detector_position=[2,0,0], rotation_axis_position=[3,0, 0])\
                                      .set_angles(self.angles_deg, angle_unit='degree')\
                                      .set_labels(['vertical', 'angle','horizontal'])\
                                      .set_panel((self.num_pixels_x,self.num_pixels_y), (self.pixel_size_x,self.pixel_size_y))
                                      
        self.assertTrue(ag.system_description=='offset')


        tg_geometry, tg_angles= CIL2TIGREGeometry.getTIGREGeometry(self.ig, ag)

        det_offset = np.array([0,-1,0])
        np.testing.assert_allclose(tg_geometry.offDetector,det_offset)

        for i, ang in enumerate(tg_angles):
            ang2 = -(self.angles_rad[i] + np.pi/2)
            self.compare_angles(ang,ang2,1e-6)

        self.assertTrue(tg_geometry.mode=='parallel')
        np.testing.assert_allclose(tg_geometry.dDetector, ag.config.panel.pixel_size[::-1])
        np.testing.assert_allclose(tg_geometry.nDetector, ag.config.panel.num_pixels[::-1])
        np.testing.assert_allclose(tg_geometry.sDetector, tg_geometry.dDetector * tg_geometry.nDetector)
        np.testing.assert_allclose(tg_geometry.offOrigin,0)

        np.testing.assert_allclose(tg_geometry.nVoxel, [self.ig.voxel_num_z,self.ig.voxel_num_y,self.ig.voxel_num_x])
        np.testing.assert_allclose(tg_geometry.dVoxel, [self.ig.voxel_size_z,self.ig.voxel_size_y,self.ig.voxel_size_x])


class TestCommon(object):

    def compare_forward(self, direct_method, atol):

        Op = ProjectionOperator(self.ig, self.ag, direct_method=direct_method)
        fp = Op.direct(self.img_data)
        np.testing.assert_allclose(fp.as_array(), self.acq_data.as_array(),atol=atol)        

        bp = Op.adjoint(fp)
        fp2 = fp.copy()
        fp2.fill(0)
        Op.direct(self.img_data,out=fp2)
        np.testing.assert_allclose(fp.as_array(), fp2.as_array(),1e-8)    

    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_backward(self):
        
        #this checks mechanics but not value
        Op = ProjectionOperator(self.ig, self.ag, adjoint_weights='matched')
        bp = Op.adjoint(self.acq_data)

        bp2 = bp.copy()
        bp2.fill(0)
        Op.adjoint(self.acq_data,out=bp2)
        np.testing.assert_allclose(bp.as_array(), bp2.as_array(), 1e-8)    


    def compare_backward_FDK_matched(self):
        #this checks mechanics but not value
        Op = ProjectionOperator(self.ig, self.ag, adjoint_weights='matched')
        bp = Op.adjoint(self.acq_data)

        Op = ProjectionOperator(self.ig, self.ag, adjoint_weights='FDK')
        bp3 = Op.adjoint(self.acq_data)

        #checks weights parameter calls different backend
        diff = (bp3 - bp).abs().mean()
        self.assertGreater(diff,1000)


    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_backward_results(self):
        #create checker-board projection
        checker = np.zeros((16,16))
        ones = np.ones((4,4))
        for j in range(4):
            for i in range(4):
                if (i + j)% 2 == 0: 
                    checker[j*4:(j+1)*4,i*4:(i+1)*4] = ones

        #create backprojection of checker-board
        res = np.zeros((16,16,16))
        ones = np.ones((4,16,4))
        for k in range(4):
            for i in range(4):
                if (i + k)% 2 == 0: 
                    res[k*4:(k+1)*4,:,i*4:(i+1)*4] = ones

        if self.ag_small.dimension == '2D':
            checker = checker[0]
            res = res[0]

        ig = self.ag_small.get_ImageGeometry()
        data = self.ag_small.allocate(0)

        data.fill(checker)

        Op = ProjectionOperator(ig,self.ag_small)
        bp = Op.adjoint(data)

        if self.ag_small.geom_type == 'cone':
            #as cone beam res is not perfect grid
            bp.array = np.round(bp.array,0)

        np.testing.assert_equal(bp.array, res)


    def compare_FBP(self,atol):
        fbp = FBP(self.ig, self.ag)
        reco = fbp(self.acq_data)
        np.testing.assert_allclose(reco.as_array(), self.img_data.as_array(),atol=atol)    

        reco2 = reco.copy()
        reco2.fill(0)
        fbp(self.acq_data,out=reco2)
        np.testing.assert_allclose(reco.as_array(), reco2.as_array(),atol=1e-8)   


    def compare_norm(self,direct_method,norm):
        ig = self.ag_small.get_ImageGeometry()
        Op = ProjectionOperator(ig, self.ag_small, direct_method=direct_method)
        n = Op.norm(seed=0.52)
        self.assertAlmostEqual(n, norm, places=1)


    def compare_FBP_roi(self, atol):
        vox_size = self.ag.config.panel.pixel_size / self.ag.magnification
        roi = [20,30,40]

        center_offset = [50,-20,5]
        center_offset = [0,-0,0]

        ig = ImageGeometry(roi[2],roi[1],roi[0],\
            vox_size[1],vox_size[1],vox_size[0],\
            center_offset[2]*vox_size[1],center_offset[1]*vox_size[0],center_offset[0]*vox_size[0])
        
        fbp = FBP(ig, self.ag)
        reco = fbp(self.acq_data)

        x0 = int(ig.get_min_x() / vox_size[1] + self.img_data.shape[2]//2)
        x1 = int(ig.get_max_x() / vox_size[1] + self.img_data.shape[2]//2)
        y0 = int(ig.get_min_y() / vox_size[1] + self.img_data.shape[1]//2)
        y1 = int(ig.get_max_y() / vox_size[1] + self.img_data.shape[1]//2)
        z0 = int(ig.get_min_z() / vox_size[0] + self.img_data.shape[0]//2)
        z1 = int(ig.get_max_z() / vox_size[0] + self.img_data.shape[0]//2)

        gold_roi = self.img_data.as_array()[z0:z1,y0:y1,x0:x1]
        np.testing.assert_allclose(reco.as_array(),gold_roi ,atol=atol)    


class Test_results_cone3D(TestCommon,unittest.TestCase):
    def setUp(self):
        self.acq_data = dataexample.SIMULATED_CONE_BEAM_DATA.get()
        self.img_data = dataexample.SIMULATED_SPHERE_VOLUME.get()

        self.acq_data=np.log(self.acq_data)
        self.acq_data*=-1.0

        self.ig = self.img_data.geometry
        self.ag = self.acq_data.geometry

        self.ag_small = AcquisitionGeometry.create_Cone3D([0,-1000,0],[0,0,0])
        self.ag_small.set_panel((16,16))
        self.ag_small.set_angles([0])


    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_forward_siddon(self):
        self.compare_forward('Siddon',0.35)


    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_forward_interpolated(self):
        self.compare_forward('interpolated',0.16)  


    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_backward_projectors(self):
        self.compare_backward_FDK_matched()


    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_FBP(self):
        self.compare_FBP(1e-3)  


    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_norm_siddon(self):
        self.compare_norm('Siddon',3.9954)


    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_norm_interpolated(self):
        self.compare_norm('interpolated',3.9965)


    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_reconstruct_roi(self):
        self.compare_FBP_roi(1e-3)  


class Test_results_parallel3D(TestCommon,unittest.TestCase):
    
    def setUp(self):
       
        self.acq_data = dataexample.SIMULATED_PARALLEL_BEAM_DATA.get()
        self.img_data = dataexample.SIMULATED_SPHERE_VOLUME.get()

        self.acq_data=np.log(self.acq_data)
        self.acq_data*=-1.0

        self.ig = self.img_data.geometry
        self.ag = self.acq_data.geometry

        self.ag_small = AcquisitionGeometry.create_Parallel3D()
        self.ag_small.set_panel((16,16))
        self.ag_small.set_angles([0])


    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_forward_siddon(self):

        self.compare_forward('Siddon',0.24)


    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_forward_interpolated(self):
    
        self.compare_forward('interpolated',0.12)  


    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_FBP(self):

        self.compare_FBP(1e-3)  


    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_norm_siddon(self):

        self.compare_norm('Siddon',4.000)


    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_norm_interpolated(self):

        self.compare_norm('interpolated',4.000)


    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_reconstruct_roi(self):
        self.compare_FBP_roi(1e-3)  


class Test_results_cone2D(TestCommon,unittest.TestCase):
    
    def setUp(self):
       
        self.acq_data = dataexample.SIMULATED_CONE_BEAM_DATA.get().get_slice(vertical='centre')
        self.img_data = dataexample.SIMULATED_SPHERE_VOLUME.get().get_slice(vertical='centre')

        self.acq_data=np.log(self.acq_data)
        self.acq_data*=-1.0

        self.ig = self.img_data.geometry
        self.ag = self.acq_data.geometry

        self.ag_small = AcquisitionGeometry.create_Cone2D([0,-1000],[0,0])
        self.ag_small.set_panel((16))
        self.ag_small.set_angles([0])


    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_forward_siddon(self):

        self.compare_forward('Siddon',0.19)


    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_forward_interpolated(self):
    
        self.compare_forward('interpolated',0.085)  


    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_backward_projectors(self):
    
        self.compare_backward_FDK_matched()


    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_FBP(self):

        self.compare_FBP(1e-3)  


    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_norm_siddon(self):

        self.compare_norm('Siddon',3.9975)
        

    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_norm_interpolated(self):

        self.compare_norm('interpolated',3.9973)


class Test_results_parallel2D(TestCommon,unittest.TestCase):
        
    def setUp(self):
       
        self.acq_data = dataexample.SIMULATED_PARALLEL_BEAM_DATA.get().get_slice(vertical='centre')
        self.img_data = dataexample.SIMULATED_SPHERE_VOLUME.get().get_slice(vertical='centre')

        self.acq_data=np.log(self.acq_data)
        self.acq_data*=-1.0

        self.ig = self.img_data.geometry
        self.ag = self.acq_data.geometry

        self.ag_small = AcquisitionGeometry.create_Parallel2D()
        self.ag_small.set_panel((16))
        self.ag_small.set_angles([0])

    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_forward_siddon(self):

        self.compare_forward('Siddon',0.15)


    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_forward_interpolated(self):
    
        self.compare_forward('interpolated',0.085)  


    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_FBP(self):

        self.compare_FBP(1e-3)  


    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_norm_siddon(self):

        self.compare_norm('Siddon',4.0000)
        

    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_norm_interpolated(self):

        self.compare_norm('interpolated',4.000)

