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
from cil.utilities import dataexample
import unittest
import numpy as np
from utils import has_gpu_tigre
from cil.utilities.display import show2D

try:
    import tigre
    from cil.plugins.tigre import CIL2TIGREGeometry
    from cil.plugins.tigre import ProjectionOperator
    from cil.plugins.tigre import FBP
    has_tigre = True
except ModuleNotFoundError:
    print(  "This plugin requires the additional package TIGRE\n" +
            "Please install it via conda as tigre from the ccpi channel\n"+
            "Minimal version is 21.01")
    has_tigre = False


has_tigre = has_tigre and has_gpu_tigre()

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
        self.assertTrue(tg_geometry.mode=='cone')
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

        self.assertTrue(ag.system_description=='simple')

        ig = ag.get_ImageGeometry()
        ig.voxel_num_y = 50
        ig.voxel_size_y /= 2

        angles_rad = np.array([-np.pi/2, -np.pi, -3 *np.pi/2])

        tg_geometry, tg_angles = CIL2TIGREGeometry.getTIGREGeometry(ig, ag)

        self.assertTrue(tg_geometry.mode=='cone')
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
    def test_cone3D_offset(self):

        #3, 4, 5 triangle for source + object
        ag = AcquisitionGeometry.create_Cone3D(source_position=[0,-4,0], detector_position=[0,4,0], rotation_axis_position=[3,0, 0])\
                                      .set_angles(self.angles_deg, angle_unit='degree')\
                                      .set_labels(['vertical', 'angle','horizontal'])\
                                      .set_panel((self.num_pixels_x,self.num_pixels_y), (self.pixel_size_x,self.pixel_size_y))
                                      
        self.assertTrue(ag.system_description=='offset')

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
        self.assertTrue(tg_geometry.mode=='cone')
        np.testing.assert_allclose(tg_geometry.dDetector, ag.config.panel.pixel_size[::-1])
        np.testing.assert_allclose(tg_geometry.nDetector, ag.config.panel.num_pixels[::-1])
        np.testing.assert_allclose(tg_geometry.sDetector, tg_geometry.dDetector * tg_geometry.nDetector)
        np.testing.assert_allclose(tg_geometry.offOrigin,0)

        mag = ag.magnification
        np.testing.assert_allclose(tg_geometry.nVoxel, [3,50,128])
        np.testing.assert_allclose(tg_geometry.dVoxel, [0.2/mag,0.05/mag,0.1/mag])

    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_cone3D_advanced(self):

        ag = AcquisitionGeometry.create_Cone3D(source_position=[0,-10,0], detector_position=[0,10,0], rotation_axis_position=[0,0, 0],rotation_axis_direction=[0,-1,1])\
                                      .set_angles(self.angles_deg, angle_unit='degree')\
                                      .set_labels(['vertical', 'angle','horizontal'])\
                                      .set_panel((self.num_pixels_x,self.num_pixels_y), (self.pixel_size_x,self.pixel_size_y))
                                      
        self.assertTrue(ag.system_description=='advanced')
        ig = ag.get_ImageGeometry()
        ig.voxel_num_y = 50
        ig.voxel_size_y /= 2

        tg_geometry, tg_angles= CIL2TIGREGeometry.getTIGREGeometry(ig, ag)

        self.assertAlmostEqual(tg_geometry.DSO, ag.dist_source_center*np.sin(np.pi/4),5)

        s2o = ag.dist_source_center * np.cos(np.pi/4)
        np.testing.assert_allclose(tg_geometry.DSO, s2o)

        s2d = (ag.dist_center_detector + ag.dist_source_center) * np.cos(np.pi/4)
        np.testing.assert_allclose(tg_geometry.DSD, s2d)

        det_rot = np.array([0,-np.pi/4,0])
        np.testing.assert_allclose(tg_geometry.rotDetector,det_rot)

        det_offset = np.array([-s2d,0,0])
        np.testing.assert_allclose(tg_geometry.offDetector,det_offset)
        
        angles_rad = np.array([-np.pi/2, -np.pi, -3 *np.pi/2])

        np.testing.assert_allclose(tg_angles, angles_rad)
        self.assertTrue(tg_geometry.mode=='cone')
        np.testing.assert_allclose(tg_geometry.dDetector, ag.config.panel.pixel_size[::-1])
        np.testing.assert_allclose(tg_geometry.nDetector, ag.config.panel.num_pixels[::-1])
        np.testing.assert_allclose(tg_geometry.sDetector, tg_geometry.dDetector * tg_geometry.nDetector)
        np.testing.assert_allclose(tg_geometry.offOrigin,0)

        mag = ag.magnification
        np.testing.assert_allclose(tg_geometry.nVoxel, [3,50,128])
        np.testing.assert_allclose(tg_geometry.dVoxel, [0.2/mag,0.05/mag,0.1/mag])

    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_parallel2D(self):

        ag = AcquisitionGeometry.create_Parallel2D()\
                                     .set_angles(self.angles_rad, angle_unit='radian')\
                                     .set_labels(['angle','horizontal'])\
                                     .set_panel(self.num_pixels_x, self.pixel_size_x)

        ig = ag.get_ImageGeometry()
        ig.voxel_num_y = 50
        ig.voxel_size_y /= 2

        angles_rad = np.array([-np.pi/2, -np.pi, -3 *np.pi/2])
        
        tg_geometry, tg_angles = CIL2TIGREGeometry.getTIGREGeometry(ig, ag)

        np.testing.assert_allclose(tg_angles, angles_rad)
        self.assertTrue(tg_geometry.mode=='parallel')
        np.testing.assert_allclose(tg_geometry.dDetector, ag.config.panel.pixel_size[::-1])
        np.testing.assert_allclose(tg_geometry.nDetector, ag.config.panel.num_pixels[::-1])
        np.testing.assert_allclose(tg_geometry.sDetector, tg_geometry.dDetector * tg_geometry.nDetector)
        np.testing.assert_allclose(tg_geometry.rotDetector,0)
        np.testing.assert_allclose(tg_geometry.offDetector,0)
        np.testing.assert_allclose(tg_geometry.offOrigin,0)

        np.testing.assert_allclose(tg_geometry.nVoxel, [1,50,128])
        np.testing.assert_allclose(tg_geometry.dVoxel, [0.1,0.05,0.1])

    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_parallel3D_simple(self):
        ag = AcquisitionGeometry.create_Parallel3D()\
                                      .set_angles(self.angles_deg, angle_unit='degree')\
                                      .set_labels(['vertical', 'angle','horizontal'])\
                                      .set_panel((self.num_pixels_x,self.num_pixels_y), (self.pixel_size_x,self.pixel_size_y))

        ig = ag.get_ImageGeometry()
        ig.voxel_num_y = 50
        ig.voxel_size_y /= 2

        angles_rad = np.array([-np.pi/2, -np.pi, -3 *np.pi/2])

        tg_geometry, tg_angles = CIL2TIGREGeometry.getTIGREGeometry(ig, ag)

        self.assertTrue(tg_geometry.mode=='parallel')
        np.testing.assert_allclose(tg_angles, angles_rad)
        np.testing.assert_allclose(tg_geometry.dDetector, ag.config.panel.pixel_size[::-1])
        np.testing.assert_allclose(tg_geometry.nDetector, ag.config.panel.num_pixels[::-1])
        np.testing.assert_allclose(tg_geometry.sDetector, tg_geometry.dDetector * tg_geometry.nDetector)
        np.testing.assert_allclose(tg_geometry.rotDetector,0)
        np.testing.assert_allclose(tg_geometry.offDetector,0)
        np.testing.assert_allclose(tg_geometry.offOrigin,0)

        mag = ag.magnification
        np.testing.assert_allclose(tg_geometry.nVoxel, [3,50,128])
        np.testing.assert_allclose(tg_geometry.dVoxel, [0.2,0.05,0.1])

    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_parallel3D_offset(self):

        ag = AcquisitionGeometry.create_Parallel3D(detector_position=[2,0,0], rotation_axis_position=[3,0, 0])\
                                      .set_angles(self.angles_deg, angle_unit='degree')\
                                      .set_labels(['vertical', 'angle','horizontal'])\
                                      .set_panel((self.num_pixels_x,self.num_pixels_y), (self.pixel_size_x,self.pixel_size_y))
                                      
        self.assertTrue(ag.system_description=='offset')

        ig = ag.get_ImageGeometry()
        ig.voxel_num_y = 50
        ig.voxel_size_y /= 2

        tg_geometry, tg_angles= CIL2TIGREGeometry.getTIGREGeometry(ig, ag)


        det_offset = np.array([0,-1,0])
        np.testing.assert_allclose(tg_geometry.offDetector,det_offset)

        angles_rad = np.array([-np.pi/2, -np.pi, -3 *np.pi/2])
        np.testing.assert_allclose(tg_angles, angles_rad)
        self.assertTrue(tg_geometry.mode=='parallel')
        np.testing.assert_allclose(tg_geometry.dDetector, ag.config.panel.pixel_size[::-1])
        np.testing.assert_allclose(tg_geometry.nDetector, ag.config.panel.num_pixels[::-1])
        np.testing.assert_allclose(tg_geometry.sDetector, tg_geometry.dDetector * tg_geometry.nDetector)
        np.testing.assert_allclose(tg_geometry.offOrigin,0)

        np.testing.assert_allclose(tg_geometry.nVoxel, [3,50,128])
        np.testing.assert_allclose(tg_geometry.dVoxel, [0.2,0.05,0.1])


class Test_results_parallel(unittest.TestCase):

    def setUp(self):
       
        self.acq_data = dataexample.SIMULATED_PARALLEL_BEAM_DATA.get()
        self.img_data = dataexample.SIMULATED_SPHERE_VOLUME.get()

        self.acq_data=np.log(self.acq_data)
        self.acq_data*=-1.0

        self.ig = self.img_data.geometry
        self.ag = self.acq_data.geometry


    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def skip_test_foward_siddon(self):

        #CURRENTLY BROKEN AT TIGRE IMPLEMENTATION FOR ALLIGNED VOXEL GRIDS
        Op = ProjectionOperator(self.ig, self.ag, direct_method='Siddon')
        fp = Op.direct(self.img_data)

        diff = fp - self.acq_data
        mean_diff = abs(diff.mean())
        max_diff = diff.abs().max()

        self.assertLess(mean_diff, 1e-4)
        self.assertLess(max_diff, 0.35)

        np.testing.assert_allclose(fp.as_array(), self.acq_data.as_array(),atol=0.35)        


    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_foward_interpolated(self):

        Op = ProjectionOperator(self.ig, self.ag, direct_method='interpolated')
        fp = Op.direct(self.img_data)

        diff = fp - self.acq_data
        mean_diff = abs(diff.mean())
        max_diff = diff.abs().max()

        self.assertLess(mean_diff, 1e-4)
        self.assertLess(max_diff, 0.12)

        np.testing.assert_allclose(fp.as_array(), self.acq_data.as_array(),atol=0.12)       


    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_backward_matched(self):

        Op = ProjectionOperator(self.ig, self.ag, direct_method='interpolated', adjoint_method='matched')

        bp = Op.adjoint(self.acq_data)
        #check not zeros
        self.assertAlmostEqual(bp.mean(), 250.2045, places=2)


    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_FBP(self):

        fbp = FBP(self.ig, self.ag)
        reco = fbp(self.acq_data)

        diff = reco - self.img_data
        mean_diff = abs(diff.mean())
        max_diff = diff.abs().max()

        self.assertLess(mean_diff, 1e-4)
        self.assertLess(max_diff, 1e-3)

        np.testing.assert_allclose(reco.as_array(), self.img_data.as_array(),atol=1e-3)    


    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_norm(self):

        Op = ProjectionOperator(self.ig, self.ag, direct_method='Siddon')
        n = Op.norm()
        self.assertAlmostEqual(n, 764.6795, places=2)

        Op = ProjectionOperator(self.ig, self.ag, direct_method='interpolated')
        n = Op.norm()
        self.assertAlmostEqual(n, 766.6383, places=2)


class Test_results_cone(unittest.TestCase):
    
    def setUp(self):
       
        self.acq_data = dataexample.SIMULATED_CONE_BEAM_DATA.get()
        self.img_data = dataexample.SIMULATED_SPHERE_VOLUME.get()

        self.acq_data=np.log(self.acq_data)
        self.acq_data*=-1.0

        self.ig = self.img_data.geometry
        self.ag = self.acq_data.geometry


    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_foward_siddon(self):

        Op = ProjectionOperator(self.ig, self.ag, direct_method='Siddon')
        fp = Op.direct(self.img_data)

        diff = fp - self.acq_data
        mean_diff = abs(diff.mean())
        max_diff = diff.abs().max()

        self.assertLess(mean_diff, 1e-4)
        self.assertLess(max_diff, 0.35)

        np.testing.assert_allclose(fp.as_array(), self.acq_data.as_array(),atol=0.35)        


    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_foward_interpolated(self):

        Op = ProjectionOperator(self.ig, self.ag, direct_method='interpolated')
        fp = Op.direct(self.img_data)

        diff = fp - self.acq_data
        mean_diff = abs(diff.mean())
        max_diff = diff.abs().max()

        self.assertLess(mean_diff, 1e-6)
        self.assertLess(max_diff, 0.16)

        np.testing.assert_allclose(fp.as_array(), self.acq_data.as_array(),atol=0.16)       


    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_backward_matched(self):

        Op = ProjectionOperator(self.ig, self.ag, adjoint_method='matched')
        bp = Op.adjoint(self.acq_data)
        #check not zeros
        self.assertAlmostEqual(bp.mean(), 250.2045, places=2)


    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_FBP(self):

        fbp = FBP(self.ig, self.ag)
        reco = fbp(self.acq_data)

        diff = reco - self.img_data
        mean_diff = abs(diff.mean())
        max_diff = diff.abs().max()

        self.assertLess(mean_diff, 1e-4)
        self.assertLess(max_diff, 1e-3)

        np.testing.assert_allclose(reco.as_array(), self.img_data.as_array(),atol=1e-3)    


    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_norm(self):

        Op = ProjectionOperator(self.ig, self.ag, direct_method='Siddon')
        n = Op.norm()
        self.assertAlmostEqual(n, 3066.5686, places=2)

        Op = ProjectionOperator(self.ig, self.ag, direct_method='interpolated')
        n = Op.norm()
        self.assertAlmostEqual(n, 3066.4844, places=2)

