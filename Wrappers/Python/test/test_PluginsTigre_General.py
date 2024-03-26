#  Copyright 2021 United Kingdom Research and Innovation
#  Copyright 2021 The University of Manchester
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Authors:
# CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt

import unittest
from cil.framework import AcquisitionGeometry
from cil.framework.framework import ImageGeometry
import numpy as np
from cil.utilities.display import show2D
from cil.utilities import dataexample
from utils_projectors import TestCommon_ProjectionOperatorBlockOperator

from utils import has_tigre, has_nvidia, initialise_tests

initialise_tests()

if has_tigre:
    from cil.plugins.tigre import ProjectionOperator
    from cil.plugins.tigre import CIL2TIGREGeometry

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


@unittest.skipUnless(has_tigre and has_nvidia, "Requires TIGRE GPU")
class TestMechanics_tigre(unittest.TestCase):
    def setUp(self):
        self.ag = AcquisitionGeometry.create_Cone2D([0,-500],[0,500]).set_angles([0]).set_panel(5,1)

        arr = np.arange(5*5).reshape(5,5)
        self.ig = ImageGeometry(5,5)
        self.data = self.ig.allocate()
        self.data.fill(arr)

        self.acq_data = self.ag.allocate()
        self.acq_data.fill(arr[0])


    def test_adjoint_weights(self):
        #checks adjoint_weights parameter calls different backend
        Op = ProjectionOperator(self.ig, self.ag, adjoint_weights='matched')
        bp1 = Op.adjoint(self.acq_data)

        Op = ProjectionOperator(self.ig, self.ag, adjoint_weights='FDK')
        bp2 = Op.adjoint(self.acq_data)


        diff = (bp1 - bp2).abs().sum()
        self.assertGreater(diff,25)


    def test_direct_method(self):

        #checks direct_method parameter calls different backend

        Op = ProjectionOperator(self.ig, self.ag, direct_method='Siddon')
        fp1 = Op.direct(self.data)

        Op = ProjectionOperator(self.ig, self.ag, direct_method='interpolated')
        fp2 = Op.direct(self.data)

        diff = (fp1 - fp2).abs().sum()
        self.assertGreater(diff,0.1)

class TestTIGREBlockOperator(unittest.TestCase, TestCommon_ProjectionOperatorBlockOperator):
    def setUp(self):
        data = dataexample.SIMULATED_PARALLEL_BEAM_DATA.get()
        self.data = data.get_slice(vertical='centre')
        ig = self.data.geometry.get_ImageGeometry()
        self.datasplit = self.data.partition(10, 'sequential')

        K = ProjectionOperator(image_geometry=ig, acquisition_geometry=self.datasplit.geometry)
        A = ProjectionOperator(image_geometry=ig, acquisition_geometry=self.data.geometry)
        self.projectionOperator = (A, K)

    @unittest.skipUnless(has_tigre and has_nvidia, "Requires TIGRE and a GPU")
    def test_partition(self):
        self.partition_test()
