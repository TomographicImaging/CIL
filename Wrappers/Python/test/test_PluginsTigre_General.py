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
from cil.framework import AcquisitionGeometry, ImageGeometry
import numpy as np
from scipy.spatial.transform import Rotation
from cil.utilities import dataexample
from utils_projectors import TestCommon_ProjectionOperatorBlockOperator

from utils import has_tigre, has_nvidia, initialise_tests

initialise_tests()

if has_tigre:
    from cil.plugins.tigre import ProjectionOperator
    from cil.plugins.tigre import CIL2TIGREGeometry


@unittest.skipUnless(has_tigre, "TIGRE not installed")
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

    def test_cone2D(self):

        ag = AcquisitionGeometry.create_Cone2D(source_position=[0,-6], detector_position=[0,16])\
                                     .set_angles(self.angles_rad, angle_unit='radian')\
                                     .set_labels(['angle','horizontal'])\
                                     .set_panel(self.num_pixels_x, self.pixel_size_x)

        #2D cone
        tg_geometry, tg_angles = CIL2TIGREGeometry.getTIGREGeometry(self.ig, ag)

        #all cone geometries use the unified per-projection ZYZ Euler route
        self.compare_cone_source(ag, tg_geometry, tg_angles)

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

    def test_cone3D_simple(self):
        ag = AcquisitionGeometry.create_Cone3D(source_position=[0,-6,0], detector_position=[0,16,0])\
                                      .set_angles(self.angles_deg, angle_unit='degree')\
                                      .set_labels(['vertical', 'angle','horizontal'])\
                                      .set_panel((self.num_pixels_x,self.num_pixels_y), (self.pixel_size_x,self.pixel_size_y))

        self.assertTrue(ag.system_description=='simple')

        tg_geometry, tg_angles = CIL2TIGREGeometry.getTIGREGeometry(self.ig, ag)

        #all 3D cone geometries use the unified per-projection ZYZ Euler route
        self.compare_cone_source(ag, tg_geometry, tg_angles)

        self.assertTrue(tg_geometry.mode=='cone')
        np.testing.assert_allclose(tg_geometry.DSD, ag.dist_center_detector + ag.dist_source_center)
        np.testing.assert_allclose(tg_geometry.DSO, ag.dist_source_center)
        np.testing.assert_allclose(tg_geometry.dDetector, ag.config.panel.pixel_size[::-1])
        np.testing.assert_allclose(tg_geometry.nDetector, ag.config.panel.num_pixels[::-1])
        np.testing.assert_allclose(tg_geometry.sDetector, tg_geometry.dDetector * tg_geometry.nDetector)
        np.testing.assert_allclose(tg_geometry.rotDetector,0, atol=1e-12)
        np.testing.assert_allclose(tg_geometry.offDetector,0, atol=1e-12)
        np.testing.assert_allclose(tg_geometry.offOrigin,0)

        np.testing.assert_allclose(tg_geometry.nVoxel, [self.ig.voxel_num_z,self.ig.voxel_num_y,self.ig.voxel_num_x])
        np.testing.assert_allclose(tg_geometry.dVoxel, [self.ig.voxel_size_z,self.ig.voxel_size_y,self.ig.voxel_size_x])

    def compare_cone_source(self, ag, tg_geometry, tg_angles):

        tg_angles = np.asarray(tg_angles)
        self.assertEqual(tg_angles.shape, (len(ag.config.angles.angle_data), 3))

        cil = ag.copy()
        cil.config.system.align_reference_frame('cil')
        S0 = cil.config.system.source.position  # reference source in the volume (CIL) frame
        if S0.size == 2:
            S0 = np.append(S0, 0.)  # promote 2D geometry to 3D

        for a, euler in zip(self.angles_rad, tg_angles):
            source = tg_geometry.DSO * Rotation.from_euler('ZYZ', euler).as_matrix()[:, 0]
            expected = Rotation.from_euler('z', -a).as_matrix() @ S0
            np.testing.assert_allclose(source, expected, atol=1e-4)

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

        self.compare_cone_source(ag, tg_geometry, tg_angles)

        self.assertTrue(tg_geometry.mode=='cone')
        np.testing.assert_allclose(tg_geometry.dDetector, ag.config.panel.pixel_size[::-1])
        np.testing.assert_allclose(tg_geometry.nDetector, ag.config.panel.num_pixels[::-1])
        np.testing.assert_allclose(tg_geometry.sDetector, tg_geometry.dDetector * tg_geometry.nDetector)
        np.testing.assert_allclose(tg_geometry.offOrigin,0)

        np.testing.assert_allclose(tg_geometry.nVoxel, [self.ig.voxel_num_z,self.ig.voxel_num_y,self.ig.voxel_num_x])
        np.testing.assert_allclose(tg_geometry.dVoxel, [self.ig.voxel_size_z,self.ig.voxel_size_y,self.ig.voxel_size_x])

    def test_cone3D_advanced(self):

        tilt = np.pi/4
        ag = AcquisitionGeometry.create_Cone3D(source_position=[0,-10,0], detector_position=[0,10,0], rotation_axis_position=[0,0, 0],rotation_axis_direction=[0,-np.sin(tilt),np.cos(tilt)])\
                                      .set_angles(self.angles_deg, angle_unit='degree')\
                                      .set_labels(['vertical', 'angle','horizontal'])\
                                      .set_panel((self.num_pixels_x,self.num_pixels_y), (self.pixel_size_x,self.pixel_size_y))

        self.assertTrue(ag.system_description=='advanced')

        tg_geometry, tg_angles= CIL2TIGREGeometry.getTIGREGeometry(self.ig, ag)

        #source kept at full distance, detector on the far side at the full source-detector distance
        np.testing.assert_allclose(tg_geometry.DSO, ag.dist_source_center)
        np.testing.assert_allclose(tg_geometry.DSD, ag.dist_center_detector + ag.dist_source_center)

        #the tilt is carried entirely by the Euler angles: no detector rotation, no offsets
        np.testing.assert_allclose(tg_geometry.rotDetector, 0)
        np.testing.assert_allclose(tg_geometry.offDetector, 0, atol=1e-12)
        np.testing.assert_allclose(tg_geometry.offOrigin, 0)

        #per-projection (alpha, theta, psi) triples
        tg_angles = np.asarray(tg_angles)
        self.assertEqual(tg_angles.shape, (len(self.angles_deg), 3))

        #reconstruct the source position from (DSO, Euler) as TIGRE's kernel does
        #(base source [DSO,0,0] rotated by R = Rz(alpha) Ry(theta) Rz(psi), i.e. col 0 of R)
        for alpha, theta, psi in tg_angles:
            source = tg_geometry.DSO * Rotation.from_euler('ZYZ', [alpha, theta, psi]).as_matrix()[:, 0]
            #source stays at the full distance from the origin for every projection
            np.testing.assert_allclose(np.linalg.norm(source), ag.dist_source_center, atol=1e-4)
            #and at a constant tilt from the rotation plane: z = DSO * sin(tilt)
            np.testing.assert_allclose(source[2], ag.dist_source_center*np.sin(tilt), atol=1e-4)

        self.assertTrue(tg_geometry.mode=='cone')
        np.testing.assert_allclose(tg_geometry.dDetector, ag.config.panel.pixel_size[::-1])
        np.testing.assert_allclose(tg_geometry.nDetector, ag.config.panel.num_pixels[::-1])
        np.testing.assert_allclose(tg_geometry.sDetector, tg_geometry.dDetector * tg_geometry.nDetector)

        np.testing.assert_allclose(tg_geometry.nVoxel, [self.ig.voxel_num_z,self.ig.voxel_num_y,self.ig.voxel_num_x])
        np.testing.assert_allclose(tg_geometry.dVoxel, [self.ig.voxel_size_z,self.ig.voxel_size_y,self.ig.voxel_size_x])

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

    def test_parallel3D_advanced(self):
        # A tilted rotation axis (laminography) with a parallel beam takes the same per-projection
        # ZYZ Euler route as the cone advanced case
        tilt = np.pi/4
        ag = AcquisitionGeometry.create_Parallel3D(rotation_axis_direction=[0,-np.sin(tilt),np.cos(tilt)])\
                                      .set_angles(self.angles_deg, angle_unit='degree')\
                                      .set_labels(['vertical', 'angle','horizontal'])\
                                      .set_panel((self.num_pixels_x,self.num_pixels_y), (self.pixel_size_x,self.pixel_size_y))

        self.assertTrue(ag.system_description=='advanced')

        tg_geometry, tg_angles = CIL2TIGREGeometry.getTIGREGeometry(self.ig, ag)

        #no detector rotation, no offsets
        np.testing.assert_allclose(tg_geometry.rotDetector, 0, atol=1e-12)
        np.testing.assert_allclose(tg_geometry.offDetector, 0, atol=1e-12)
        np.testing.assert_allclose(tg_geometry.offOrigin, 0)

        #per-projection (alpha, theta, psi) triples
        tg_angles = np.asarray(tg_angles)
        self.assertEqual(tg_angles.shape, (len(self.angles_deg), 3))

        #TIGRE rotates the volume by R = Rz(alpha) Ry(theta) Rz(psi); the volume z-axis (the
        #rotation axis) holds a constant tilt from the beam-frame z-axis: z-component = cos(tilt)
        for alpha, theta, psi in tg_angles:
            axis = Rotation.from_euler('ZYZ', [alpha, theta, psi]).as_matrix()[:, 2]
            np.testing.assert_allclose(np.linalg.norm(axis), 1.0, atol=1e-4)
            np.testing.assert_allclose(axis[2], np.cos(tilt), atol=1e-4)

        self.assertTrue(tg_geometry.mode=='parallel')
        np.testing.assert_allclose(tg_geometry.dDetector, ag.config.panel.pixel_size[::-1])
        np.testing.assert_allclose(tg_geometry.nDetector, ag.config.panel.num_pixels[::-1])
        np.testing.assert_allclose(tg_geometry.sDetector, tg_geometry.dDetector * tg_geometry.nDetector)
        np.testing.assert_allclose(tg_geometry.nVoxel, [self.ig.voxel_num_z,self.ig.voxel_num_y,self.ig.voxel_num_x])
        np.testing.assert_allclose(tg_geometry.dVoxel, [self.ig.voxel_size_z,self.ig.voxel_size_y,self.ig.voxel_size_x])

    def test_panel_origin_flips(self):
        # The panel storage origin flips the detector by pi about the matching axis (roll/pitch/yaw);

        expected = {
            'bottom-left':  [0, 0, 0],
            'top-left':     [0, np.pi, 0],
            'bottom-right': [0, 0, np.pi],
            'top-right':    [np.pi, 0, 0],
        }
        for origin, rot in expected.items():
            rot_mat = Rotation.from_euler('xyz', rot).as_matrix()
            cone = AcquisitionGeometry.create_Cone3D(source_position=[0,-6,0], detector_position=[0,16,0])\
                                          .set_angles(self.angles_deg, angle_unit='degree')\
                                          .set_labels(['vertical', 'angle','horizontal'])\
                                          .set_panel((self.num_pixels_x,self.num_pixels_y), (self.pixel_size_x,self.pixel_size_y), origin=origin)
            tg_cone, _ = CIL2TIGREGeometry.getTIGREGeometry(self.ig, cone)
            np.testing.assert_allclose(Rotation.from_euler('xyz', tg_cone.rotDetector).as_matrix(), rot_mat, atol=1e-12, err_msg=f"cone origin {origin}")

            par = AcquisitionGeometry.create_Parallel3D()\
                                          .set_angles(self.angles_deg, angle_unit='degree')\
                                          .set_labels(['vertical', 'angle','horizontal'])\
                                          .set_panel((self.num_pixels_x,self.num_pixels_y), (self.pixel_size_x,self.pixel_size_y), origin=origin)
            tg_par, _ = CIL2TIGREGeometry.getTIGREGeometry(self.ig, par)
            np.testing.assert_allclose(Rotation.from_euler('xyz', tg_par.rotDetector).as_matrix(), rot_mat, atol=1e-12, err_msg=f"parallel origin {origin}")


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
