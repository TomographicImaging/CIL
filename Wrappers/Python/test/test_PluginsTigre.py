from cil.framework import AcquisitionGeometry, ImageGeometry
from cil.plugins.tigre import CIL2TIGREGeometry
from cil.plugins.tigre import ProjectionOperator
from cil.plugins.tigre import FBP

import unittest
import numpy as np
import sys

if 'tigre' in sys.modules:
    has_tigre = True
else:
    print(  "This plugin requires the additional package TIGRE\n" +
            "Please install it via conda as tigre from the ccpi channel\n"+
            "Minimal version is 21.01")
    has_tigre = False

class BPasicTigreTests(unittest.TestCase):
    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_1(self):
        assert True

    def test_2(self):
        assert True


class TestGeometry(unittest.TestCase):
    def setUp(self): 
        # Define image geometry.
        pixels_x = 128
        pixels_y = 3
        
        angles_deg = np.asarray([0,90.0,180.0], dtype='float32')
        angles_rad = angles_deg * np.pi /180.0

        self.ag_cone = AcquisitionGeometry.create_Cone2D(source_position=[0,-2], detector_position=[0,1])\
                                     .set_angles(angles_rad, angle_unit='radian')\
                                     .set_labels(['angle','horizontal'])\
                                     .set_panel(pixels_x, 0.1)

        self.ag3_cone = AcquisitionGeometry.create_Cone3D(source_position=[0,-2,0], detector_position=[0,1,0])\
                                      .set_angles(angles_deg, angle_unit='degree')\
                                      .set_labels(['vertical', 'angle','horizontal'])\
                                      .set_panel((pixels_x,pixels_y), (0.1,0.2))

        self.ig = self.ag_cone.get_ImageGeometry()
        self.ig.voxel_num_y = 50
        self.ig.voxel_size_y /= 2

        self.ig3 = self.ag3_cone.get_ImageGeometry()
        self.ig3.voxel_num_y = 50
        self.ig3.voxel_size_y /= 2

    def test_convert_geometry_to_tigre(self):

        angles_rad = np.zeros([3,3])
        angles_rad[0,0] = -np.pi/2
        angles_rad[1,0] = -np.pi
        angles_rad[2,0] = -3 *np.pi/2

        #2D cone
        geometry = CIL2TIGREGeometry.getTIGREGeometry(self.ig, self.ag_cone)
        np.testing.assert_allclose(geometry.DSD, self.ag_cone.dist_center_detector + self.ag_cone.dist_source_center)
        np.testing.assert_allclose(geometry.DSO, self.ag_cone.dist_source_center)
        np.testing.assert_allclose(geometry.angles, angles_rad)
        np.testing.assert_allclose(geometry.dDetector, self.ag_cone.config.panel.pixel_size[::-1])
        np.testing.assert_allclose(geometry.nDetector, self.ag_cone.config.panel.num_pixels[::-1])
        np.testing.assert_allclose(geometry.sDetector, geometry.dDetector * geometry.nDetector)
        np.testing.assert_allclose(geometry.COR,0)
        np.testing.assert_allclose(geometry.rotDetector,0)
        np.testing.assert_allclose(geometry.offDetector,0)
        np.testing.assert_allclose(geometry.offOrigin,0)

        mag = self.ag_cone.magnification
        np.testing.assert_allclose(geometry.nVoxel, [1,128,50])
        np.testing.assert_allclose(geometry.dVoxel, [1,0.1/mag,0.05/mag])

        #3D cone
        geometry = CIL2TIGREGeometry.getTIGREGeometry(self.ig3, self.ag3_cone)

        np.testing.assert_allclose(geometry.DSD, self.ag3_cone.dist_center_detector + self.ag3_cone.dist_source_center)
        np.testing.assert_allclose(geometry.DSO, self.ag3_cone.dist_source_center)
        np.testing.assert_allclose(geometry.angles, angles_rad)
        np.testing.assert_allclose(geometry.dDetector, self.ag3_cone.config.panel.pixel_size[::-1])
        np.testing.assert_allclose(geometry.nDetector, self.ag3_cone.config.panel.num_pixels[::-1])
        np.testing.assert_allclose(geometry.sDetector, geometry.dDetector * geometry.nDetector)
        np.testing.assert_allclose(geometry.COR,0)
        np.testing.assert_allclose(geometry.rotDetector,0)
        np.testing.assert_allclose(geometry.offDetector,0)
        np.testing.assert_allclose(geometry.offOrigin,0)

        mag = self.ag_cone.magnification
        np.testing.assert_allclose(geometry.nVoxel, [3,128,50])
        np.testing.assert_allclose(geometry.dVoxel, [0.2/mag,0.1/mag,0.05/mag])

    def test_ProjectionOperator(self):
        Op = ProjectionOperator(self.ig, self.ag_cone)
        n = Op.norm()
        print ("norm A GPU", n)
        self.assertAlmostEqual(n, norm, places=2)

        Op = ProjectionOperator(self.ig3, self.ag3_cone)
        n = Op.norm()
        print ("norm A GPU", n)
        self.assertAlmostEqual(n, norm, places=2)

    def test_FBP(self):
        pass