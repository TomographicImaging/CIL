from cil.framework import AcquisitionGeometry, ImageGeometry
import unittest
import numpy as np

try:
    from cil.plugins.tigre import CIL2TIGREGeometry
    from cil.plugins.tigre import ProjectionOperator
    from cil.plugins.tigre import FBP
    has_tigre = True
except ModuleNotFoundError:
    print(  "This plugin requires the additional package TIGRE\n" +
            "Please install it via conda as tigre from the ccpi channel\n"+
            "Minimal version is 21.01")
    has_tigre = False

try:
    from cil.plugins.astra.operators import ProjectionOperator as AstraProjectionOperator
    from cil.plugins.astra.processors import FBP as AstraFBP
    has_astra = True
except ModuleNotFoundError:
    has_astra = False

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

        angles_rad = np.zeros([3,3])
        angles_rad[0,0] = -np.pi/2
        angles_rad[1,0] = -np.pi
        angles_rad[2,0] = -3 *np.pi/2

        #2D cone
        geometry = CIL2TIGREGeometry.getTIGREGeometry(ig, ag)
        np.testing.assert_allclose(geometry.DSD, ag.dist_center_detector + ag.dist_source_center)
        np.testing.assert_allclose(geometry.DSO, ag.dist_source_center)
        np.testing.assert_allclose(geometry.angles, angles_rad)
        np.testing.assert_allclose(geometry.dDetector, ag.config.panel.pixel_size[::-1])
        np.testing.assert_allclose(geometry.nDetector, ag.config.panel.num_pixels[::-1])
        np.testing.assert_allclose(geometry.sDetector, geometry.dDetector * geometry.nDetector)
        np.testing.assert_allclose(geometry.COR,0)
        np.testing.assert_allclose(geometry.rotDetector,0)
        np.testing.assert_allclose(geometry.offDetector,0)
        np.testing.assert_allclose(geometry.offOrigin,0)

        mag = ag.magnification
        np.testing.assert_allclose(geometry.nVoxel, [1,128,50])
        np.testing.assert_allclose(geometry.dVoxel, [0.1/mag,0.1/mag,0.05/mag])

    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_cone3D_simple(self):
        ag = AcquisitionGeometry.create_Cone3D(source_position=[0,-2,0], detector_position=[0,1,0])\
                                      .set_angles(self.angles_deg, angle_unit='degree')\
                                      .set_labels(['vertical', 'angle','horizontal'])\
                                      .set_panel((self.num_pixels_x,self.num_pixels_y), (self.pixel_size_x,self.pixel_size_y))

        ig = ag.get_ImageGeometry()
        ig.voxel_num_y = 50
        ig.voxel_size_y /= 2

        angles_rad = np.zeros([3,3])
        angles_rad[0,0] = -np.pi/2
        angles_rad[1,0] = -np.pi
        angles_rad[2,0] = -3 *np.pi/2

        geometry = CIL2TIGREGeometry.getTIGREGeometry(ig, ag)

        np.testing.assert_allclose(geometry.DSD, ag.dist_center_detector + ag.dist_source_center)
        np.testing.assert_allclose(geometry.DSO, ag.dist_source_center)
        np.testing.assert_allclose(geometry.angles, angles_rad)
        np.testing.assert_allclose(geometry.dDetector, ag.config.panel.pixel_size[::-1])
        np.testing.assert_allclose(geometry.nDetector, ag.config.panel.num_pixels[::-1])
        np.testing.assert_allclose(geometry.sDetector, geometry.dDetector * geometry.nDetector)
        np.testing.assert_allclose(geometry.COR,0)
        np.testing.assert_allclose(geometry.rotDetector,0)
        np.testing.assert_allclose(geometry.offDetector,0)
        np.testing.assert_allclose(geometry.offOrigin,0)

        mag = ag.magnification
        np.testing.assert_allclose(geometry.nVoxel, [3,128,50])
        np.testing.assert_allclose(geometry.dVoxel, [0.2/mag,0.1/mag,0.05/mag])

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

        geometry = CIL2TIGREGeometry.getTIGREGeometry(ig, ag)

        np.testing.assert_allclose(geometry.DSO, ag.dist_source_center)

        yaw = np.arcsin(3./5.)
        det_rot = np.zeros((3,3))
        det_rot[:,2] = yaw
        np.testing.assert_allclose(geometry.rotDetector,det_rot)

        offset = 4 * 6 /5
        det_offset = np.zeros((3,3))
        det_offset[:,1] = -offset
        np.testing.assert_allclose(geometry.offDetector,det_offset)
    
        s2d = ag.dist_center_detector + ag.dist_source_center - 6 * 3 /5   
        np.testing.assert_allclose(geometry.DSD, s2d)

        angles_rad = np.zeros([3,3])
        angles_rad[0,0] = -np.pi/2- yaw
        angles_rad[1,0] = -np.pi- yaw
        angles_rad[2,0] = -3 *np.pi/2- yaw

        np.testing.assert_allclose(geometry.angles, angles_rad)
        np.testing.assert_allclose(geometry.dDetector, ag.config.panel.pixel_size[::-1])
        np.testing.assert_allclose(geometry.nDetector, ag.config.panel.num_pixels[::-1])
        np.testing.assert_allclose(geometry.sDetector, geometry.dDetector * geometry.nDetector)
        np.testing.assert_allclose(geometry.COR,0)
        np.testing.assert_allclose(geometry.offOrigin,0)

        mag = ag.magnification
        np.testing.assert_allclose(geometry.nVoxel, [3,128,50])
        np.testing.assert_allclose(geometry.dVoxel, [0.2/mag,0.1/mag,0.05/mag])

    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_cone3D_cofr2(self):
        pass

    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_cone3D_lamino(self):
        pass

    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_IG(self):
        pass

class Test_ProjectionOperator(unittest.TestCase):
    def setUp(self): 
        
        N = 128
        angles = np.linspace(0, np.pi, 4, dtype='float32')

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
        self.assertAlmostEqual(n, 0.741749, places=3)

        n3D = self.Op3D.norm()
        self.assertAlmostEqual(n3D, 0.74150, places=3)

    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_bp(self):
        ad = self.ag.allocate(0)
        res = self.Op.adjoint(ad)   
        self.assertEqual(res.shape, self.ig.shape)

        res = self.ig.allocate(None)
        self.Op.adjoint(ad, out=res)   
        self.assertEqual(res.shape, self.ig.shape)

    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_fp(self):
        data = self.ig.allocate(0)
        res = self.Op.direct(data)
        self.assertEqual(res.shape, self.ag.shape)

        res = self.ag.allocate(None)
        self.Op.direct(data, out=res)
        self.assertEqual(res.shape, self.ag.shape)

    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_consistency(self):
        pass   

    @unittest.skipUnless(has_tigre and has_astra, "TIGRE or ASTRA not installed")
    def test_with_astra(self):
        pass   

class Test_FBP(unittest.TestCase):
    def setUp(self):
        #%% Setup Geometry
        voxel_num_xy = 255
        voxel_num_z = 15
        self.cs_ind = (voxel_num_z-1)//2

        mag = 2
        src_to_obj = 50
        src_to_det = src_to_obj * mag

        pix_size = 0.2
        det_pix_x = voxel_num_xy
        det_pix_y = voxel_num_z

        num_projections = 360
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

        self.golden_data_cs = self.golden_data.subset(vertical=self.cs_ind, force=True)

    @unittest.skipUnless(has_tigre, "TIGRE not installed")
    def test_result_3D(self):
        Op = ProjectionOperator(self.ig3D, self.ag3D)
        fp = Op.direct(self.golden_data)

        reco_out = FBP(self.ig, self.ag)(fp.subset(vertical='centre'))
        mean_diff = (self.golden_data_cs-reco_out).abs().mean()
        self.assertLess(mean_diff, 0.01)
        np.testing.assert_allclose(self.golden_data_cs.as_array(),reco_out.as_array(),atol=1)

        reco_out3D = FBP(self.ig3D, self.ag3D)(fp)
        diff = (self.golden_data-reco_out3D).abs()
        self.assertLess(diff.mean(), 0.01)
        np.testing.assert_allclose(self.golden_data.as_array(),reco_out3D.as_array(),atol=1)

