
from cil.optimisation.algorithms import SIRT, GD, ISTA, FISTA
from cil.optimisation.functions import LeastSquares, IndicatorBox
from cil.framework import ImageGeometry, VectorGeometry
from cil.optimisation.operators import IdentityOperator, MatrixOperator

from cil.optimisation.utilities import Sensitivity, AdaptiveSensitivity, Preconditioner
import numpy as np

from testclass import CCPiTestClass
from unittest.mock import MagicMock

from cil.framework import AcquisitionGeometry

from cil.plugins.astra.operators import ProjectionOperator
from cil.optimisation.operators import ScaledOperator
import random

# set up TGV
from cil.optimisation.functions import MixedL21Norm, BlockFunction, L2NormSquared, ScaledFunction
from cil.optimisation.operators import BlockOperator, IdentityOperator, GradientOperator, \
    SymmetrisedGradientOperator, ZeroOperator


from cil.optimisation.utilities.PDHG import setup_explicit_TGV, setup_explicit_TV

class TestPDHGUtilities(CCPiTestClass):


    def setUp(self):

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

        ag2D = AcquisitionGeometry.create_Cone2D([0,-src_to_obj],[0,src_to_det-src_to_obj])\
                                            .set_angles(angles)\
                                            .set_panel(det_pix_x, pix_size)\
                                            .set_labels(['angle','horizontal'])

        self.ad2D = ag2D.allocate('random')
        ig2D = ag2D.get_ImageGeometry()

        ag3D = AcquisitionGeometry.create_Cone3D([0,-src_to_obj,0],[0,src_to_det-src_to_obj,0])\
                                        .set_angles(angles)\
                                        .set_panel((det_pix_x,det_pix_y), (pix_size,pix_size))\
                                        .set_labels(['angle','vertical','horizontal'])
        
        ig3D = ag3D.get_ImageGeometry()

        self.ad3D = ag3D.allocate('random')
        self.ad3D.reorder('astra')
        ig3D = ag3D.get_ImageGeometry()


        self.A_2D = ProjectionOperator(ig2D, ag2D, device = "gpu")
        self.A_3D = ProjectionOperator(ig3D, self.ad3D.geometry, device = "gpu")



    def test_setup_explicit_TV(self):

        alphas=[1, random.randint(2,10)]
        alpha = alphas[1]
    
        omegas = [1, random.randint(2,10)]

        for omega in omegas:
            for alpha in alphas:

                K_2D, F_2D = setup_explicit_TV(self.A_2D, self.ad2D, alpha, omega)
                K_3D,  F_3D = setup_explicit_TV(self.A_3D, self.ad3D, alpha, omega)

                case_2D = {'A': self.A_2D, 'K': K_2D, 'F': F_2D, 'ad': self.ad2D}
                case_3D = {'A': self.A_3D, 'K': K_3D, 'F': F_3D, 'ad': self.ad3D}

                cases = [case_2D, case_3D]

                case_names = ['2D', '3D']

                for i, case in enumerate(cases):
                        
                    with self.subTest(case_names[i]):
                        A = case['A']
                        K = case['K']
                        F = case['F']
                        ad = case['ad']

                        # Testing K --------------------------
                        np.testing.assert_equal(type(K), BlockOperator)
                        np.testing.assert_equal(K.shape, (2,1))

                        # K[0]
                        np.testing.assert_equal(K[0], A)

                        # K [1]

                        # We expect the second part of the K operator to be Grad multiplied by alpha
                        np.testing.assert_equal(type(K[1]), ScaledOperator)
                        np.testing.assert_equal(K[1].scalar, alpha)
                        np.testing.assert_equal(type(K[1].operator), GradientOperator)
                        ig = ad.geometry.get_ImageGeometry()
                        expected_grad = alpha*GradientOperator(ig)
                        np.testing.assert_allclose(expected_grad.direct(ad)[1].as_array(), expected_grad.direct(ad)[1].as_array())
                        np.testing.assert_allclose(expected_grad.direct(ad)[1].as_array(), K[1].direct(ad)[1].as_array(), 10**(-4))
                        np.testing.assert_allclose(expected_grad.direct(ad)[0].as_array(), K[1].direct(ad)[0].as_array(), 10**(-4))


                        # Testing F --------------------------------------
                        np.testing.assert_equal(type(F), BlockFunction)
                        np.testing.assert_equal(F.length, 2)

                        # F[0]

                        if omega == 1:
                            np.testing.assert_equal(L2NormSquared, type(F[0]))
                            function = F[0]
                        else:
                            np.testing.assert_equal(ScaledFunction, type(F[0]))
                            np.testing.assert_equal(F[0].scalar, omega)
                            function = F[0].function

                        np.testing.assert_equal(type(function), L2NormSquared)
                        np.testing.assert_array_equal(function.b.as_array(), ad.as_array())


                        # F[1]

                        np.testing.assert_equal(MixedL21Norm, type(F[1]))

    
    

   