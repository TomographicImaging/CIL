from cil.utilities import dataexample
from cil.optimisation.functions import TotalGeneralisedVariation, LeastSquares
from cil.optimisation.functions import FISTA
import unittest
import numpy as np

try:
    from ccpi.filters import regularisers
    has_reg_toolkit = True
except ImportError as ie:
    has_reg_toolkit = False
if has_reg_toolkit:
    from cil.plugins.ccpi_regularisation.functions import TGV

from utils import has_astra

if has_astra:
    from cil.plugins.astra.operators import ProjectionOperator
    from cil.framework import AcquisitionGeometry    


class TestTotalGeneralisedVariation(unittest.TestCase):

    def setUp(self):

        # default test data
        self.data = dataexample.CAMERA.get(size=(32, 32))
        
        self.alpha1 = 0.1
        self.alpha0 = 0.5
        self.num_iter = 2000

    @unittest.skipUnless(has_reg_toolkit, "Regularisation Toolkit not present")
    def test_compare_TGV_CIL_vs_CCPiRegTk_denoising(self):

        tgv_cil_fun = TotalGeneralisedVariation(alpha1 = self.alpha1, alpha0 = self.alpha0, max_iteration=self.num_iter)
        sol1 = tgv_cil_fun.proximal(self.data, tau=1.0)

        tgv_ccpi_plugin_fun = TGV(alpha=1.0, alpha1=self.alpha1, alpha0=self.alpha0, max_iteration=self.num_iter,tolerance=0, device='cpu')
        sol2 = tgv_ccpi_plugin_fun.proximal(self.data, tau=1.0)

        np.testing.assert_allclose(sol1.array, sol2.array, atol=1e-1) 

    @unittest.skipUnless(has_astra and has_reg_toolkit, "Astra and CCPi-Regularisation toolkit are required")
    def test_compare_TGV_CIL_vs_CCPiRegTk_tomography(self): 

        # Detectors
        N = 32
        detectors =  N

        # Angles
        angles = np.linspace(0,180,90, dtype='float32')

        # Setup acquisition geometry
        ag = AcquisitionGeometry.create_Parallel2D()\
                                .set_angles(angles)\
                                .set_panel(detectors)
        # Get image geometry
        ig = ag.get_ImageGeometry()

        # Create projection operator using Astra-Toolbox. Available CPU/CPU
        A = ProjectionOperator(ig, ag, device = 'cpu')    

        # Get phantom
        phantom = dataexample.SIMPLE_PHANTOM_2D.get(size=(N, N))

        # Create an acqusition data
        tmp_sino = A.direct(phantom)

        # Simulate Gaussian noise for the sinogram
        gaussian_var = 0.5
        gaussian_mean = 0

        n1 = np.random.normal(gaussian_mean, gaussian_var, size = ag.shape)
                            
        sino = ag.allocate()
        sino.fill(n1 + tmp_sino.array)
        sino.array[sino.array<0]=0            


        f = LeastSquares(A = A, b = sino, c=1.0)

        alpha1 = 10
        alpha0 = 50

        g_cil = TotalGeneralisedVariation(alpha1 = alpha1, alpha0 = alpha0, 
                                    max_iteration=50)
        g_ccpi_regtk = TGV(alpha=1, alpha1=alpha1, alpha0=alpha0, 
                                    max_iteration=50,
                                    tolerance=0,
                                    device='cpu')

        tmp_initial = ig.allocate()

        fista_cil = FISTA(initial=tmp_initial, f=f, g=g_cil, max_iteration=500)
        fista_cil.run(verbose=1)

        fista_ccpi_regtk = FISTA(initial=tmp_initial, f=f, g=g_ccpi_regtk, 
                                update_objective_interval=50, max_iteration=500)
        fista_ccpi_regtk.run(verbose=1) 

        np.testing.assert_allclose(fista_cil.solution.array, fista_ccpi_regtk.solution.array, atol=1e-2) #does not pass
       
   



       







    # def setUp(self) -> None:


    #     self.alpha1 = 0.1
    #     self.alpha0 = 0.5
    #     sel
    #     self.tv = TotalVariation()
    #     self.alpha = 0.15
    #     self.tv_scaled = self.alpha * TotalVariation()
    #     self.tv_iso = TotalVariation()
    #     self.tv_aniso = TotalVariation(isotropic=False)
    #     self.ig_real = ImageGeometry(3,4)   
    #     self.grad = GradientOperator(self.ig_real)  
        
    # def test_regularisation_parameter(self):
    #     np.testing.assert_almost_equal(self.tv.regularisation_parameter, 1.)


    # def test_regularisation_parameter2(self):
    #     np.testing.assert_almost_equal(self.tv_scaled.regularisation_parameter, self.alpha)


    # def test_rmul(self):
    #     assert isinstance(self.tv_scaled, TotalVariation)


    # def test_regularisation_parameter3(self):
    #     with self.assertRaises(TypeError):
    #         self.tv.regularisation_parameter = 'string'
            

    # def test_rmul2(self):
    #     alpha = 'string'
    #     with self.assertRaises(TypeError):
    #         tv = alpha * TotalVariation()
            

    # def test_call_real_isotropic(self):
    #     x_real = self.ig_real.allocate('random', seed=4)  
        
    #     res1 = self.tv_iso(x_real)
    #     res2 = self.grad.direct(x_real).pnorm(2).sum()
    #     np.testing.assert_equal(res1, res2)  


    # def test_call_real_anisotropic(self):
    #     x_real = self.ig_real.allocate('random', seed=4) 
        
    #     res1 = self.tv_aniso(x_real)
    #     res2 = self.grad.direct(x_real).pnorm(1).sum()
    #     np.testing.assert_equal(res1, res2)                
    

    # @unittest.skipUnless(has_reg_toolkit, "Regularisation Toolkit not present")
    # def test_compare_regularisation_toolkit(self):
    #     data = dataexample.SHAPES.get(size=(64,64))
    #     ig = data.geometry
    #     ag = ig

    #     np.random.seed(0)
    #     # Create noisy data. 
    #     n1 = np.random.normal(0, 0.0005, size = ig.shape)
    #     noisy_data = ig.allocate()
    #     noisy_data.fill(n1+data.as_array())
        
    #     alpha = 0.1
    #     iters = 500
            
    #     # CIL_FGP_TV no tolerance
    #     g_CIL = alpha * TotalVariation(iters, tolerance=None, lower = 0, info = True)
    #     t0 = timer()
    #     res1 = g_CIL.proximal(noisy_data, 1.)
    #     t1 = timer()
    #     # print(t1-t0)
        
    #     r_alpha = alpha
    #     r_iterations = iters
    #     r_tolerance = 1e-9
    #     r_iso = True
    #     r_nonneg = True
    #     g_CCPI_reg_toolkit = alpha * FGP_TV(max_iteration=r_iterations, tolerance=r_tolerance, 
    #          isotropic=r_iso, nonnegativity=r_nonneg, device='cpu')
        
    #     t2 = timer()
    #     res2 = g_CCPI_reg_toolkit.proximal(noisy_data, 1.)
    #     t3 = timer()
        
    #     np.testing.assert_array_almost_equal(res1.as_array(), res2.as_array(), decimal = 4)

    #     ###################################################################
    #     ###################################################################
    #     ###################################################################
    #     ###################################################################    
        
    #     # print("Compare CIL_FGP_TV vs CCPiReg_FGP_TV with iterations.")
    #     iters = 408
    #     # CIL_FGP_TV no tolerance
    #     g_CIL = alpha * TotalVariation(iters, tolerance=1e-9, lower = 0.)
    #     t0 = timer()
    #     res1 = g_CIL.proximal(noisy_data, 1.)
    #     t1 = timer()
    #     # print(t1-t0)
        
    #     r_alpha = alpha
    #     r_iterations = iters
    #     r_tolerance = 1e-9
    #     r_iso = True
    #     r_nonneg = True
    #     g_CCPI_reg_toolkit = alpha * FGP_TV(max_iteration=r_iterations, tolerance=r_tolerance, 
    #          isotropic=r_iso, nonnegativity=r_nonneg, device='cpu')

    #     t2 = timer()
    #     res2 = g_CCPI_reg_toolkit.proximal(noisy_data, 1.)
    #     t3 = timer()
    #     # print(t3-t2)
    #     np.testing.assert_array_almost_equal(res1.as_array(), res2.as_array(), decimal=3)    
        
    #     ###################################################################
    #     ###################################################################
    #     ###################################################################
    #     ###################################################################
    
    
    # @unittest.skipUnless(has_tomophantom and has_reg_toolkit, "Missing Tomophantom or Regularisation-Toolkit")
    # def test_compare_regularisation_toolkit_tomophantom(self):
    #     # print ("Building 3D phantom using TomoPhantom software")
    #     model = 13 # select a model number from the library
    #     N_size = 64 # Define phantom dimensions using a scalar value (cubic phantom)
    #     #This will generate a N_size x N_size x N_size phantom (3D)
        
    #     ig = ImageGeometry(N_size, N_size, N_size)
    #     data = TomoPhantom.get_ImageData(num_model=model, geometry=ig)

    #     noisy_data = noise.gaussian(data, seed=10)
        
    #     alpha = 0.1
    #     iters = 100
        
    #     # print("Use tau as an array of ones")
    #     # CIL_TotalVariation no tolerance
    #     g_CIL = alpha * TotalVariation(iters, tolerance=None, info=True)
    #     # res1 = g_CIL.proximal(noisy_data, ig.allocate(1.))
    #     t0 = timer()   
    #     res1 = g_CIL.proximal(noisy_data, ig.allocate(1.))
    #     t1 = timer()
    #     # print(t1-t0)

    #     # CCPi Regularisation toolkit high tolerance
        
    #     r_alpha = alpha
    #     r_iterations = iters
    #     r_tolerance = 1e-9
    #     r_iso = True
    #     r_nonneg = True
    #     g_CCPI_reg_toolkit = alpha * FGP_TV(max_iteration=r_iterations, tolerance=r_tolerance, 
    #          isotropic=r_iso, nonnegativity=r_nonneg, device='cpu')


    #     t2 = timer()
    #     res2 = g_CCPI_reg_toolkit.proximal(noisy_data, 1.)
    #     t3 = timer()
    #     # print (t3-t2)
        
    #     np.testing.assert_allclose(res1.as_array(), res2.as_array(), atol=7.5e-2)
