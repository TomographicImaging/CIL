# Copyright 2022 United Kingdom Research and Innovation
# Copyright 2022 The University of Manchester

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from cil.utilities import dataexample
from cil.optimisation.functions import TotalGeneralisedVariation, LeastSquares
from cil.optimisation.algorithms import FISTA
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
        self.num_iter = 500

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

        fista_cil = FISTA(initial=tmp_initial, f=f, g=g_cil, max_iteration=100)
        fista_cil.run()

        fista_ccpi_regtk = FISTA(initial=tmp_initial, f=f, g=g_ccpi_regtk, 
                                update_objective_interval=50, max_iteration=100)
        fista_ccpi_regtk.run() 

        np.testing.assert_allclose(fista_cil.solution.array, fista_ccpi_regtk.solution.array, atol=1e-2) 
       
   



       







    