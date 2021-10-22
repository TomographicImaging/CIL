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

import unittest
import numpy
from cil.utilities import dataexample
from cil.optimisation.functions import TotalVariation
import os

try:
    from ccpi.filters import regularisers
    from ccpi.filters.cpu_regularisers import TV_ENERGY
    from cil.plugins.ccpi_regularisation.functions import FGP_TV
    has_regularisation_toolkit = True
except ImportError as ie:
    has_regularisation_toolkit = False
print ("has_regularisation_toolkit", has_regularisation_toolkit)
TNV_fixed = False

def has_nvidia_smi():
    return os.system('nvidia-smi') == 0


class TestPlugin(unittest.TestCase):
    def setUp(self):

        #Default test image
        self.data = dataexample.SIMPLE_PHANTOM_2D.get(size=(64,64))

    def tearDown(self):
        pass
    @unittest.skipUnless(has_regularisation_toolkit, "Skipping as CCPi Regularisation Toolkit is not installed")
    def test_import_FGP_TV(self):
        try:
            from cil.plugins.ccpi_regularisation.functions.regularisers import FGP_TV
            assert True
        except ModuleNotFoundError as ie:
            print (ie)
            assert False
    
    @unittest.skipUnless(has_regularisation_toolkit, "Skipping as CCPi Regularisation Toolkit is not installed")
    def test_import_TGV(self):
        try:
            from cil.plugins.ccpi_regularisation.functions import TGV
            assert True
        except ModuleNotFoundError as ie:
            assert False
    
    @unittest.skipUnless(has_regularisation_toolkit, "Skipping as CCPi Regularisation Toolkit is not installed")
    def test_import_FGP_dTV(self):
        try:
            from cil.plugins.ccpi_regularisation.functions import FGP_dTV
            assert True
        except ModuleNotFoundError as ie:
            assert False
    
    @unittest.skipUnless(has_regularisation_toolkit, "Skipping as CCPi Regularisation Toolkit is not installed")
    def test_import_TNV(self):
        try:
            from cil.plugins.ccpi_regularisation.functions import TNV
            assert True
        except ModuleNotFoundError as ie:
            assert False

    @unittest.skipUnless(has_regularisation_toolkit, "Skipping as CCPi Regularisation Toolkit is not installed")
    def test_FGP_TV_complex(self):
        datarr = self.data.as_array()
        cmpx = numpy.zeros(self.data.shape, dtype=numpy.complex)
        cmpx.real = datarr[:]
        cmpx.imag = datarr[:]
        self.data.array = cmpx
        reg = FGP_TV()
        out = reg.proximal(self.data, 1)
        outarr = out.as_array()
        numpy.testing.assert_almost_equal(outarr.imag, outarr.real)

    def test_TotalVariation_vs_FGP_TV(self):

        alpha = 2.0
        iterations = 500

        # Isotropic TV cil
        TV_cil_iso = alpha * TotalVariation(max_iteration=iterations)

        # Anisotropic TV cil
        TV_cil_aniso = alpha * TotalVariation(max_iteration=iterations, isotropic=False)

        # Isotropic FGP_TV CCPiReg toolkit (cpu)
        TV_regtoolkit_cpu_iso = alpha * FGP_TV(max_iteration=iterations, device = 'cpu')

        # Anisotropic FGP_TV CCPiReg toolkit (cpu)
        TV_regtoolkit_cpu_aniso = alpha * FGP_TV(max_iteration=iterations, device = 'cpu', isotropic=False)

        res_TV_cil_iso = TV_cil_iso.proximal(self.data, tau=1.0)
        res_TV_cil_aniso = TV_cil_aniso.proximal(self.data, tau=1.0)
        res_TV_regtoolkit_cpu_iso = TV_regtoolkit_cpu_iso.proximal(self.data, tau=1.0)
        res_TV_regtoolkit_cpu_aniso = TV_regtoolkit_cpu_aniso.proximal(self.data, tau=1.0)  

        # compare TV vs FGP_TV (anisotropic, isotropic, cpu)
        numpy.testing.assert_array_almost_equal(res_TV_cil_iso.array, res_TV_regtoolkit_cpu_iso.array, decimal=3)              
        numpy.testing.assert_array_almost_equal(res_TV_cil_aniso.array, res_TV_regtoolkit_cpu_aniso.array, decimal=3)
       
        if has_nvidia_smi:

            try:
                # Isotropic FGP_TV CCPiReg toolkit (gpu)
                TV_regtoolkit_gpu_iso = alpha * FGP_TV(max_iteration=iterations, device = 'gpu') 

                # Anisotropic FGP_TV CCPiReg toolkit (gpu)
                TV_regtoolkit_gpu_aniso = alpha * FGP_TV(max_iteration=iterations, device = 'gpu', isotropic=False)  

                res_TV_regtoolkit_gpu_iso = TV_regtoolkit_gpu_iso.proximal(self.data, tau=1.0)
                res_TV_regtoolkit_gpu_aniso = TV_regtoolkit_gpu_aniso.proximal(self.data, tau=1.0)   

                numpy.testing.assert_array_almost_equal(res_TV_cil_iso.array, res_TV_regtoolkit_gpu_iso.array, decimal=3)
                numpy.testing.assert_array_almost_equal(res_TV_regtoolkit_cpu_iso.array, res_TV_regtoolkit_gpu_iso.array, decimal=3)

                numpy.testing.assert_array_almost_equal(res_TV_cil_aniso.array, res_TV_regtoolkit_gpu_aniso.array, decimal=3)
                numpy.testing.assert_array_almost_equal(res_TV_regtoolkit_cpu_aniso.array, res_TV_regtoolkit_gpu_aniso.array, decimal=3)    
            except:
                print("No GPU available")    
