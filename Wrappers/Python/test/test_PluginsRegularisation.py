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
import numpy as np
from cil.utilities import dataexample
from cil.optimisation.functions import TotalVariation
import os
from utils import has_nvidia_smi
from cil.framework import ImageGeometry

try:
    from ccpi.filters import regularisers
    from ccpi.filters.cpu_regularisers import TV_ENERGY
    from cil.plugins.ccpi_regularisation.functions import FGP_TV
    has_regularisation_toolkit = True
except ImportError as ie:
    has_regularisation_toolkit = False
print ("has_regularisation_toolkit", has_regularisation_toolkit)
TNV_fixed = False

class TestPlugin(unittest.TestCase):
    def setUp(self):
        #Default test image
        self.data = dataexample.SIMPLE_PHANTOM_2D.get(size=(64,64))
        self.alpha = 2.0
        self.iterations = 1000     
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
        data = dataexample.CAMERA.get(size=(256,256))
        datarr = data.as_array()
        cmpx = np.zeros(data.shape, dtype=np.complex)
        cmpx.real = datarr[:]
        cmpx.imag = datarr[:]
        data.array = cmpx
        reg = FGP_TV()
        out = reg.proximal(data, 1)
        outarr = out.as_array()
        np.testing.assert_almost_equal(outarr.imag, outarr.real)

    def rmul_test(self, f):

        alpha = f.alpha
        scalar = 2.123
        af = scalar*f

        assert (id(af) == id(f))
        assert af.alpha == scalar * alpha
    
    @unittest.skipUnless(has_regularisation_toolkit, "Skipping as CCPi Regularisation Toolkit is not installed")
    def test_FGP_TV_rmul(self):
        from cil.plugins.ccpi_regularisation.functions import FGP_TV
        f = FGP_TV()

        self.rmul_test(f)
    
    @unittest.skipUnless(has_regularisation_toolkit, "Skipping as CCPi Regularisation Toolkit is not installed")
    def test_FGP_TGV_rmul(self):
        from cil.plugins.ccpi_regularisation.functions import FGP_TGV
        f = FGP_TGV()

        self.rmul_test(f)

    @unittest.skipUnless(has_regularisation_toolkit, "Skipping as CCPi Regularisation Toolkit is not installed")
    def test_FGP_TGV_rmul(self):
        from cil.plugins.ccpi_regularisation.functions import TNV
        f = TNV()

        self.rmul_test(f)
    @unittest.skipUnless(has_regularisation_toolkit, "Skipping as CCPi Regularisation Toolkit is not installed")
    def test_FGP_dTV_rmul(self):
        from cil.plugins.ccpi_regularisation.functions import FGP_dTV
        data = dataexample.CAMERA.get(size=(256,256))
        f = FGP_dTV(data)

        self.rmul_test(f)
        

    @unittest.skipUnless(has_regularisation_toolkit, "Skipping as CCPi Regularisation Toolkit is not installed")
    def test_functionality_FGP_TV(self):

        data = dataexample.CAMERA.get(size=(256,256))
        datarr = data.as_array()
        from cil.plugins.ccpi_regularisation.functions import FGP_TV
        from ccpi.filters import regularisers

        tau = 1.
        fcil = FGP_TV()
        outcil = fcil.proximal(data, tau=tau)
        # use CIL defaults
        outrgl, info = regularisers.FGP_TV(datarr, fcil.alpha*tau, fcil.max_iteration, fcil.tolerance, 0, 1, 'cpu' )
        np.testing.assert_almost_equal(outrgl, outcil.as_array())

    @unittest.skipUnless(has_regularisation_toolkit, "Skipping as CCPi Regularisation Toolkit is not installed")
    def test_functionality_TGV(self):

        data = dataexample.CAMERA.get(size=(256,256))
        datarr = data.as_array()
        from cil.plugins.ccpi_regularisation.functions import TGV
        from ccpi.filters import regularisers

        tau = 1.
        fcil = TGV()
        outcil = fcil.proximal(data, tau=tau)
        # use CIL defaults
        outrgl, info = regularisers.TGV(datarr, fcil.alpha*tau, 1,1, fcil.max_iteration, 12, fcil.tolerance, 'cpu' )

        np.testing.assert_almost_equal(outrgl, outcil.as_array())

    @unittest.skipUnless(has_regularisation_toolkit, "Skipping as CCPi Regularisation Toolkit is not installed")
    def test_functionality_FGP_dTV(self):

        data = dataexample.CAMERA.get(size=(256,256))
        datarr = data.as_array()
        ref = data*0.3
        from cil.plugins.ccpi_regularisation.functions import FGP_dTV
        from ccpi.filters import regularisers

        tau = 1.
        fcil = FGP_dTV(ref)
        outcil = fcil.proximal(data, tau=tau)
        # use CIL defaults
        outrgl, info = regularisers.FGP_dTV(datarr, ref.as_array(), fcil.alpha*tau, fcil.max_iteration, fcil.tolerance, 0.01, 0, 1, 'cpu' )
        np.testing.assert_almost_equal(outrgl, outcil.as_array())

    @unittest.skipUnless(has_regularisation_toolkit, "Skipping as CCPi Regularisation Toolkit is not installed")
    def test_functionality_TNV(self):

        # fake a 2D+channel image
        d = dataexample.SYNCHROTRON_PARALLEL_BEAM_DATA.get()
        ig = ImageGeometry(160, 135, channels=91)
        data = ig.allocate(None)
        data.fill(d)
        del d
    
        datarr = data.as_array()
        from cil.plugins.ccpi_regularisation.functions import TNV
        from ccpi.filters import regularisers

        tau = 1.
        
        # CIL defaults
        outrgl = regularisers.TNV(datarr, 1, 100, 1e-6 )
        
        fcil = TNV()
        outcil = fcil.proximal(data, tau=tau)
        np.testing.assert_almost_equal(outrgl, outcil.as_array())

    @unittest.skipUnless(has_regularisation_toolkit, "Skipping as CCPi Regularisation Toolkit is not installed")
    def test_TNV_raise_on_2D(self):

        # data = dataexample.SYNCHROTRON_PARALLEL_BEAM_DATA.get()
        data = dataexample.CAMERA.get(size=(256,256))
        datarr = data.as_array()
        from cil.plugins.ccpi_regularisation.functions import TNV
        
        tau = 1.
        
        fcil = TNV()
        try:
            outcil = fcil.proximal(data, tau=tau)
            assert False
        except ValueError:
            assert True
    @unittest.skipUnless(has_regularisation_toolkit, "Skipping as CCPi Regularisation Toolkit is not installed")
    def test_TNV_raise_on_3D_nochannel(self):

        # data = dataexample.SYNCHROTRON_PARALLEL_BEAM_DATA.get()
        data = dataexample.CAMERA.get(size=(256,256))
        datarr = data.as_array()
        from cil.plugins.ccpi_regularisation.functions import TNV
        
        tau = 1.
        
        fcil = TNV()
        try:
            outcil = fcil.proximal(data, tau=tau)
            assert False
        except ValueError:
            assert True
    @unittest.skipUnless(has_regularisation_toolkit, "Skipping as CCPi Regularisation Toolkit is not installed")
    def test_TNV_raise_on_4D(self):

        from cil.plugins.ccpi_regularisation.functions import TNV
        
        data = ImageGeometry(3,4,5,channels=5).allocate(1)

        tau = 1.
        
        fcil = TNV()
        try:
            outcil = fcil.proximal(data, tau=tau)
            assert False
        except ValueError:
            assert True

    @unittest.skipUnless(has_regularisation_toolkit, "Skipping as CCPi Regularisation Toolkit is not installed")
    def test_FGP_TV_raise_on_4D_data(self):

        from cil.plugins.ccpi_regularisation.functions import FGP_TV
        
        tau = 1.
        fcil = FGP_TV()
        data = ImageGeometry(3,4,5,channels=10).allocate(0)


        try:
            outcil = fcil.proximal(data, tau=tau)
            assert False
        except ValueError:
            assert True

    @unittest.skipUnless(has_regularisation_toolkit, "Skipping as CCPi Regularisation Toolkit is not installed")
    def test_TGV_raise_on_4D_data(self):

        from cil.plugins.ccpi_regularisation.functions import TGV
        
        tau = 1.
        fcil = TGV()
        data = ImageGeometry(3,4,5,channels=10).allocate(0)


        try:
            outcil = fcil.proximal(data, tau=tau)
            assert False
        except ValueError:
            assert True
    @unittest.skipUnless(has_regularisation_toolkit, "Skipping as CCPi Regularisation Toolkit is not installed")
    def test_FGP_dTV_raise_on_4D_data(self):

        from cil.plugins.ccpi_regularisation.functions import FGP_dTV
        
        tau = 1.
        
        data = ImageGeometry(3,4,5,channels=10).allocate(0)
        ref = data * 2
        
        fcil = FGP_dTV(ref)

        try:
            outcil = fcil.proximal(data, tau=tau)
            assert False
        except ValueError:
            assert True

    @unittest.skipUnless(has_regularisation_toolkit, "Skipping as CCPi Regularisation Toolkit is not installed")
    def test_TotalVariation_vs_FGP_TV_cpu(self):

        # Isotropic TV cil
        TV_cil_iso = self.alpha * TotalVariation(max_iteration=self.iterations)

        # Anisotropic TV cil
        TV_cil_aniso = self.alpha * TotalVariation(max_iteration=self.iterations, isotropic=False)

        # Isotropic FGP_TV CCPiReg toolkit (cpu)
        TV_regtoolkit_cpu_iso = self.alpha * FGP_TV(max_iteration=self.iterations, device = 'cpu')

        # Anisotropic FGP_TV CCPiReg toolkit (cpu)
        TV_regtoolkit_cpu_aniso = self.alpha * FGP_TV(max_iteration=self.iterations, device = 'cpu', isotropic=False)

        res_TV_cil_iso = TV_cil_iso.proximal(self.data, tau=1.0)
        res_TV_cil_aniso = TV_cil_aniso.proximal(self.data, tau=1.0)
        res_TV_regtoolkit_cpu_iso = TV_regtoolkit_cpu_iso.proximal(self.data, tau=1.0)
        res_TV_regtoolkit_cpu_aniso = TV_regtoolkit_cpu_aniso.proximal(self.data, tau=1.0)  

        # compare TV vs FGP_TV (anisotropic, isotropic, cpu)
        np.testing.assert_array_almost_equal(res_TV_cil_iso.array, res_TV_regtoolkit_cpu_iso.array, decimal=3)              
        np.testing.assert_array_almost_equal(res_TV_cil_aniso.array, res_TV_regtoolkit_cpu_aniso.array, decimal=3)
       
    @unittest.skipUnless(has_regularisation_toolkit and has_nvidia_smi(), "Skipping as CCPi Regularisation Toolkit is not installed")  
    def test_TotalVariation_vs_FGP_TV_gpu(self):   

        # Isotropic TV cil
        TV_cil_iso = self.alpha * TotalVariation(max_iteration=self.iterations)
        res_TV_cil_iso = TV_cil_iso.proximal(self.data, tau=1.0)        

        # Anisotropic TV cil
        TV_cil_aniso = self.alpha * TotalVariation(max_iteration=self.iterations, isotropic=False) 
        res_TV_cil_aniso = TV_cil_aniso.proximal(self.data, tau=1.0)               
        
        # Isotropic FGP_TV CCPiReg toolkit (gpu)
        TV_regtoolkit_gpu_iso = self.alpha * FGP_TV(max_iteration=self.iterations, device = 'gpu') 
        res_TV_regtoolkit_gpu_iso = TV_regtoolkit_gpu_iso.proximal(self.data, tau=1.0)

        # Anisotropic FGP_TV CCPiReg toolkit (gpu)
        TV_regtoolkit_gpu_aniso = self.alpha * FGP_TV(max_iteration=self.iterations, device = 'gpu', isotropic=False)  
        res_TV_regtoolkit_gpu_aniso = TV_regtoolkit_gpu_aniso.proximal(self.data, tau=1.0)  

        # Anisotropic FGP_TV CCPiReg toolkit (cpu)
        TV_regtoolkit_cpu_aniso = self.alpha * FGP_TV(max_iteration=self.iterations, device = 'cpu', isotropic=False)         
        res_TV_regtoolkit_cpu_aniso = TV_regtoolkit_cpu_aniso.proximal(self.data, tau=1.0)          

        # Isotropic FGP_TV CCPiReg toolkit (cpu)
        TV_regtoolkit_cpu_iso = self.alpha * FGP_TV(max_iteration=self.iterations, device = 'cpu')
        res_TV_regtoolkit_cpu_iso = TV_regtoolkit_cpu_iso.proximal(self.data, tau=1.0)        

        np.testing.assert_array_almost_equal(res_TV_cil_iso.array, res_TV_regtoolkit_gpu_iso.array, decimal=3)
        np.testing.assert_array_almost_equal(res_TV_regtoolkit_cpu_iso.array, res_TV_regtoolkit_gpu_iso.array, decimal=3)

        np.testing.assert_array_almost_equal(res_TV_cil_aniso.array, res_TV_regtoolkit_gpu_aniso.array, decimal=3)
        np.testing.assert_array_almost_equal(res_TV_regtoolkit_cpu_aniso.array, res_TV_regtoolkit_gpu_aniso.array, decimal=3)              
