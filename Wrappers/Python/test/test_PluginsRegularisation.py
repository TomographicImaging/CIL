#  Copyright 2020 United Kingdom Research and Innovation
#  Copyright 2020 The University of Manchester
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
import numpy as np
from cil.utilities import dataexample
from cil.optimisation.functions import TotalVariation
from cil.framework import ImageGeometry

from utils import has_nvidia, has_ccpi_regularisation, initialise_tests

initialise_tests()

if has_ccpi_regularisation:
    from ccpi.filters import regularisers
    from cil.plugins.ccpi_regularisation.functions import FGP_TV, TGV, FGP_dTV, TNV



class TestPlugin(unittest.TestCase):
    def setUp(self):
        #Default test image
        self.data = dataexample.SIMPLE_PHANTOM_2D.get(size=(64,30))
        self.alpha = 2.0
        self.iterations = 1000


    def tearDown(self):
        pass


    @unittest.skipUnless(has_ccpi_regularisation, "Skipping as CCPi Regularisation Toolkit is not installed")
    def test_FGP_TV_complex(self):
        data = dataexample.CAMERA.get(size=(256,100))
        datarr = data.as_array()
        cmpx = np.zeros(data.shape, dtype=np.complex64)
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


    @unittest.skipUnless(has_ccpi_regularisation, "Skipping as CCPi Regularisation Toolkit is not installed")
    def test_FGP_TV_rmul(self):
        f = FGP_TV()

        self.rmul_test(f)


    @unittest.skipUnless(has_ccpi_regularisation, "Skipping as CCPi Regularisation Toolkit is not installed")
    def test_FGP_TGV_rmul(self):
        f = FGP_TGV()

        self.rmul_test(f)


    @unittest.skipUnless(has_ccpi_regularisation, "Skipping as CCPi Regularisation Toolkit is not installed")
    def test_FGP_TGV_rmul(self):
        f = TNV()

        self.rmul_test(f)


    @unittest.skipUnless(has_ccpi_regularisation, "Skipping as CCPi Regularisation Toolkit is not installed")
    def test_FGP_dTV_rmul(self):
        data = dataexample.CAMERA.get(size=(256,100))
        f = FGP_dTV(data)

        self.rmul_test(f)


    @unittest.skipUnless(has_ccpi_regularisation, "Skipping as CCPi Regularisation Toolkit is not installed")
    def test_functionality_FGP_TV(self):
        data = dataexample.CAMERA.get(size=(256,100))
        datarr = data.as_array()

        tau = 1.
        fcil = FGP_TV()
        outcil = fcil.proximal(data, tau=tau)
        # use CIL defaults
        outrgl = regularisers.FGP_TV(datarr, fcil.alpha*tau, fcil.max_iteration, fcil.tolerance, 0, 1, device='cpu' )
        np.testing.assert_almost_equal(outrgl, outcil.as_array())


    @unittest.skipUnless(has_ccpi_regularisation, "Skipping as CCPi Regularisation Toolkit is not installed")
    def test_functionality_TGV(self):
        data = dataexample.CAMERA.get(size=(256,100))
        datarr = data.as_array()

        tau = 1.
        fcil = TGV()
        outcil = fcil.proximal(data, tau=tau)
        # use CIL defaults
        outrgl = regularisers.TGV(datarr, fcil.alpha*tau, 1,1, fcil.max_iteration, 12, fcil.tolerance, device='cpu' )

        np.testing.assert_almost_equal(outrgl, outcil.as_array())


    @unittest.skipUnless(has_ccpi_regularisation, "Skipping as CCPi Regularisation Toolkit is not installed")
    def test_functionality_FGP_dTV(self):
        data = dataexample.CAMERA.get(size=(256,100))
        datarr = data.as_array()
        ref = data*0.3

        tau = 1.
        fcil = FGP_dTV(ref)
        outcil = fcil.proximal(data, tau=tau)
        # use CIL defaults
        outrgl = regularisers.FGP_dTV(datarr, ref.as_array(), fcil.alpha*tau, fcil.max_iteration, fcil.tolerance, 0.01, 0, 1, device='cpu' )
        np.testing.assert_almost_equal(outrgl, outcil.as_array())


    @unittest.skipUnless(has_ccpi_regularisation, "Skipping as CCPi Regularisation Toolkit is not installed")
    def test_functionality_TNV(self):
        # fake a 2D+channel image
        d = dataexample.SYNCHROTRON_PARALLEL_BEAM_DATA.get()
        ig = ImageGeometry(160, 135, channels=91)
        data = ig.allocate(None)
        data.fill(d.as_array())
        del d

        datarr = data.as_array()
        tau = 1.

        # CIL defaults
        outrgl = regularisers.TNV(datarr, 1, 100, 1e-6 )

        fcil = TNV()
        outcil = fcil.proximal(data, tau=tau)
        np.testing.assert_almost_equal(outrgl, outcil.as_array())


    @unittest.skipUnless(has_ccpi_regularisation, "Skipping as CCPi Regularisation Toolkit is not installed")
    def test_TNV_raise_on_2D(self):
        # data = dataexample.SYNCHROTRON_PARALLEL_BEAM_DATA.get()
        data = dataexample.CAMERA.get(size=(256,100))
        datarr = data.as_array()

        tau = 1.

        fcil = TNV()
        with self.assertRaises(ValueError):
            outcil = fcil.proximal(data, tau=tau)


    @unittest.skipUnless(has_ccpi_regularisation, "Skipping as CCPi Regularisation Toolkit is not installed")
    def test_TNV_raise_on_3D_nochannel(self):
        # data = dataexample.SYNCHROTRON_PARALLEL_BEAM_DATA.get()
        data = dataexample.CAMERA.get(size=(256,100))
        datarr = data.as_array()
        tau = 1.

        fcil = TNV()
        with self.assertRaises(ValueError):
            outcil = fcil.proximal(data, tau=tau)


    @unittest.skipUnless(has_ccpi_regularisation, "Skipping as CCPi Regularisation Toolkit is not installed")
    def test_TNV_raise_on_4D(self):

        data = ImageGeometry(3,4,5,channels=5).allocate(1)

        tau = 1.

        fcil = TNV()
        with self.assertRaises(ValueError):
            outcil = fcil.proximal(data, tau=tau)


    @unittest.skipUnless(has_ccpi_regularisation, "Skipping as CCPi Regularisation Toolkit is not installed")
    def test_FGP_TV_raise_on_4D_data(self):

        tau = 1.
        fcil = FGP_TV()
        data = ImageGeometry(3,4,5,channels=10).allocate(0)


        with self.assertRaises(ValueError):
            outcil = fcil.proximal(data, tau=tau)


    @unittest.skipUnless(has_ccpi_regularisation, "Skipping as CCPi Regularisation Toolkit is not installed")
    def test_TGV_raise_on_4D_data(self):

        tau = 1.
        fcil = TGV()
        data = ImageGeometry(3,4,5,channels=10).allocate(0)


        with self.assertRaises(ValueError):
            outcil = fcil.proximal(data, tau=tau)


    @unittest.skipUnless(has_ccpi_regularisation, "Skipping as CCPi Regularisation Toolkit is not installed")
    def test_FGP_dTV_raise_on_4D_data(self):

        tau = 1.

        data = ImageGeometry(3,4,5,channels=10).allocate(0)
        ref = data * 2

        fcil = FGP_dTV(ref)

        with self.assertRaises(ValueError):
            outcil = fcil.proximal(data, tau=tau)


    @unittest.skipUnless(has_ccpi_regularisation, "Skipping as CCPi Regularisation Toolkit is not installed")
    def test_TotalVariation_vs_FGP_TV_cpu(self):
        # Isotropic TV cil
        TV_cil_iso = self.alpha * TotalVariation(max_iteration=self.iterations, warm_start=False)

        # Anisotropic TV cil
        TV_cil_aniso = self.alpha * TotalVariation(max_iteration=self.iterations, isotropic=False, warm_start=False)

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

    @unittest.skipUnless(has_ccpi_regularisation, "Skipping as CCPi Regularisation Toolkit is not installed")
    def test_TotalVariation_warm_start_vs_FGP_TV_cpu(self):
        # Isotropic TV cil
        TV_cil_iso = self.alpha * TotalVariation(max_iteration=self.iterations, warm_start=True)

        # Anisotropic TV cil
        TV_cil_aniso = self.alpha * TotalVariation(max_iteration=self.iterations, isotropic=False, warm_start=True)

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

    @unittest.skipUnless(has_ccpi_regularisation and has_nvidia, "Skipping as CCPi Regularisation Toolkit is not installed")
    def test_TotalVariation_vs_FGP_TV_gpu(self):
        # Isotropic TV cil
        TV_cil_iso = self.alpha * TotalVariation(max_iteration=self.iterations, warm_start=False)
        res_TV_cil_iso = TV_cil_iso.proximal(self.data, tau=1.0)

        # Anisotropic TV cil
        TV_cil_aniso = self.alpha * TotalVariation(max_iteration=self.iterations, isotropic=False, warm_start=False)
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
