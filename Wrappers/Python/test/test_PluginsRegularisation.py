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

import sys
import unittest
import numpy
import numpy as np
from cil.framework import DataContainer
from cil.framework import ImageData
from cil.framework import AcquisitionData
from cil.framework import ImageGeometry
from cil.framework import AcquisitionGeometry
from cil.utilities import dataexample
from timeit import default_timer as timer


try:
    from ccpi.filters import regularisers
    from ccpi.filters.cpu_regularisers import TV_ENERGY
    from cil.plugins.ccpi_regularisation.functions import FGP_TV
    has_regularisation_toolkit = True
except ImportError as ie:
    # raise ImportError(ie + "\n\n" + 
    #                   "This plugin requires the additional package ccpi-regularisation\n" +
    #                   "Please install it via conda as ccpi-regularisation from the ccpi channel\n"+
    #                   "Minimal version is 20.04")
    has_regularisation_toolkit = False
print ("has_regularisation_toolkit", has_regularisation_toolkit)
TNV_fixed = False

class TestPlugin(unittest.TestCase):
    def setUp(self):
        # print ("test plugins")
        pass
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
