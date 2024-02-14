# -*- coding: utf-8 -*-
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

import os
import shutil
import unittest

import numpy as np

from cil.framework import AcquisitionData, ImageData, BlockDataContainer
from cil.optimisation.algorithms import FISTA
from cil.optimisation.functions import TotalVariation, L2NormSquared, KullbackLeibler
from cil.optimisation.operators import GradientOperator, LinearOperator
from testclass import CCPiTestClass
from utils import has_ccpi_regularisation, initialise_tests

initialise_tests()

try:
    import sirf.STIR as pet
    import sirf.Gadgetron as mr
    import sirf.Reg as reg
    from sirf.Utilities import examples_data_path
    
    has_sirf = True
except ImportError as ie:
    has_sirf = False

if has_ccpi_regularisation:
    from cil.plugins.ccpi_regularisation.functions import FGP_TV, TGV, FGP_dTV, TNV



class KullbackLeiblerSIRF(object):

    def setUp(self):

        if has_sirf:
            self.image1 = ImageData(os.path.join(
                examples_data_path('PET'),'thorax_single_slice','emission.hv')
                )

            self.eta = self.image1.get_uniform_copy(0.1)
            self.x = self.image1.get_uniform_copy(0.4)                    

        self.f_np = KullbackLeibler(b = self.image1, backend='numpy')  
        self.f1_np = KullbackLeibler(b = self.image1, eta = self.eta,  backend='numpy') 
        self.out_np = self.image1.get_uniform_copy(0.)
        self.out_nb = self.image1.get_uniform_copy(0.)

        self.f_nb = KullbackLeibler(b = self.image1, backend='numba')  
        self.f1_nb = KullbackLeibler(b = self.image1, eta = self.eta,  backend='numba')         
        self.out1_np = self.image1.get_uniform_copy(0.)
        self.out1_nb = self.image1.get_uniform_copy(0.)

        self.tau = 400.4      

    def tearDown(self):
        pass
    
    @unittest.skipUnless(has_sirf, "Skipping as SIRF is not available")
    def test_KullbackLeibler_call(self):
        np.testing.assert_almost_equal(self.f_np(self.x), self.f_nb(self.x), decimal = 2)
        np.testing.assert_almost_equal(self.f1_np(self.x), self.f1_nb(self.x), decimal = 2)


    @unittest.skipUnless(has_sirf, "Skipping as SIRF is not available")
    def test_KullbackLeibler_gradient(self):

        self.f_np.gradient(self.x, out = self.out_np)
        self.f_nb.gradient(self.x, out = self.out_nb)
        self.f1_np.gradient(self.x, out = self.out1_np)
        self.f1_nb.gradient(self.x, out = self.out1_nb)        

        np.testing.assert_array_almost_equal(self.out_np.as_array(), self.out_nb.as_array(), decimal = 2)
        np.testing.assert_array_almost_equal(self.out1_np.as_array(), self.out1_nb.as_array(), decimal = 2)


    @unittest.skipUnless(has_sirf, "Skipping as SIRF is not available")
    def test_KullbackLeibler_convex_conjugate(self):
       
        np.testing.assert_almost_equal(self.f_np.convex_conjugate(self.x), self.f_nb.convex_conjugate(self.x), decimal = 2)        
        np.testing.assert_almost_equal(self.f1_np.convex_conjugate(self.x), self.f1_nb.convex_conjugate(self.x), decimal = 2)


    @unittest.skipUnless(has_sirf, "Skipping as SIRF is not available")
    def test_KullbackLeibler_proximal(self):

        self.f_np.proximal(self.x, self.tau, out = self.out_np)
        self.f_nb.proximal(self.x, self.tau, out = self.out_nb)
        self.f1_np.proximal(self.x, self.tau, out = self.out1_np)
        self.f1_nb.proximal(self.x, self.tau, out = self.out1_nb)  

        np.testing.assert_array_almost_equal(self.out_np.as_array(), self.out_nb.as_array(), decimal = 2)
        np.testing.assert_array_almost_equal(self.out1_np.as_array(), self.out1_nb.as_array(), decimal = 2)


    @unittest.skipUnless(has_sirf, "Skipping as SIRF is not available")
    def test_KullbackLeibler_proximal_conjugate(self):

        self.f_np.proximal_conjugate(self.x, self.tau, out = self.out_np)
        self.f_nb.proximal_conjugate(self.x, self.tau, out = self.out_nb)
        self.f1_np.proximal_conjugate(self.x, self.tau, out = self.out1_np)
        self.f1_nb.proximal_conjugate(self.x, self.tau, out = self.out1_nb)  

        np.testing.assert_array_almost_equal(self.out_np.as_array(), self.out_nb.as_array(), decimal = 2)
        np.testing.assert_array_almost_equal(self.out1_np.as_array(), self.out1_nb.as_array(), decimal = 2)


class GradientSIRF(object):
    
    @unittest.skipUnless(has_sirf, "Skipping as SIRF is not available")
    def test_Gradient(self):

        #######################################
        ##### Test Gradient numpy backend #####
        #######################################

        Grad_numpy = GradientOperator(self.image1, backend='numpy')

        res1 = Grad_numpy.direct(self.image1)         
        res2 = Grad_numpy.range_geometry().allocate()
        Grad_numpy.direct(self.image1, out=res2)

        self.assertTrue(isinstance(res1,BlockDataContainer))
        self.assertTrue(isinstance(res2,BlockDataContainer))

        for i in range(len(res1)):
        
            if isinstance(self.image1, ImageData):
                self.assertTrue(isinstance(res1[i], ImageData))
                self.assertTrue(isinstance(res2[i], ImageData))
            else:
                self.assertTrue(isinstance(res1[i], ImageData))
                self.assertTrue(isinstance(res2[i], ImageData))
            # test direct with and without out
            np.testing.assert_array_almost_equal(res1[i].as_array(), res2[i].as_array())                         

        # test adjoint with and without out
        res3 = Grad_numpy.adjoint(res1)
        res4 = Grad_numpy.domain_geometry().allocate()
        Grad_numpy.adjoint(res2, out=res4)
        np.testing.assert_array_almost_equal(res3.as_array(), res4.as_array()) 

        # test dot_test
        for sd in [5,10]:
            
            self.assertTrue(LinearOperator.dot_test(Grad_numpy, seed=sd))

        # test shape of output of direct
        
        # check in the case of pseudo 2D data, e.g., (1, 155, 155)
        if 1 in self.image1.shape:
            self.assertEqual(res1.shape, (2,1))
        else:
            self.assertEqual(res1.shape, (3,1))            

        ########################################
        ##### Test Gradient c backend  #####
        ########################################
        Grad_c = GradientOperator(self.image1, backend='c')

        # test direct with and without out
        res5 = Grad_c.direct(self.image1) 
        res6 = Grad_c.range_geometry().allocate()*0.
        Grad_c.direct(self.image1, out=res6)

        for i in range(len(res5)):
            np.testing.assert_array_almost_equal(res5[i].as_array(), res6[i].as_array())

            # compare c vs numpy gradient backends 
            np.testing.assert_array_almost_equal(res6[i].as_array(), res2[i].as_array())


        # test dot_test
        for sd in [5,10]:
            self.assertTrue(LinearOperator.dot_test(Grad_c, seed=sd))

        # test adjoint
        res7 = Grad_c.adjoint(res5) 
        res8 = Grad_c.domain_geometry().allocate()*0.
        Grad_c.adjoint(res5, out=res8)
        np.testing.assert_array_almost_equal(res7.as_array(), res8.as_array()) 



class TestGradientPET_2D(unittest.TestCase, GradientSIRF):  

    def setUp(self):

        if has_sirf:
            self.image1 = ImageData(os.path.join(
                examples_data_path('PET'),'thorax_single_slice','emission.hv')
                )

    def tearDown(self):
        pass    

class TestGradientPET_3D(unittest.TestCase, GradientSIRF):  

    def setUp(self):

        if has_sirf:
            self.image1 = ImageData(os.path.join(
                examples_data_path('PET'),'brain','emission.hv')
                )

    def tearDown(self):
        pass  

class TestGradientMR_2D(unittest.TestCase, GradientSIRF):  

    def setUp(self):

        if has_sirf:
            acq_data = AcquisitionData(os.path.join
                (examples_data_path('MR'),'simulated_MR_2D_cartesian.h5')
                                                                     )
            preprocessed_data = mr.preprocess_acquisition_data(acq_data)
            recon = mr.FullySampledReconstructor()
            recon.set_input(preprocessed_data)
            recon.process()
            self.image1 = recon.get_output()
        
    def tearDown(self):
        pass      

    @unittest.skipUnless(has_sirf, "Has SIRF")
    def test_TVdenoisingMR(self):
        
        # compare inplace proximal method of TV
        alpha = 0.5
        TV = alpha * TotalVariation(max_iteration=10, warm_start=False)
        res1 = TV.proximal(self.image1, tau=1.0)

        res2 = self.image1*0.
        TV.proximal(self.image1, tau=1.0, out=res2)   
        np.testing.assert_array_almost_equal(res1.as_array(), res2.as_array(), decimal=3)

        # compare with FISTA algorithm   
        f =  0.5 * L2NormSquared(b=self.image1)
        fista = FISTA(initial=self.image1*0.0, f=f, g=TV, max_iteration=10, update_objective_interval=10)
        fista.run(verbose=0)
        np.testing.assert_array_almost_equal(fista.solution.as_array(), res2.as_array(), decimal=3)      

        

  

    
class TestSIRFCILIntegration(CCPiTestClass):
    
    def setUp(self):
        if has_sirf:
            os.chdir(examples_data_path('PET'))
            # Copy files to a working folder and change directory to where these files are.
            # We do this to avoid cluttering your SIRF files. This way, you can delete 
            # working_folder and start from scratch.
            shutil.rmtree('working_folder/brain',True)
            shutil.copytree('brain','working_folder/brain')
            os.chdir('working_folder/brain')

            self.cwd = os.getcwd()

    
    def tearDown(self):
        if has_sirf:
            shutil.rmtree(self.cwd)


    @unittest.skipUnless(has_sirf, "Has SIRF")
    def test_BlockDataContainer_with_SIRF_DataContainer_divide(self):
        os.chdir(self.cwd)
        image1 = ImageData('emission.hv')
        image2 = ImageData('emission.hv')
        image1.fill(1.)
        image2.fill(2.)
        
        tmp = image1.divide(1.)
        np.testing.assert_array_equal(image1.as_array(), tmp.as_array())
        tmp = image2.divide(1.)
        np.testing.assert_array_equal(image2.as_array(), tmp.as_array())
        

        # image.fill(1.)
        bdc = BlockDataContainer(image1, image2)
        bdc1 = bdc.divide(1.)

        # self.assertBlockDataContainerEqual(bdc , bdc1)
        np.testing.assert_allclose(bdc.get_item(0).as_array(), bdc1.get_item(0).as_array())
        np.testing.assert_allclose(bdc.get_item(1).as_array(), bdc1.get_item(1).as_array())


    @unittest.skipUnless(has_sirf, "Has SIRF")
    def test_BlockDataContainer_with_SIRF_DataContainer_multiply(self):
        os.chdir(self.cwd)
        image1 = ImageData('emission.hv')
        image2 = ImageData('emission.hv')
        image1.fill(1.)
        image2.fill(2.)
        
        tmp = image1.multiply(1.)
        np.testing.assert_array_equal(image1.as_array(), tmp.as_array())
        tmp = image2.multiply(1.)
        np.testing.assert_array_equal(image2.as_array(), tmp.as_array())
        

        # image.fill(1.)
        bdc = BlockDataContainer(image1, image2)
        bdc1 = bdc.multiply(1.)

        # self.assertBlockDataContainerEqual(bdc , bdc1)
        np.testing.assert_allclose(bdc.get_item(0).as_array(), bdc1.get_item(0).as_array())
        np.testing.assert_allclose(bdc.get_item(1).as_array(), bdc1.get_item(1).as_array())
    

    @unittest.skipUnless(has_sirf, "Has SIRF")
    def test_BlockDataContainer_with_SIRF_DataContainer_add(self):
        os.chdir(self.cwd)
        image1 = ImageData('emission.hv')
        image2 = ImageData('emission.hv')
        image1.fill(0)
        image2.fill(1)
        
        tmp = image1.add(1.)
        np.testing.assert_array_equal(image2.as_array(), tmp.as_array())
        tmp = image2.subtract(1.)
        np.testing.assert_array_equal(image1.as_array(), tmp.as_array())
        
        bdc = BlockDataContainer(image1, image2)
        bdc1 = bdc.add(1.)

        image1.fill(1)
        image2.fill(2)

        bdc = BlockDataContainer(image1, image2)

        np.testing.assert_allclose(bdc.get_item(0).as_array(), bdc1.get_item(0).as_array())
        np.testing.assert_allclose(bdc.get_item(1).as_array(), bdc1.get_item(1).as_array())
        # self.assertBlockDataContainerEqual(bdc , bdc1)


    @unittest.skipUnless(has_sirf, "Has SIRF")
    def test_BlockDataContainer_with_SIRF_DataContainer_subtract(self):
        os.chdir(self.cwd)
        image1 = ImageData('emission.hv')
        image2 = ImageData('emission.hv')
        image1.fill(2)
        image2.fill(1)

        
        bdc = BlockDataContainer(image1, image2)
        bdc1 = bdc.subtract(1.)

        image1.fill(1)
        image2.fill(0)

        bdc = BlockDataContainer(image1, image2)

        # self.assertBlockDataContainerEqual(bdc , bdc1)
        np.testing.assert_allclose(bdc.get_item(0).as_array(), bdc1.get_item(0).as_array())
        np.testing.assert_allclose(bdc.get_item(1).as_array(), bdc1.get_item(1).as_array())



class CCPiRegularisationWithSIRFTests():
    
    def setUpFGP_TV(self, max_iteration=100, alpha=1.):
        return alpha*FGP_TV(max_iteration=max_iteration)

    @unittest.skipUnless(has_sirf and has_ccpi_regularisation, "Has SIRF and CCPi Regularisation")
    def test_FGP_TV_call_works(self):
        regulariser = self.setUpFGP_TV()
        output_number = regulariser(self.image1)
        self.assertTrue(True)
        # TODO: test the actual value
        # expected = 160600016.0
        # np.testing.assert_allclose(output_number, expected, rtol=1e-5)
    
    @unittest.skipUnless(has_sirf and has_ccpi_regularisation, "Has SIRF and CCPi Regularisation")
    def test_FGP_TV_proximal_works(self):
        regulariser = self.setUpFGP_TV()
        solution = regulariser.proximal(x=self.image1, tau=1)
        self.assertTrue(True)

    # TGV
    def setUpTGV(self, max_iteration=100, alpha=1.):
        return alpha * TGV(max_iteration=max_iteration)

    @unittest.skipUnless(has_sirf and has_ccpi_regularisation, "Has SIRF and CCPi Regularisation")
    def test_TGV_call_works(self):
        regulariser = self.setUpTGV()
        output_number = regulariser(self.image1)
        self.assertTrue(True)

    @unittest.skipUnless(has_sirf and has_ccpi_regularisation, "Has SIRF and CCPi Regularisation")
    def test_TGV_proximal_works(self):
        regulariser = self.setUpTGV()
        solution = regulariser.proximal(x=self.image1, tau=1)
        self.assertTrue(True)
        
    # dTV
    def setUpdTV(self, max_iteration=100, alpha=1.):
        return alpha * FGP_dTV(reference=self.image2, max_iteration=max_iteration)

    @unittest.skipUnless(has_sirf and has_ccpi_regularisation, "Has SIRF and CCPi Regularisation")
    def test_TGV_call_works(self):
        regulariser = self.setUpTGV()
        output_number = regulariser(self.image1)
        self.assertTrue(True)

    @unittest.skipUnless(has_sirf and has_ccpi_regularisation, "Has SIRF and CCPi Regularisation")
    def test_TGV_proximal_works(self):
        regulariser = self.setUpTGV()
        solution = regulariser.proximal(x=self.image1, tau=1)
        self.assertTrue(True)

    # TNV
    def setUpTNV(self, max_iteration=100, alpha=1.):
        return alpha * TNV(max_iteration=max_iteration)

    @unittest.skipUnless(has_sirf and has_ccpi_regularisation, "Has SIRF and CCPi Regularisation")
    def test_TNV_call_works(self):
        new_shape = [ i for i in self.image1.shape if i!=1]
        if len(new_shape) == 3:
            regulariser = self.setUpTNV()
            output_number = regulariser(self.image1)
            self.assertTrue(True)

    @unittest.skipUnless(has_sirf and has_ccpi_regularisation, "Has SIRF and CCPi Regularisation")
    def test_TNV_proximal_works(self):
        new_shape = [ i for i in self.image1.shape if i!=1]
        if len(new_shape) == 3:
            regulariser = self.setUpTNV()
            solution = regulariser.proximal(x=self.image1, tau=1.)
            self.assertTrue(True)

class TestPETRegularisation(unittest.TestCase, CCPiRegularisationWithSIRFTests):
    skip_TNV_on_2D = True
    def setUp(self):
        self.image1 = ImageData(os.path.join(
            examples_data_path('PET'),'thorax_single_slice','emission.hv'
            ))
        self.image2 = self.image1 * 0.5

    @unittest.skipIf(skip_TNV_on_2D, "TNV not implemented for 2D")
    def test_TNV_call_works(self):
        super().test_TNV_call_works()
    
    @unittest.skipIf(skip_TNV_on_2D, "TNV not implemented for 2D")
    def test_TNV_proximal_works(self):
        super().test_TNV_proximal_works()
        
class TestRegRegularisation(unittest.TestCase, CCPiRegularisationWithSIRFTests):
    def setUp(self):
        self.image1 = ImageData(os.path.join(examples_data_path('Registration'), 'test2.nii.gz'))
        self.image2 = self.image1 * 0.5

class TestMRRegularisation(unittest.TestCase, CCPiRegularisationWithSIRFTests):
    def setUp(self):
        acq_data = AcquisitionData(os.path.join(examples_data_path('MR'), 'simulated_MR_2D_cartesian.h5'))
        preprocessed_data = mr.preprocess_acquisition_data(acq_data)
        recon = mr.FullySampledReconstructor()
        recon.set_input(preprocessed_data)
        recon.process()
        self.image1 = recon.get_output()
        self.image2 = self.image1 * 0.5
