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

import numpy
from numpy.linalg import norm
import os
import sys
import shutil
import unittest
from cil.framework import BlockDataContainer
from cil.optimisation.operators import LinearOperator, GradientOperator
try:
    import sirf.STIR as pet
    from sirf.Utilities import examples_data_path
    has_sirf = True
except ImportError as ie:
    has_sirf = False


class TestSIRFCILIntegration(unittest.TestCase):
    
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
        image1 = pet.ImageData('emission.hv')
        image2 = pet.ImageData('emission.hv')
        image1.fill(1.)
        image2.fill(2.)
        print (image1.shape, image2.shape)
        
        tmp = image1.divide(1.)
        # numpy.testing.assert_array_equal(image1.as_array(), tmp.as_array())
        tmp = image2.divide(1.)
        # numpy.testing.assert_array_equal(image2.as_array(), tmp.as_array())
        

        # image.fill(1.)
        bdc = BlockDataContainer(image1, image2)
        bdc1 = bdc.divide(1.)

        self.assertBlockDataContainerEqual(bdc , bdc1)

    @unittest.skipUnless(has_sirf, "Has SIRF")
    def test_BlockDataContainer_with_SIRF_DataContainer_multiply(self):
        os.chdir(self.cwd)
        image1 = pet.ImageData('emission.hv')
        image2 = pet.ImageData('emission.hv')
        image1.fill(1.)
        image2.fill(2.)
        print (image1.shape, image2.shape)
        
        tmp = image1.multiply(1.)
        numpy.testing.assert_array_equal(image1.as_array(), tmp.as_array())
        tmp = image2.multiply(1.)
        numpy.testing.assert_array_equal(image2.as_array(), tmp.as_array())
        

        # image.fill(1.)
        bdc = BlockDataContainer(image1, image2)
        bdc1 = bdc.multiply(1.)

        self.assertBlockDataContainerEqual(bdc , bdc1)
    
    @unittest.skipUnless(has_sirf, "Has SIRF")
    def test_BlockDataContainer_with_SIRF_DataContainer_add(self):
        os.chdir(self.cwd)
        image1 = pet.ImageData('emission.hv')
        image2 = pet.ImageData('emission.hv')
        image1.fill(0)
        image2.fill(1)
        print (image1.shape, image2.shape)
        
        tmp = image1.add(1.)
        numpy.testing.assert_array_equal(image2.as_array(), tmp.as_array())
        tmp = image2.subtract(1.)
        numpy.testing.assert_array_equal(image1.as_array(), tmp.as_array())
        
        bdc = BlockDataContainer(image1, image2)
        bdc1 = bdc.add(1.)

        image1.fill(1)
        image2.fill(2)

        bdc = BlockDataContainer(image1, image2)

        self.assertBlockDataContainerEqual(bdc , bdc1)

    @unittest.skipUnless(has_sirf, "Has SIRF")
    def test_GradientSIRF_2D_pseudo_geometries(self):

        os.chdir(examples_data_path('PET'))
        shutil.rmtree('working_folder/thorax_single_slice',True)
        shutil.copytree('thorax_single_slice','working_folder/thorax_single_slice')
        os.chdir('working_folder/thorax_single_slice')

        image1 = pet.ImageData('emission.hv')
        self.assertTrue(isinstance(image1, pet.ImageData))

        ########################################
        ##### Test Gradient numpy backend  #####
        ########################################

        Grad_numpy = GradientOperator(image1, backend='numpy')

        res1 = Grad_numpy.direct(image1)         
        res2 = Grad_numpy.range_geometry().allocate()
        Grad_numpy.direct(image1, out=res2)

        self.assertTrue(isinstance(res1,BlockDataContainer))
        self.assertTrue(isinstance(res1[0], pet.ImageData))
        self.assertTrue(isinstance(res1[1], pet.ImageData))

        self.assertTrue(isinstance(res2,BlockDataContainer))
        self.assertTrue(isinstance(res2[0], pet.ImageData))
        self.assertTrue(isinstance(res2[1], pet.ImageData))               

        # test direct with and without out
        numpy.testing.assert_array_almost_equal(res1[0].as_array(), res2[0].as_array()) 
        numpy.testing.assert_array_almost_equal(res1[1].as_array(), res2[1].as_array()) 

        # test adjoint with and without out
        res3 = Grad_numpy.adjoint(res1)
        res4 = Grad_numpy.domain_geometry().allocate()
        Grad_numpy.adjoint(res2, out=res4)
        numpy.testing.assert_array_almost_equal(res3.as_array(), res4.as_array()) 

        # test dot_test
        self.assertTrue(LinearOperator.dot_test(Grad_numpy))

        # test shape of output of direct
        self.assertEqual(res1[0].shape, image1.shape)
        self.assertEqual(res1.shape, (2,1))

        ########################################
        ##### Test Gradient c backend  #####
        ########################################
        Grad_c = GradientOperator(image1, backend='c')

        # test direct with and without out
        res5 = Grad_c.direct(image1) 
        res6 = Grad_c.range_geometry().allocate()*0.
        Grad_c.direct(image1, out=res6)

        numpy.testing.assert_array_almost_equal(res5[0].as_array(), res6[0].as_array()) 
        numpy.testing.assert_array_almost_equal(res5[1].as_array(), res6[1].as_array())

        # test direct numpy vs direct c backends (with and without out)
        numpy.testing.assert_array_almost_equal(res5[0].as_array(), res1[0].as_array()) 
        numpy.testing.assert_array_almost_equal(res6[1].as_array(), res2[1].as_array())

        # test dot_test
        self.assertTrue(LinearOperator.dot_test(Grad_c))

        # test adjoint
        res7 = Grad_c.adjoint(res5) 
        res8 = Grad_c.domain_geometry().allocate()*0.
        Grad_c.adjoint(res5, out=res8)
        numpy.testing.assert_array_almost_equal(res7.as_array(), res8.as_array()) 


    @unittest.skipUnless(has_sirf, "Has SIRF")
    def test_BlockDataContainer_with_SIRF_DataContainer_subtract(self):
        os.chdir(self.cwd)
        image1 = pet.ImageData('emission.hv')
        image2 = pet.ImageData('emission.hv')
        image1.fill(2)
        image2.fill(1)
        print (image1.shape, image2.shape)
        
        bdc = BlockDataContainer(image1, image2)
        bdc1 = bdc.subtract(1.)

        image1.fill(1)
        image2.fill(0)

        bdc = BlockDataContainer(image1, image2)

        self.assertBlockDataContainerEqual(bdc , bdc1)


    def assertBlockDataContainerEqual(self, container1, container2):
        print ("assert Block Data Container Equal")
        self.assertTrue(issubclass(container1.__class__, container2.__class__))
        for col in range(container1.shape[0]):
            if hasattr(container1.get_item(col), 'as_array'):
                print ("Checking col ", col)
                self.assertNumpyArrayEqual(
                    container1.get_item(col).as_array(), 
                    container2.get_item(col).as_array()
                    )
            else:
                self.assertBlockDataContainerEqual(container1.get_item(col),container2.get_item(col))
    
    def assertNumpyArrayEqual(self, first, second):
        numpy.testing.assert_array_equal(first, second)
