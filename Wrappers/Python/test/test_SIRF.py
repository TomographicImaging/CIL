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
from cil.optimisation.operators import GradientOperator
from utils import GradientSIRF
try:
    import sirf.STIR as pet
    import sirf.Gadgetron as mr
    from sirf.Utilities import examples_data_path
    has_sirf = True
except ImportError as ie:
    has_sirf = False


class TestGradientPET_2D(unittest.TestCase, GradientSIRF):  

    def setUp(self):

        if has_sirf:
            self.image1 = pet.ImageData(os.path.join(
                examples_data_path('PET'),'thorax_single_slice','emission.hv')
                )

    def tearDown(self):
        pass    

class TestGradientPET_3D(unittest.TestCase, GradientSIRF):  

    def setUp(self):

        if has_sirf:
            self.image1 = pet.ImageData(os.path.join(
                examples_data_path('PET'),'brain','emission.hv')
                )

    def tearDown(self):
        pass  

class TestGradientMR_2D(unittest.TestCase, GradientSIRF):  

    def setUp(self):

        if has_sirf:
            acq_data = mr.AcquisitionData(os.path.join
                (examples_data_path('MR'),'simulated_MR_2D_cartesian.h5')
            )
            preprocessed_data = mr.preprocess_acquisition_data(acq_data)
            recon = mr.FullySampledReconstructor()
            recon.set_input(preprocessed_data)
            recon.process()
            self.image1 = recon.get_output()
        
    def tearDown(self):
        pass      

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
        
        tmp = image1.divide(1.)
        numpy.testing.assert_array_equal(image1.as_array(), tmp.as_array())
        tmp = image2.divide(1.)
        numpy.testing.assert_array_equal(image2.as_array(), tmp.as_array())
        

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
    def test_BlockDataContainer_with_SIRF_DataContainer_subtract(self):
        os.chdir(self.cwd)
        image1 = pet.ImageData('emission.hv')
        image2 = pet.ImageData('emission.hv')
        image1.fill(2)
        image2.fill(1)

        
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
