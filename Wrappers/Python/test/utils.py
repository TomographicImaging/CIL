try:
    import tigre
    has_tigre = True
except ModuleNotFoundError:
    has_tigre = False
else:
    from cil.plugins.tigre import ProjectionOperator
    from tigre.utilities.errors import TigreCudaCallError

try:
    import astra
    has_astra = True
except ModuleNotFoundError:
    has_astra = False

try:
    import sirf.STIR as pet
    import sirf.Gadgetron as mr
    from sirf.Utilities import examples_data_path
    has_sirf = True
except ImportError as ie:
    has_sirf = False    

import os
from cil.framework import AcquisitionGeometry, BlockDataContainer
from cil.optimisation.operators import GradientOperator, LinearOperator
import numpy as np
import unittest

def has_nvidia_smi():
    return os.system('nvidia-smi') == 0


def has_gpu_tigre():
    print ("has_gpu_tigre")
    if not has_nvidia_smi():
        return False
    N = 3
    angles = np.linspace(0, np.pi, 2, dtype='float32')

    ag = AcquisitionGeometry.create_Cone2D([0,-100],[0,200])\
                            .set_angles(angles, angle_unit='radian')\
                            .set_panel(N, 0.1)\
                            .set_labels(['angle', 'horizontal'])
    
    ig = ag.get_ImageGeometry()
    
    data = ig.allocate(1)

    Op = ProjectionOperator(ig, ag)

    has_gpu = True
    try:
        res = Op.direct(data)
    except TigreCudaCallError:
        has_gpu = False
    return has_gpu

def has_gpu_astra():
    print ("has_gpu_astra")
    if not has_nvidia_smi():
        return False
    has_gpu = True
    try:
        astra.test_CUDA()
    except:
        has_gpu = False
    return has_gpu

def has_ipp():
    print ("has_ipp")
    from cil.reconstructors import FBP
    try:
        fbp = FBP('fail')
    except ImportError:
        return False
    except TypeError:
        return True


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
        
            if isinstance(self.image1, pet.ImageData):
                self.assertTrue(isinstance(res1[i], pet.ImageData))
                self.assertTrue(isinstance(res2[i], pet.ImageData)) 
            else:
                self.assertTrue(isinstance(res1[i], mr.ImageData))
                self.assertTrue(isinstance(res2[i], mr.ImageData))                                 
            # test direct with and without out
            np.testing.assert_array_almost_equal(res1[i].as_array(), res2[i].as_array())                         

        # test adjoint with and without out
        res3 = Grad_numpy.adjoint(res1)
        res4 = Grad_numpy.domain_geometry().allocate()
        Grad_numpy.adjoint(res2, out=res4)
        np.testing.assert_array_almost_equal(res3.as_array(), res4.as_array()) 

        # test dot_test
        self.assertTrue(LinearOperator.dot_test(Grad_numpy, decimal=3))

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
        self.assertTrue(LinearOperator.dot_test(Grad_c, decimal=3))

        # test adjoint
        res7 = Grad_c.adjoint(res5) 
        res8 = Grad_c.domain_geometry().allocate()*0.
        Grad_c.adjoint(res5, out=res8)
        np.testing.assert_array_almost_equal(res7.as_array(), res8.as_array()) 


    
