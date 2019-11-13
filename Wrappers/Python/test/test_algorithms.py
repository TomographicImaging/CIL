# -*- coding: utf-8 -*-
#  CCP in Tomographic Imaging (CCPi) Core Imaging Library (CIL).

#   Copyright 2017 UKRI-STFC
#   Copyright 2017 University of Manchester

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
from ccpi.framework import DataContainer
from ccpi.framework import ImageData
from ccpi.framework import AcquisitionData
from ccpi.framework import ImageGeometry
from ccpi.framework import AcquisitionGeometry
from ccpi.optimisation.operators import Identity
from ccpi.optimisation.functions import Norm2Sq, ZeroFunction, \
   L2NormSquared, FunctionOperatorComposition
from ccpi.optimisation.algorithms import GradientDescent
from ccpi.optimisation.algorithms import CGLS
from ccpi.optimisation.algorithms import FISTA

from ccpi.optimisation.algorithms import PDHG

from ccpi.optimisation.operators import Gradient, BlockOperator, FiniteDiff
from ccpi.optimisation.functions import MixedL21Norm, BlockFunction, L1Norm, KullbackLeibler                     
from ccpi.framework import TestData
import os ,sys

class TestAlgorithms(unittest.TestCase):
    def setUp(self):
        #wget.download('https://github.com/DiamondLightSource/Savu/raw/master/test_data/data/24737_fd.nxs')
        #self.filename = '24737_fd.nxs'
        # we use Identity as the operator and solve the simple least squares 
        # problem for a random-valued ImageData or AcquisitionData b?  
        # Then we know the minimiser is b itself
        
        # || I x -b ||^2
        
        # create an ImageGeometry
        ig = ImageGeometry(12,13,14)
        pass

    def tearDown(self):
        #os.remove(self.filename)
        pass
    
    def test_GradientDescent(self):
        print ("Test GradientDescent")
        ig = ImageGeometry(12,13,14)
        x_init = ig.allocate()
        # b = x_init.copy()
        # fill with random numbers
        # b.fill(numpy.random.random(x_init.shape))
        b = ig.allocate('random')
        identity = Identity(ig)
        
        norm2sq = Norm2Sq(identity, b)
        rate = 0.3
        rate = norm2sq.L / 3.
        
        alg = GradientDescent(x_init=x_init, 
                              objective_function=norm2sq, 
                              rate=rate)
        alg.max_iteration = 20
        alg.run(20, verbose=True)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())
        alg = GradientDescent(x_init=x_init, 
                              objective_function=norm2sq, 
                              rate=rate, max_iteration=20,
                              update_objective_interval=2)
        alg.max_iteration = 20
        self.assertTrue(alg.max_iteration == 20)
        self.assertTrue(alg.update_objective_interval==2)
        alg.run(20, verbose=True)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())
    def test_CGLS(self):
        print ("Test CGLS")
        #ig = ImageGeometry(124,153,154)
        ig = ImageGeometry(10,2)
        numpy.random.seed(2)
        x_init = ig.allocate(0.)
        b = ig.allocate('random')
        # b = x_init.copy()
        # fill with random numbers
        # b.fill(numpy.random.random(x_init.shape))
        # b = ig.allocate()
        # bdata = numpy.reshape(numpy.asarray([i for i in range(20)]), (2,10))
        # b.fill(bdata)
        identity = Identity(ig)
        
        alg = CGLS(x_init=x_init, operator=identity, data=b)
        alg.max_iteration = 200
        alg.run(20, verbose=True)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())

        alg = CGLS(x_init=x_init, operator=identity, data=b, max_iteration=200, update_objective_interval=2)
        self.assertTrue(alg.max_iteration == 200)
        self.assertTrue(alg.update_objective_interval==2)
        alg.run(20, verbose=True)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())
        
    def test_FISTA(self):
        print ("Test FISTA")
        ig = ImageGeometry(127,139,149)
        x_init = ig.allocate()
        b = x_init.copy()
        # fill with random numbers
        b.fill(numpy.random.random(x_init.shape))
        x_init = ig.allocate(ImageGeometry.RANDOM)
        identity = Identity(ig)
        
	#### it seems FISTA does not work with Nowm2Sq
        # norm2sq = Norm2Sq(identity, b)
        # norm2sq.L = 2 * norm2sq.c * identity.norm()**2
        norm2sq = FunctionOperatorComposition(L2NormSquared(b=b), identity)
        opt = {'tol': 1e-4, 'memopt':False}
        print ("initial objective", norm2sq(x_init))
        alg = FISTA(x_init=x_init, f=norm2sq, g=ZeroFunction())
        alg.max_iteration = 2
        alg.run(20, verbose=True)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())

        alg = FISTA(x_init=x_init, f=norm2sq, g=ZeroFunction(), max_iteration=2, update_objective_interval=2)
        
        self.assertTrue(alg.max_iteration == 2)
        self.assertTrue(alg.update_objective_interval==2)

        alg.run(20, verbose=True)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())

               
    def test_FISTA_Norm2Sq(self):
        print ("Test FISTA Norm2Sq")
        ig = ImageGeometry(127,139,149)
        b = ig.allocate(ImageGeometry.RANDOM)
        # fill with random numbers
        x_init = ig.allocate(ImageGeometry.RANDOM)
        identity = Identity(ig)
        
	    #### it seems FISTA does not work with Nowm2Sq
        norm2sq = Norm2Sq(identity, b)
        #norm2sq.L = 2 * norm2sq.c * identity.norm()**2
        #norm2sq = FunctionOperatorComposition(L2NormSquared(b=b), identity)
        opt = {'tol': 1e-4, 'memopt':False}
        print ("initial objective", norm2sq(x_init))
        alg = FISTA(x_init=x_init, f=norm2sq, g=ZeroFunction())
        alg.max_iteration = 2
        alg.run(20, verbose=True)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())

        alg = FISTA(x_init=x_init, f=norm2sq, g=ZeroFunction(), max_iteration=2, update_objective_interval=3)
        self.assertTrue(alg.max_iteration == 2)
        self.assertTrue(alg.update_objective_interval== 3)

        alg.run(20, verbose=True)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())

    def test_FISTA_catch_Lipschitz(self):
        print ("Test FISTA catch Lipschitz")
        ig = ImageGeometry(127,139,149)
        x_init = ImageData(geometry=ig)
        x_init = ig.allocate()
        b = x_init.copy()
        # fill with random numbers
        b.fill(numpy.random.random(x_init.shape))
        x_init = ig.allocate(ImageGeometry.RANDOM)
        identity = Identity(ig)
        
	    #### it seems FISTA does not work with Nowm2Sq
        norm2sq = Norm2Sq(identity, b)
        print ('Lipschitz', norm2sq.L)
        norm2sq.L = None
        #norm2sq.L = 2 * norm2sq.c * identity.norm()**2
        #norm2sq = FunctionOperatorComposition(L2NormSquared(b=b), identity)
        opt = {'tol': 1e-4, 'memopt':False}
        print ("initial objective", norm2sq(x_init))
        try:
            alg = FISTA(x_init=x_init, f=norm2sq, g=ZeroFunction())
            self.assertTrue(False)
        except ValueError as ve:
            print (ve)
            self.assertTrue(True)
    def test_PDHG_Denoising(self):
        print ("PDHG Denoising with 3 noises")
        # adapted from demo PDHG_TV_Color_Denoising.py in CIL-Demos repository
        
        # loader = TestData(data_dir=os.path.join(os.environ['SIRF_INSTALL_PATH'], 'share','ccpi'))
        # loader = TestData(data_dir=os.path.join(sys.prefix, 'share','ccpi'))
        loader = TestData()
        
        data = loader.load(TestData.PEPPERS, size=(256,256))
        ig = data.geometry
        ag = ig

        which_noise = 0
        # Create noisy data. 
        noises = ['gaussian', 'poisson', 's&p']
        noise = noises[which_noise]
        
        def setup(data, noise):
            if noise == 's&p':
                n1 = TestData.random_noise(data.as_array(), mode = noise, salt_vs_pepper = 0.9, amount=0.2, seed=10)
            elif noise == 'poisson':
                scale = 5
                n1 = TestData.random_noise( data.as_array()/scale, mode = noise, seed = 10)*scale
            elif noise == 'gaussian':
                n1 = TestData.random_noise(data.as_array(), mode = noise, seed = 10)
            else:
                raise ValueError('Unsupported Noise ', noise)
            noisy_data = ig.allocate()
            noisy_data.fill(n1)
        
            # Regularisation Parameter depending on the noise distribution
            if noise == 's&p':
                alpha = 0.8
            elif noise == 'poisson':
                alpha = 1
            elif noise == 'gaussian':
                alpha = .3
                # fidelity
            if noise == 's&p':
                g = L1Norm(b=noisy_data)
            elif noise == 'poisson':
                g = KullbackLeibler(noisy_data)
            elif noise == 'gaussian':
                g = 0.5 * L2NormSquared(b=noisy_data)
            return noisy_data, alpha, g

        noisy_data, alpha, g = setup(data, noise)
        operator = Gradient(ig, correlation=Gradient.CORRELATION_SPACE)

        f1 =  alpha * MixedL21Norm()

        
                    
        # Compute operator Norm
        normK = operator.norm()

        # Primal & dual stepsizes
        sigma = 1
        tau = 1/(sigma*normK**2)

        # Setup and run the PDHG algorithm
        pdhg1 = PDHG(f=f1,g=g,operator=operator, tau=tau, sigma=sigma)
        pdhg1.max_iteration = 2000
        pdhg1.update_objective_interval = 200
        pdhg1.run(1000, very_verbose=True)

        rmse = (pdhg1.get_output() - data).norm() / data.as_array().size
        print ("RMSE", rmse)
        self.assertLess(rmse, 2e-4)

        which_noise = 1
        noise = noises[which_noise]
        noisy_data, alpha, g = setup(data, noise)
        operator = Gradient(ig, correlation=Gradient.CORRELATION_SPACE)

        f1 =  alpha * MixedL21Norm()

        
                    
        # Compute operator Norm
        normK = operator.norm()

        # Primal & dual stepsizes
        sigma = 1
        tau = 1/(sigma*normK**2)

        # Setup and run the PDHG algorithm
        pdhg1 = PDHG(f=f1,g=g,operator=operator, tau=tau, sigma=sigma, 
                     max_iteration=2000, update_objective_interval=200)
        
        pdhg1.run(1000)

        rmse = (pdhg1.get_output() - data).norm() / data.as_array().size
        print ("RMSE", rmse)
        self.assertLess(rmse, 2e-4)
        
        
        which_noise = 2
        noise = noises[which_noise]
        noisy_data, alpha, g = setup(data, noise)
        operator = Gradient(ig, correlation=Gradient.CORRELATION_SPACE)

        f1 =  alpha * MixedL21Norm()

        
                    
        # Compute operator Norm
        normK = operator.norm()

        # Primal & dual stepsizes
        sigma = 1
        tau = 1/(sigma*normK**2)

        # Setup and run the PDHG algorithm
        pdhg1 = PDHG(f=f1,g=g,operator=operator, tau=tau, sigma=sigma)
        pdhg1.max_iteration = 2000
        pdhg1.update_objective_interval = 200
        pdhg1.run(1000)

        rmse = (pdhg1.get_output() - data).norm() / data.as_array().size
        print ("RMSE", rmse)
        self.assertLess(rmse, 2e-4)

    def test_FISTA_Denoising(self):
        print ("FISTA Denoising Poisson Noise Tikhonov")
        # adapted from demo FISTA_Tikhonov_Poisson_Denoising.py in CIL-Demos repository
        #loader = TestData(data_dir=os.path.join(sys.prefix, 'share','ccpi'))
        loader = TestData()
        data = loader.load(TestData.SHAPES)
        ig = data.geometry
        ag = ig
        N=300
        # Create Noisy data with Poisson noise
        scale = 5
        n1 = TestData.random_noise( data.as_array()/scale, mode = 'poisson', seed = 10)*scale
        noisy_data = ImageData(n1)

        # Regularisation Parameter
        alpha = 10

        # Setup and run the FISTA algorithm
        operator = Gradient(ig)
        fid = KullbackLeibler(noisy_data)

        def KL_Prox_PosCone(x, tau, out=None):
                
            if out is None: 
                tmp = 0.5 *( (x - fid.bnoise - tau) + ( (x + fid.bnoise - tau)**2 + 4*tau*fid.b   ) .sqrt() )
                return tmp.maximum(0)
            else:            
                tmp =  0.5 *( (x - fid.bnoise - tau) + 
                            ( (x + fid.bnoise - tau)**2 + 4*tau*fid.b   ) .sqrt()
                            )
                x.add(fid.bnoise, out=out)
                out -= tau
                out *= out
                tmp = fid.b * (4 * tau)
                out.add(tmp, out=out)
                out.sqrt(out=out)
                
                x.subtract(fid.bnoise, out=tmp)
                tmp -= tau
                
                out += tmp
                
                out *= 0.5
                
                # ADD the constraint here
                out.maximum(0, out=out)
                
        fid.proximal = KL_Prox_PosCone

        reg = FunctionOperatorComposition(alpha * L2NormSquared(), operator)

        x_init = ig.allocate()
        fista = FISTA(x_init=x_init , f=reg, g=fid)
        fista.max_iteration = 3000
        fista.update_objective_interval = 500
        fista.run(verbose=True)
        rmse = (fista.get_output() - data).norm() / data.as_array().size
        print ("RMSE", rmse)
        self.assertLess(rmse, 4.2e-4)

    def assertNumpyArrayEqual(self, first, second):
        res = True
        try:
            numpy.testing.assert_array_equal(first, second)
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

    def assertNumpyArrayAlmostEqual(self, first, second, decimal=6):
        res = True
        try:
            numpy.testing.assert_array_almost_equal(first, second, decimal)
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)
        
    
        


if __name__ == '__main__':
    unittest.main()
 
