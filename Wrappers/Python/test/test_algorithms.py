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
from __future__ import division

import unittest
import numpy
import numpy as np
from ccpi.framework import DataContainer
from ccpi.framework import ImageData
from ccpi.framework import AcquisitionData
from ccpi.framework import ImageGeometry
from ccpi.framework import AcquisitionGeometry
from ccpi.optimisation.operators import Identity
from ccpi.optimisation.functions import LeastSquares, ZeroFunction, \
   L2NormSquared, FunctionOperatorComposition
from ccpi.optimisation.algorithms import GradientDescent
from ccpi.optimisation.algorithms import CGLS
from ccpi.optimisation.algorithms import FISTA

from ccpi.optimisation.algorithms import PDHG

from ccpi.optimisation.operators import Gradient, BlockOperator, FiniteDiff
from ccpi.optimisation.functions import MixedL21Norm, BlockFunction, L1Norm, KullbackLeibler                     
from ccpi.framework import TestData
import os ,sys


try:
    from ccpi.astra.operators import AstraProjectorSimple
    astra_not_available = False    
except ImportError as ie:
    # skip test
    astra_not_available = True

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
        
        norm2sq = LeastSquares(identity, b)
        rate = norm2sq.L / 3.
        
        alg = GradientDescent(x_init=x_init, 
                              objective_function=norm2sq, 
                              rate=rate, atol=1e-9, rtol=1e-6)
        alg.max_iteration = 1000
        alg.run()
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())
        alg = GradientDescent(x_init=x_init, 
                              objective_function=norm2sq, 
                              rate=rate, max_iteration=20,
                              update_objective_interval=2,
                              atol=1e-9, rtol=1e-6)
        alg.max_iteration = 20
        self.assertTrue(alg.max_iteration == 20)
        self.assertTrue(alg.update_objective_interval==2)
        alg.run(20, verbose=True)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())
    def test_GradientDescentArmijo(self):
        print ("Test GradientDescent")
        ig = ImageGeometry(12,13,14)
        x_init = ig.allocate()
        # b = x_init.copy()
        # fill with random numbers
        # b.fill(numpy.random.random(x_init.shape))
        b = ig.allocate('random')
        identity = Identity(ig)
        
        norm2sq = LeastSquares(identity, b)
        rate = None
        
        alg = GradientDescent(x_init=x_init, 
                              objective_function=norm2sq, rate=rate)
        alg.max_iteration = 100
        alg.run()
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())
        alg = GradientDescent(x_init=x_init, 
                              objective_function=norm2sq, 
                              max_iteration=20,
                              update_objective_interval=2)
        #alg.max_iteration = 20
        self.assertTrue(alg.max_iteration == 20)
        self.assertTrue(alg.update_objective_interval==2)
        alg.run(20, verbose=True)
        self.assertNumpyArrayAlmostEqual(alg.x.as_array(), b.as_array())
    def test_GradientDescentArmijo2(self):
        from ccpi.optimisation.functions import Rosenbrock
        from ccpi.framework import VectorData, VectorGeometry

        f = Rosenbrock (alpha = 1., beta=100.)
        vg = VectorGeometry(2)
        x = vg.allocate('random_int', seed=2)
        # x = vg.allocate('random', seed=1) 
        x.fill(numpy.asarray([10.,-3.]))
        
        max_iter = 1000000
        update_interval = 100000

        alg = GradientDescent(x, f, max_iteration=max_iter, update_objective_interval=update_interval, alpha=1e6)
        
        alg.run()
        
        print (alg.get_output().as_array(), alg.step_size, alg.kmax, alg.k)

        numpy.testing.assert_array_almost_equal(alg.get_output().as_array(), [1,1], decimal = 1)
        numpy.testing.assert_array_almost_equal(alg.get_output().as_array(), [0.982744, 0.965725], decimal = 6)

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
        norm2sq = LeastSquares(identity, b)
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
        norm2sq = LeastSquares(identity, b)
        print ('Lipschitz', norm2sq.L)
        # norm2sq.L = None
        #norm2sq.L = 2 * norm2sq.c * identity.norm()**2
        #norm2sq = FunctionOperatorComposition(L2NormSquared(b=b), identity)
        opt = {'tol': 1e-4, 'memopt':False}
        print ("initial objective", norm2sq(x_init))
        try:
            alg = FISTA(x_init=x_init, f=L1Norm(), g=ZeroFunction())
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
                g = KullbackLeibler(b=noisy_data)
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
        fid = KullbackLeibler(b=noisy_data)
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

    @unittest.skipIf(astra_not_available, "ccpi-astra not available")
    def test_SPDHG_vs_PDHG_implicit(self):
        from ccpi.astra.operators import AstraProjectorSimple
        from ccpi.framework import BlockDataContainer, AcquisitionData, AcquisitionGeometry, ImageData, ImageGeometry
        from ccpi.optimisation.operators import BlockOperator, Gradient
        from ccpi.optimisation.functions import BlockFunction, KullbackLeibler, MixedL21Norm, IndicatorBox
        from ccpi.optimisation.algorithms import SPDHG, PDHG
        # Fast Gradient Projection algorithm for Total Variation(TV)
        from ccpi.optimisation.functions import TotalVariation
        loader = TestData()
        data = loader.load(TestData.SIMPLE_PHANTOM_2D, size=(128,128))
        print ("here")
        ig = data.geometry
        ig.voxel_size_x = 0.1
        ig.voxel_size_y = 0.1
            
        detectors = ig.shape[0]
        angles = np.linspace(0, np.pi, 180)
        ag = AcquisitionGeometry('parallel','2D',angles, detectors, pixel_size_h = 0.1)
        # Select device
        # device = input('Available device: GPU==1 / CPU==0 ')
        # if device=='1':
        #     dev = 'gpu'
        # else:
        #     dev = 'cpu'
        dev = 'gpu'

        Aop = AstraProjectorSimple(ig, ag, dev)
        
        sin = Aop.direct(data)
        # Create noisy data. Apply Gaussian noise
        noises = ['gaussian', 'poisson']
        noise = noises[1]
        if noise == 'poisson':
            np.random.seed(10)
            scale = 5
            eta = 0
            noisy_data = AcquisitionData(np.random.poisson( scale * (eta + sin.as_array()))/scale, ag)
        elif noise == 'gaussian':
            np.random.seed(10)
            n1 = np.random.normal(0, 0.1, size = ag.shape)
            noisy_data = AcquisitionData(n1 + sin.as_array(), ag)
            
        else:
            raise ValueError('Unsupported Noise ', noise)
        
        # Create BlockOperator
        operator = Aop 
        f = KullbackLeibler(b=noisy_data)        
        alpha = 0.5
        g =  TotalVariation(alpha, 50, 1e-4, lower=0)   
        normK = operator.norm()
        sigma = 1/normK
        tau = 1/normK
            
        # Setup and run the PDHG algorithm
        pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma)
        pdhg.max_iteration = 1000
        pdhg.update_objective_interval = 200
        #pdhg.run(200, very_verbose = True)

        #%% 'implicit' PDHG, preconditioned step-sizes


        #normK = operator.norm()
        tau_tmp = 1
        sigma_tmp = 1
        tau = sigma_tmp / operator.adjoint(tau_tmp * operator.range_geometry().allocate(1.))
        sigma = tau_tmp / operator.direct(sigma_tmp * operator.domain_geometry().allocate(1.))
        x_init = operator.domain_geometry().allocate()

        # Setup and run the PDHG algorithm
        pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma)
        pdhg.max_iteration = 1000
        pdhg.update_objective_interval = 200
        pdhg.run(1000, very_verbose = True)

        subsets = 10
        size_of_subsets = int(len(angles)/subsets)
        # take angles and create uniform subsets in uniform+sequential setting
        list_angles = [angles[i:i+size_of_subsets] for i in range(0, len(angles), size_of_subsets)]
        # create acquisitioin geometries for each the interval of splitting angles
        list_geoms = [AcquisitionGeometry('parallel','2D',list_angles[i], detectors, pixel_size_h = 0.1) 
                        for i in range(len(list_angles))]
        # create with operators as many as the subsets
        A = BlockOperator(*[AstraProjectorSimple(ig, list_geoms[i], dev) for i in range(subsets)])
        ## number of subsets
        #(sub2ind, ind2sub) = divide_1Darray_equally(range(len(A)), subsets)
        #
        ## acquisisiton data
        g = BlockDataContainer(*[AcquisitionData(noisy_data.as_array()[i:i+size_of_subsets,:])
                                    for i in range(0, len(angles), size_of_subsets)])
        alpha = 0.5
        ## block function
        F = BlockFunction(*[KullbackLeibler(b=g[i]) for i in range(subsets)]) 
        G = TotalVariation(alpha, 50, 1e-4, lower=0) 

        prob = [1/len(A)]*len(A)
        spdhg = SPDHG(f=F,g=G,operator=A, 
                    max_iteration = 1000,
                    update_objective_interval=200, prob = prob)
        spdhg.run(1000, very_verbose = True)
        from ccpi.utilities.quality_measures import mae, mse, psnr
        qm = (mae(spdhg.get_output(), pdhg.get_output()),
            mse(spdhg.get_output(), pdhg.get_output()),
            psnr(spdhg.get_output(), pdhg.get_output())
            )
        print ("Quality measures", qm)
         
        np.testing.assert_almost_equal( mae(spdhg.get_output(), pdhg.get_output()), 0.0028578834608197212, decimal=5)
        np.testing.assert_almost_equal( mse(spdhg.get_output(), pdhg.get_output()), 3.885594196617603e-05, decimal=5)

    @unittest.skipIf(astra_not_available, "ccpi-astra not available")
    def test_SPDHG_vs_PDHG_explicit(self):
        from ccpi.astra.operators import AstraProjectorSimple
        from ccpi.framework import BlockDataContainer, AcquisitionData, AcquisitionGeometry, ImageData, ImageGeometry
        from ccpi.optimisation.operators import BlockOperator, Gradient
        from ccpi.optimisation.functions import BlockFunction, KullbackLeibler, MixedL21Norm, IndicatorBox
        from ccpi.optimisation.algorithms import SPDHG, PDHG
        loader = TestData()
        data = loader.load(TestData.SIMPLE_PHANTOM_2D, size=(128,128))
        print ("here")
        ig = data.geometry
        ig.voxel_size_x = 0.1
        ig.voxel_size_y = 0.1
            
        detectors = ig.shape[0]
        angles = np.linspace(0, np.pi, 180)
        ag = AcquisitionGeometry('parallel','2D',angles, detectors, pixel_size_h = 0.1)
        # Select device
        # device = input('Available device: GPU==1 / CPU==0 ')
        # if device=='1':
        #     dev = 'gpu'
        # else:
        #     dev = 'cpu'
        dev = 'gpu'

        Aop = AstraProjectorSimple(ig, ag, dev)
        
        sin = Aop.direct(data)
        # Create noisy data. Apply Gaussian noise
        noises = ['gaussian', 'poisson']
        noise = noises[1]
        if noise == 'poisson':
            np.random.seed(10)
            scale = 5
            eta = 0
            noisy_data = AcquisitionData(np.random.poisson( scale * (eta + sin.as_array()))/scale, ag)
        elif noise == 'gaussian':
            np.random.seed(10)
            n1 = np.random.normal(0, 0.1, size = ag.shape)
            noisy_data = AcquisitionData(n1 + sin.as_array(), ag)
            
        else:
            raise ValueError('Unsupported Noise ', noise)
        
        #%% 'explicit' SPDHG, scalar step-sizes
        subsets = 10
        size_of_subsets = int(len(angles)/subsets)
        # create Gradient operator
        op1 = Gradient(ig)
        # take angles and create uniform subsets in uniform+sequential setting
        list_angles = [angles[i:i+size_of_subsets] for i in range(0, len(angles), size_of_subsets)]
        # create acquisitioin geometries for each the interval of splitting angles
        list_geoms = [AcquisitionGeometry('parallel','2D',list_angles[i], detectors, pixel_size_h = 0.1) 
        for i in range(len(list_angles))]
        # create with operators as many as the subsets
        A = BlockOperator(*[AstraProjectorSimple(ig, list_geoms[i], dev) for i in range(subsets)] + [op1])
        ## number of subsets
        #(sub2ind, ind2sub) = divide_1Darray_equally(range(len(A)), subsets)
        #
        ## acquisisiton data
        g = BlockDataContainer(*[AcquisitionData(noisy_data.as_array()[i:i+size_of_subsets,:]) for i in range(0, len(angles), size_of_subsets)])
        alpha = 0.5
        ## block function
        F = BlockFunction(*[*[KullbackLeibler(b=g[i]) for i in range(subsets)] + [alpha * MixedL21Norm()]]) 
        G = IndicatorBox(lower=0)
        print ("here")
        prob = [1/(2*subsets)]*(len(A)-1) + [1/2]
        spdhg = SPDHG(f=F,g=G,operator=A, 
                    max_iteration = 1000,
                    update_objective_interval=200, prob = prob)
        spdhg.run(1000, very_verbose = True)


        #%% with different probability choice
        #prob = [1/len(A)]*(len(A))
        #spdhg = SPDHG(f=F,g=G,operator=A, 
        #              max_iteration = 1000,
        #              update_objective_interval=200, prob = prob)
        #spdhg.run(1000, very_verbose = True)
        #plt.figure()
        #plt.imshow(spdhg.get_output().as_array())
        #plt.colorbar()
        #plt.show()
        #%% 'explicit' PDHG, scalar step-sizes
        op1 = Gradient(ig)
        op2 = Aop
        # Create BlockOperator
        operator = BlockOperator(op1, op2, shape=(2,1) ) 
        f2 = KullbackLeibler(b=noisy_data)  
        g =  IndicatorBox(lower=0)    
        normK = operator.norm()
        sigma = 1/normK
        tau = 1/normK
            
        f1 = alpha * MixedL21Norm() 
        f = BlockFunction(f1, f2)   
        # Setup and run the PDHG algorithm
        pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma)
        pdhg.max_iteration = 1000
        pdhg.update_objective_interval = 200
        pdhg.run(1000, very_verbose = True)

        #%% show diff between PDHG and SPDHG
        # plt.imshow(spdhg.get_output().as_array() -pdhg.get_output().as_array())
        # plt.colorbar()
        # plt.show()

        from ccpi.utilities.quality_measures import mae, mse, psnr
        qm = (mae(spdhg.get_output(), pdhg.get_output()),
            mse(spdhg.get_output(), pdhg.get_output()),
            psnr(spdhg.get_output(), pdhg.get_output())
            )
        print ("Quality measures", qm)
        np.testing.assert_almost_equal( mae(spdhg.get_output(), pdhg.get_output()), 0.0015075773699209094 , decimal=5)
        np.testing.assert_almost_equal( mse(spdhg.get_output(), pdhg.get_output()), 1.6859006791491993e-05, decimal=5)
    
    @unittest.skipIf(astra_not_available, "ccpi-astra not available")
    def test_SPDHG_vs_SPDHG_explicit_axpby(self):
        from ccpi.astra.operators import AstraProjectorSimple
        from ccpi.framework import BlockDataContainer, AcquisitionData, AcquisitionGeometry, ImageData, ImageGeometry
        from ccpi.optimisation.operators import BlockOperator, Gradient
        from ccpi.optimisation.functions import BlockFunction, KullbackLeibler, MixedL21Norm, IndicatorBox
        from ccpi.optimisation.algorithms import SPDHG, PDHG
        loader = TestData()
        data = loader.load(TestData.SIMPLE_PHANTOM_2D, size=(128,128))
        print ("here")
        ig = data.geometry
        ig.voxel_size_x = 0.1
        ig.voxel_size_y = 0.1
            
        detectors = ig.shape[0]
        angles = np.linspace(0, np.pi, 180)
        ag = AcquisitionGeometry('parallel','2D',angles, detectors, pixel_size_h = 0.1)
        # Select device
        # device = input('Available device: GPU==1 / CPU==0 ')
        # if device=='1':
        #     dev = 'gpu'
        # else:
        #     dev = 'cpu'
        dev = 'gpu'

        Aop = AstraProjectorSimple(ig, ag, dev)
        
        sin = Aop.direct(data)
        # Create noisy data. Apply Gaussian noise
        noises = ['gaussian', 'poisson']
        noise = noises[1]
        if noise == 'poisson':
            np.random.seed(10)
            scale = 5
            eta = 0
            noisy_data = AcquisitionData(np.random.poisson( scale * (eta + sin.as_array()))/scale, ag)
        elif noise == 'gaussian':
            np.random.seed(10)
            n1 = np.random.normal(0, 0.1, size = ag.shape)
            noisy_data = AcquisitionData(n1 + sin.as_array(), ag)
            
        else:
            raise ValueError('Unsupported Noise ', noise)
        
        #%% 'explicit' SPDHG, scalar step-sizes
        subsets = 10
        size_of_subsets = int(len(angles)/subsets)
        # create Gradient operator
        op1 = Gradient(ig)
        # take angles and create uniform subsets in uniform+sequential setting
        list_angles = [angles[i:i+size_of_subsets] for i in range(0, len(angles), size_of_subsets)]
        # create acquisitioin geometries for each the interval of splitting angles
        list_geoms = [AcquisitionGeometry('parallel','2D',list_angles[i], detectors, pixel_size_h = 0.1) 
        for i in range(len(list_angles))]
        # create with operators as many as the subsets
        A = BlockOperator(*[AstraProjectorSimple(ig, list_geoms[i], dev) for i in range(subsets)] + [op1])
        ## number of subsets
        #(sub2ind, ind2sub) = divide_1Darray_equally(range(len(A)), subsets)
        #
        ## acquisisiton data
        g = BlockDataContainer(*[AcquisitionData(noisy_data.as_array()[i:i+size_of_subsets,:]) for i in range(0, len(angles), size_of_subsets)])
        alpha = 0.5
        ## block function
        F = BlockFunction(*[*[KullbackLeibler(b=g[i]) for i in range(subsets)] + [alpha * MixedL21Norm()]]) 
        G = IndicatorBox(lower=0)
        print ("here")
        prob = [1/(2*subsets)]*(len(A)-1) + [1/2]
        algos = []
        algos.append( SPDHG(f=F,g=G,operator=A, 
                    max_iteration = 1000,
                    update_objective_interval=200, prob = prob, use_axpby=True)
        )
        algos[0].run(1000, very_verbose = True)

        algos.append( SPDHG(f=F,g=G,operator=A, 
                    max_iteration = 1000,
                    update_objective_interval=200, prob = prob, use_axpby=False)
        )
        algos[1].run(1000, very_verbose = True)
        

        # np.testing.assert_array_almost_equal(algos[0].get_output().as_array(), algos[1].get_output().as_array())
        from ccpi.utilities.quality_measures import mae, mse, psnr
        qm = (mae(algos[0].get_output(), algos[1].get_output()),
            mse(algos[0].get_output(), algos[1].get_output()),
            psnr(algos[0].get_output(), algos[1].get_output())
            )
        print ("Quality measures", qm)
        np.testing.assert_array_less( qm[0], 0.005 )
        np.testing.assert_array_less( qm[1], 3.e-05)
    
    @unittest.skipIf(astra_not_available, "ccpi-astra not available")
    def test_PDHG_vs_PDHG_explicit_axpby(self):
        from ccpi.astra.operators import AstraProjectorSimple
        from ccpi.framework import BlockDataContainer, AcquisitionData, AcquisitionGeometry, ImageData, ImageGeometry
        from ccpi.optimisation.operators import BlockOperator, Gradient
        from ccpi.optimisation.functions import BlockFunction, KullbackLeibler, MixedL21Norm, IndicatorBox
        from ccpi.optimisation.algorithms import PDHG
        loader = TestData()
        data = loader.load(TestData.SIMPLE_PHANTOM_2D, size=(128,128))
        print ("here")
        ig = data.geometry
        ig.voxel_size_x = 0.1
        ig.voxel_size_y = 0.1
            
        detectors = ig.shape[0]
        angles = np.linspace(0, np.pi, 180)
        ag = AcquisitionGeometry('parallel','2D',angles, detectors, pixel_size_h = 0.1)
        
        dev = 'gpu'

        Aop = AstraProjectorSimple(ig, ag, dev)
        
        sin = Aop.direct(data)
        # Create noisy data. Apply Gaussian noise
        noises = ['gaussian', 'poisson']
        noise = noises[1]
        if noise == 'poisson':
            np.random.seed(10)
            scale = 5
            eta = 0
            noisy_data = AcquisitionData(np.random.poisson( scale * (eta + sin.as_array()))/scale, ag)
        elif noise == 'gaussian':
            np.random.seed(10)
            n1 = np.random.normal(0, 0.1, size = ag.shape)
            noisy_data = AcquisitionData(n1 + sin.as_array(), ag)
            
        else:
            raise ValueError('Unsupported Noise ', noise)
        alpha = 0.5
        op1 = Gradient(ig)
        op2 = Aop
        # Create BlockOperator
        operator = BlockOperator(op1, op2, shape=(2,1) ) 
        f2 = KullbackLeibler(b=noisy_data)  
        g =  IndicatorBox(lower=0)    
        normK = operator.norm()
        sigma = 1/normK
        tau = 1/normK
            
        f1 = alpha * MixedL21Norm() 
        f = BlockFunction(f1, f2)   
        # Setup and run the PDHG algorithm
        
        algos = []
        algos.append( PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma,  
                    max_iteration = 1000,
                    update_objective_interval=200, use_axpby=True)
        )
        algos[0].run(1000, very_verbose = True)

        algos.append( PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma,  
                    max_iteration = 1000,
                    update_objective_interval=200, use_axpby=False)
        )
        algos[1].run(1000, very_verbose = True)
        

        from ccpi.utilities.quality_measures import mae, mse, psnr
        qm = (mae(algos[0].get_output(), algos[1].get_output()),
            mse(algos[0].get_output(), algos[1].get_output()),
            psnr(algos[0].get_output(), algos[1].get_output())
            )
        print ("Quality measures", qm)
        np.testing.assert_array_less( qm[0], 0.005 )
        np.testing.assert_array_less( qm[1], 3.e-05)
        


if __name__ == '__main__':
    
    d = TestAlgorithms()
    d.test_GradientDescentArmijo2()
 
