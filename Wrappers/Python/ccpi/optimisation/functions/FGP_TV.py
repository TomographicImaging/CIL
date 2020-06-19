# -*- coding: utf-8 -*-
#  CCP in Tomographic Imaging (CCPi) Core Imaging Library (CIL).

#   Copyright 2017-2020 UKRI-STFC
#   Copyright 2017-2020 University of Manchester

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ccpi.optimisation.functions import Function, MixedL21Norm, IndicatorBox
from ccpi.optimisation.operators import Gradient
import numpy 
import functools



class FGP_TV(Function):
    
    r'''Fast Gradient Projection algorithm for Total Variation(TV) Denoising (ROF problem)
    
    .. math::  \min_{x} \alpha TV(x) + \frac{1}{2}||x-b||^{2}_{2}
                
    Parameters:
      
      :param domain: Domain of the reconstruction
      :param regularising_parameter: TV regularising parameter 
      :param inner_iterations: Iterations of FGP algorithm
      :param tolerance: Stopping criterion (Default=1e-4)
      :param lower: ( Default = - numpy.inf ) lower bound for the orthogonal projection onto the convex set C
      :param upper: ( Default = + numpy.inf ) upper bound for the orthogonal projection onto the convex set C
      
    Reference:
      
        A. Beck and M. Teboulle, "Fast Gradient-Based Algorithms for Constrained Total Variation 
        Image Denoising and Deblurring Problems," in IEEE Transactions on Image Processing,
        vol. 18, no. 11, pp. 2419-2434, Nov. 2009, 
        doi: 10.1109/TIP.2009.2028250.
        
    '''    
    
    
    def __init__(self, regularising_parameter, 
                 inner_iterations, 
                 tolerance, 
                 lower = -numpy.inf, 
                 upper = numpy.inf,
                 info = False):
        

        super(FGP_TV, self).__init__(L = None)
        # Regularising parameter = alpha
        self.regularising_parameter = regularising_parameter
        
        # Iterations for FGP_TV
        self.inner_iterations = inner_iterations
        
        # Tolerance for FGP_TV
        self.tolerance = tolerance
        
        # Define (ISOTROPIC) Total variation penalty ( Note it is without the regularising paremeter)
        # TODO add anisotropic???
        self.TV = MixedL21Norm()  
        
        # Define orthogonal projection onto the convex set C
        self.lower = lower
        self.upper = upper
        self.tmp_proj_C = IndicatorBox(lower, upper).proximal
                        
#         Setup Gradient as None. This is to avoid domain argument in the __init__     
#         self.gradient = None

        # self.gradient = Gradient(domain)
        # self.Lipschitz = (1./self.gradient.norm())**2   
        self._gradient = None
        self._domain = None
        
        # Print stopping information (iterations and tolerance error) of FGP_TV  
        self.info = info
                    
    def __call__(self, x):
        
        r''' Returns the value of the \alpha * TV(x)'''
        self._domain = x.geometry
#         if self.gradient is None:
#             self.gradient = Gradient(x.geometry)
        
        # evaluate objective function of TV gradient
        return regularising_parameter * self.TV(self.gradient.direct(x))
    
    
    def projection_C(self, x, out=None):   
                     
        r''' Returns orthogonal projection onto the convex set C'''

        self._domain = x.geometry
        return self.tmp_proj_C(x, tau = None, out = out)
                        
    def projection_P(self, x, out=None):
                       
        r''' Returns the projection P onto \|\cdot\|_{\infty} '''  
        self._domain = x.geometry
        
        res1 = functools.reduce(lambda a,b: a + b*b, x.containers, x.get_item(0) * 0 )
        res1.sqrt(out=res1)	
        res1.maximum(1.0, out=res1)	    

        if out is None:
            return x.divide(res1)
        else:
            x.divide(res1, out=out)
    
    
    def proximal(self, x, tau, out = None):
        
        ''' Returns the solution of the FGP_TV algorithm '''         
                        
#         if self.gradient is None:
            
#             # Define 1/Lipschitz of the Gradient
#             self.gradient = Gradient(x.geometry)            
#             self.Lipschitz = (1./self.gradient.norm())**2            
        self._domain = x.geometry
        
        # initialise
        t = 1        
        tmp_p = self.gradient.range_geometry().allocate()  
        tmp_q = tmp_p.copy()
        tmp_x = self.gradient.domain_geometry().allocate()     
        p1 = tmp_p.copy()

        for k in range(self.inner_iterations):
                                                                                   
            t0 = t
            self.gradient.adjoint(tmp_q, out = tmp_x)
            
            # this can be replaced by axpby
            tmp_x *= -self.regularising_parameter
            tmp_x *= tau
            tmp_x += x
            self.projection_C(tmp_x, out = tmp_x)                       

            self.gradient.direct(tmp_x, out=p1)
            p1 *= self.L
            p1 /= self.regularising_parameter
            p1 /= tau
            p1 += tmp_q
            self.projection_P(p1, out=p1)
            
            if self.tolerance is not None:
                
                if k%5==0:
                    error = (p1-tmp_q).norm()/p1.norm()            
                    if error<=self.tolerance:                           
                        break

            t = (1 + numpy.sqrt(1 + 4 * t0 ** 2)) / 2                                                                                                                                             
            tmp_q.fill(p1 + (t0 - 1) / t * (p1 - tmp_p))                                      
            tmp_p.fill(p1)             
        
        # Print stopping information (iterations and tolerance error) of FGP_TV     
        if self.info:
            if self.tolerance is not None:
                print("Stop at {} iterations with error {} .".format(k, error))
            else:
                print("Stop at {} iterations.".format(k))                
            
        if out is None:                        
            return self.projection_C(x - self.regularising_parameter*tau*self.gradient.adjoint(tmp_q))
        else:          
            self.gradient.adjoint(tmp_q, out = out)
            out*=tau
            out*=-self.regularising_parameter
            out+=x
            self.projection_C(out, out=out)
    
    def convex_conjugate(self,x):        
        return 0.0    
    @property
    def L(self):
        if self._L is None:
            self.calculate_Lipschitz()
        return self._L
    @L.setter
    def L(self, value):
        warnings.warn("You should set the Lipschitz constant with calculate_Lipschitz().")
        if isinstance(value, (Number,)) and value >= 0:
            self._L = value
        else:
            raise TypeError('The Lipschitz constant is a real positive number')

    def calculate_Lipschitz(self):
        # Compute the Lipschitz parameter from the operator if possible
        # Leave it initialised to None otherwise
        self._L = (1./self.gradient.norm())**2  
    
    @property
    def gradient(self):
        '''creates a gradient operator if not instantiated yet

        There is no check that the variable _domain is changed after instantiation (should not be the case)'''
        if self._gradient is None:
            if self._domain is not None:
                self._gradient = Gradient(self._domain)
        return self._gradient
    

if __name__ == '__main__':
    
    import numpy as np                         
    import matplotlib.pyplot as plt
    from ccpi.framework import TestData
    import os
    import sys
    from ccpi.plugins.regularisers import FGP_TV as CCPiReg_FGP_TV
    from ccpi.filters import regularisers    
    from timeit import default_timer as timer       
    import tomophantom
    from tomophantom import TomoP3D
    from ccpi.framework import ImageGeometry
    from ccpi.utilities.quality_measures import mae
    from ccpi.utilities.display import show
    
        
    ###################################################################
    ###################################################################
    ###################################################################
    ###################################################################
    print("Compare CIL_FGP_TV vs CCPiReg_FGP_TV no tolerance (2D)")
    
    loader = TestData(data_dir=os.path.join(sys.prefix, 'share','ccpi'))
    data = loader.load(TestData.SHAPES)
    ig = data.geometry
    ag = ig

    # Create noisy data. 
    n1 = np.random.normal(0, 0.1, size = ig.shape)
    noisy_data = ig.allocate()
    noisy_data.fill(n1+data.as_array())

    # Show Ground Truth and Noisy Data
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(data.as_array())
    plt.title('Ground Truth')
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.imshow(noisy_data.as_array())
    plt.title('Noisy Data')
    plt.colorbar()
    plt.show()
    
    alpha = 0.1
    iters = 1000
        
    # CIL_FGP_TV no tolerance
    g_CIL = FGP_TV(alpha, iters, tolerance=None, lower = 0, info = True)
    t0 = timer()
    res1 = g_CIL.proximal(noisy_data, 1.)
    t1 = timer()
    print(t1-t0)
    
    plt.figure()
    plt.imshow(res1.as_array())
    plt.colorbar()
    plt.show()
    
    # CCPi Regularisation toolkit high tolerance
    r_alpha = alpha
    r_iterations = iters
    r_tolerance = 1e-9
    r_iso = 0
    r_nonneg = 1
    r_printing = 0
    g_CCPI_reg_toolkit = CCPiReg_FGP_TV(r_alpha, r_iterations, r_tolerance, r_iso, r_nonneg, r_printing, 'cpu')
    
    t2 = timer()
    res2 = g_CCPI_reg_toolkit.proximal(noisy_data, 1.)
    t3 = timer()
    print(t3-t1)
    
    plt.figure()
    plt.imshow(res2.as_array())
    plt.colorbar()
    plt.show()
    
    plt.figure()
    plt.imshow(np.abs(res1.as_array()-res2.as_array()))
    plt.colorbar()
    plt.title("Difference CIL_FGP_TV vs CCPi_FGP_TV")
    plt.show()

    np.testing.assert_array_almost_equal(res1.as_array(), res2.as_array(), decimal = 4)
    
    ###################################################################
    ###################################################################
    ###################################################################
    ###################################################################    
    
    print("Compare CIL_FGP_TV vs CCPiReg_FGP_TV with iterations.")
    iters = 408
    # CIL_FGP_TV no tolerance
    g_CIL = FGP_TV(alpha, iters, tolerance=1e-9, lower = 0.)
    t0 = timer()
    res1 = g_CIL.proximal(noisy_data, 1.)
    t1 = timer()
    print(t1-t0)
    
    # CCPi Regularisation toolkit high tolerance
    r_alpha = alpha
    r_iterations = iters
    r_tolerance = 1e-9
    r_iso = 0
    r_nonneg = 1
    r_printing = 0
    g_CCPI_reg_toolkit = CCPiReg_FGP_TV(r_alpha, r_iterations, r_tolerance, r_iso, r_nonneg, r_printing, 'cpu')

    t2 = timer()
    res2 = g_CCPI_reg_toolkit.proximal(noisy_data, 1.)
    t3 = timer()
    print(t3-t2)
    
    plt.figure()
    plt.imshow(np.abs(res1.as_array()-res2.as_array()))
    plt.colorbar()
    plt.title("Difference CIL_FGP_TV vs CCPi_FGP_TV")
    plt.show()          
    
    print(mae(res1, res2))
    np.testing.assert_array_almost_equal(res1.as_array(), res2.as_array(), decimal=3)    
    
    ###################################################################
    ###################################################################
    ###################################################################
    ###################################################################
    print("Compare CIL_FGP_TV vs CCPiReg_FGP_TV no tolerance (3D)") 
        
    print ("Building 3D phantom using TomoPhantom software")
    model = 13 # select a model number from the library
    N_size = 64 # Define phantom dimensions using a scalar value (cubic phantom)
    path = os.path.dirname(tomophantom.__file__)
    path_library3D = os.path.join(path, "Phantom3DLibrary.dat")
    #This will generate a N_size x N_size x N_size phantom (3D)
    phantom_tm = TomoP3D.Model(model, N_size, path_library3D)    
    
    ig = ImageGeometry(N_size, N_size, N_size)
    data = ig.allocate()
    data.fill(phantom_tm)
        
    n1 = TestData.random_noise(data.as_array(), mode = 'gaussian', seed = 10)

    noisy_data = ig.allocate()
    noisy_data.fill(n1)    
    
    # Show Ground Truth and Noisy Data
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(data.as_array()[32])
    plt.title('Ground Truth')
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.imshow(noisy_data.as_array()[32])
    plt.title('Noisy Data')
    plt.colorbar()
    plt.show()
    
    alpha = 0.1
    iters = 1000
    
    print("Use tau as an array of ones")
    # CIL_FGP_TV no tolerance
    g_CIL = FGP_TV(alpha, iters, tolerance=None, info=True)
    t0 = timer()   
    res1 = g_CIL.proximal(noisy_data, ig.allocate(1.))
    t1 = timer()
    print(t1-t0)

    show(res1, cmap='viridis')
    
    # CCPi Regularisation toolkit high tolerance
    r_alpha = alpha
    r_iterations = iters
    r_tolerance = 1e-9
    r_iso = 0
    r_nonneg = 0
    r_printing = 0
    g_CCPI_reg_toolkit = CCPiReg_FGP_TV(r_alpha, r_iterations, r_tolerance, r_iso, r_nonneg, r_printing, 'cpu')

    t2 = timer()
    res2 = g_CCPI_reg_toolkit.proximal(noisy_data, 1.)
    t3 = timer()
    print (t3-t2)
    show(res1, cmap='viridis')

    show((res1-res2).abs(), title = "Difference CIL_FGP_TV vs CCPi_FGP_TV", cmap='viridis')     
    np.testing.assert_array_almost_equal(res1.as_array(), res2.as_array(), decimal=3)

    
    
    # CIL_FGP_TV no tolerance
    #g_CIL = FGP_TV(ig, alpha, iters, tolerance=None, info=True)
    g_CIL.tolerance = None
    t0 = timer()
    res1 = g_CIL.proximal(noisy_data, 1.)
    t1 = timer()
    print(t1-t0)

    show(res1, cmap='viridis')    
    
    
    ###################################################################
    ###################################################################
    ###################################################################
    ###################################################################     
#     print("Compare CIL_FGP_TV vs CCPiReg_FGP_TV with tolerance. There is a problem with this line https://github.com/vais-ral/CCPi-Regularisation-Toolkit/blob/master/src/Core/regularisers_GPU/TV_FGP_GPU_core.cu#L456")
    
#     g_CIL = FGP_TV(ig, alpha, iters, tolerance=1e-3, lower = 0)
#     res1 = g_CIL.proximal(noisy_data, 1.)

#     plt.imshow(res1.as_array())
#     plt.colorbar()
#     plt.show()
    
#     r_alpha = alpha
#     r_iterations = iters
#     r_tolerance = 1e-3
#     r_iso = 0
#     r_nonneg = 1
#     r_printing = 0
#     g_CCPI_reg_toolkit = CCPiReg_FGP_TV(r_alpha, r_iterations, r_tolerance, r_iso, r_nonneg, r_printing, 'gpu')

#     res2 = g_CCPI_reg_toolkit.proximal(noisy_data, 1.)
#     plt.imshow(res2.as_array())
#     plt.colorbar()
#     plt.show()

#     plt.imshow(np.abs(res1.as_array()-res2.as_array()))
#     plt.colorbar()
#     plt.title("Difference CIL_FGP_TV vs CCPi_FGP_TV")
#     plt.show() 
    
#     if mae(res1, res2)>1e-5:
#         raise ValueError ("2 solutions are not the same")      
            
       
    loader = TestData(data_dir=os.path.join(sys.prefix, 'share','ccpi'))
    data = loader.load(TestData.PEPPERS, size=(256,256))
    ig = data.geometry
    ag = ig

    n1 = TestData.random_noise(data.as_array(), mode = 'gaussian', seed = 10)

    noisy_data = ig.allocate()
    noisy_data.fill(n1)    
    
    # Show Ground Truth and Noisy Data
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(data.as_array())
    plt.title('Ground Truth')
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.imshow(noisy_data.as_array())
    plt.title('Noisy Data')
    plt.colorbar()
    plt.show()
    
    alpha = 0.1
    iters = 1000
    
    # CIL_FGP_TV no tolerance
    g_CIL = FGP_TV(alpha, iters, tolerance=None)
    t0 = timer()
    res1 = g_CIL.proximal(noisy_data, 1.)
    t1 = timer()
    print(t1-t0)

    plt.figure()
    plt.imshow(res1.as_array())
    plt.colorbar()
    plt.show()
    
    # CCPi Regularisation toolkit high tolerance
    r_alpha = alpha
    r_iterations = iters
    r_tolerance = 1e-8
    r_iso = 0
    r_nonneg = 0
    r_printing = 0
    g_CCPI_reg_toolkit = CCPiReg_FGP_TV(r_alpha, r_iterations, r_tolerance, r_iso, r_nonneg, r_printing, 'cpu')
    
    t2 = timer()
    res2 = g_CCPI_reg_toolkit.proximal(noisy_data, 1.)
    t3 = timer()
    print (t3-t2)
    plt.figure()
    plt.imshow(res2.as_array())
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.imshow(np.abs(res1.as_array()-res2.as_array()))
    plt.colorbar()
    plt.title("Difference CIL_FGP_TV vs CCPi_FGP_TV")
    plt.show()    
    
    
    
    
    
