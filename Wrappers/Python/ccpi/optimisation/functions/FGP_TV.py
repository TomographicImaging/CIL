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

from ccpi.optimisation.functions import Function, MixedL21Norm
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
      :param nonnegativity: 
      
    Reference:
      
        Beck, A. and Teboulle, M., 2009. Fast Gradient-Based Algorithms for Constrained
        Total Variation Image Denoising and Deblurring Problems 
        IEEE TRANSACTIONS ON IMAGE PROCESSING, VOL. 18, NO. 11, NOVEMBER 2009.
        
    '''    
    
    
    def __init__(self, domain, regularising_parameter, inner_iterations, tolerance, nonnegativity):
        
        self.regularising_parameter = regularising_parameter
        self.inner_iterations = inner_iterations
        self.tolerance = tolerance
        self.nonnegativity = nonnegativity                
        self.gradient = Gradient(domain)
        self.TV = regularising_parameter * MixedL21Norm()                 
            
    def __call__(self,x):
        
        # evaluate objective function of TV gradient
        return self.TV(self.gradient.direct(x))
    
    
    def projection_C(self, x, out=None):   

        # This can be replaced by IndicatorBox(lower=0), but is less effient.
                        
        if out is None:            
            if self.nonnegativity:
                return x.maximum(0.)
            else:
                return x
        else:
            if self.nonnegativity:            
                x.maximum(0., out=out)                
            else:
                out.fill(x)

    def projection_P(self, x, out=None):

        # This can be replaced by the proximal conjugate of MixedL21Norm, but is less effient.
        
        res1 = functools.reduce(lambda a,b: a + b*b, x.containers, x.get_item(0) * 0 )
        res1.sqrt(out=res1)	
        res1.maximum(1.0, out=res1)	    

        if out is None:
            return x.divide(res1)
        else:
            x.divide(res1, out=out)
    
    
    def proximal(self, b, tau, out = None):
                
        # initialise
        t = 1        
        tmp_p = self.gradient.range_geometry().allocate()  
        tmp_q = tmp_p.copy()
        x = self.gradient.domain_geometry().allocate()     
        p1 = tmp_p.copy()

        for k in range(self.inner_iterations):
                                                                                   
            t0 = t
            self.gradient.adjoint(tmp_q, out = x)
            
            # this can be replaced by axpby
            x *= -self.regularising_parameter
            x *= tau
            x += b
            self.projection_C(x, out = x)                       

            self.gradient.direct(x, out=p1)
            p1 *= 1./(8*self.regularising_parameter)
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
                                        
        if out is None:            
            print("Run for {} iterations".format(k))
            return self.projection_C(b - self.regularising_parameter*tau*self.gradient.adjoint(tmp_q))
        else:
            print("Run for {} iterations".format(k))            
            self.gradient.adjoint(tmp_q, out = out)
            out*=tau
            out*=-self.regularising_parameter
            out+=b
            self.projection_C(out, out=out)
    
    def convex_conjugate(self,x):        
        return 0.0    
    
if __name__ == '__main__':
    
    import numpy as np                         
    import matplotlib.pyplot as plt

#     from ccpi.optimisation.algorithms import PDHG

#     from ccpi.optimisation.operators import BlockOperator, Identity, Gradient
#     from ccpi.optimisation.functions import ZeroFunction, L1Norm, \
#                           MixedL21Norm, BlockFunction, L2NormSquared,\
#                               KullbackLeibler
    from ccpi.framework import TestData
    import os
    import sys
    from ccpi.plugins.regularisers import FGP_TV as CCPiReg_FGP_TV
    from ccpi.filters import regularisers    
    
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
    
    print("Compare CIL_FGP_TV vs CCPiReg_FGP_TV no tolerance")
    # CIL_FGP_TV no tolerance
    g_CIL = FGP_TV(ig, alpha, iters, tolerance=None, nonnegativity=True)
    res1 = g_CIL.proximal(noisy_data, 1.)

    plt.imshow(res1.as_array())
    plt.colorbar()
    plt.show()
    
    # CCPi Regularisation toolkit high tolerance
    r_alpha = alpha
    r_iterations = iters
    r_tolerance = 1e-8
    r_iso = 0
    r_nonneg = 1
    r_printing = 0
    g_CCPI_reg_toolkit = CCPiReg_FGP_TV(r_alpha, r_iterations, r_tolerance, r_iso, r_nonneg, r_printing, 'gpu')

    res2 = g_CCPI_reg_toolkit.proximal(noisy_data, 1.)
    plt.imshow(res2.as_array())
    plt.colorbar()
    plt.show()

    plt.imshow(np.abs(res1.as_array()-res2.as_array()))
    plt.colorbar()
    plt.title("Difference CIL_FGP_TV vs CCPi_FGP_TV")
    plt.show()
    
    print("Compare CIL_FGP_TV vs CCPiReg_FGP_TV with tolerance. There is a problem with this line https://github.com/vais-ral/CCPi-Regularisation-Toolkit/blob/master/src/Core/regularisers_GPU/TV_FGP_GPU_core.cu#L456")
    # CIL_FGP_TV no tolerance
    g_CIL = FGP_TV(ig, alpha, iters, tolerance=1e-3, nonnegativity=True)
    res1 = g_CIL.proximal(noisy_data, 1.)

    plt.imshow(res1.as_array())
    plt.colorbar()
    plt.show()
    
    # CCPi Regularisation toolkit high tolerance
    r_alpha = alpha
    r_iterations = iters
    r_tolerance = 1e-3
    r_iso = 0
    r_nonneg = 1
    r_printing = 0
    g_CCPI_reg_toolkit = CCPiReg_FGP_TV(r_alpha, r_iterations, r_tolerance, r_iso, r_nonneg, r_printing, 'gpu')

    res2 = g_CCPI_reg_toolkit.proximal(noisy_data, 1.)
    plt.imshow(res2.as_array())
    plt.colorbar()
    plt.show()

    plt.imshow(np.abs(res1.as_array()-res2.as_array()))
    plt.colorbar()
    plt.title("Difference CIL_FGP_TV vs CCPi_FGP_TV")
    plt.show()    
    
    print("Compare CIL_FGP_TV vs CCPiReg_FGP_TV with iterations.")
    iters = 208
    # CIL_FGP_TV no tolerance
    g_CIL = FGP_TV(ig, alpha, iters, tolerance=1e-9, nonnegativity=True)
    res1 = g_CIL.proximal(noisy_data, 1.)

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
    g_CCPI_reg_toolkit = CCPiReg_FGP_TV(r_alpha, r_iterations, r_tolerance, r_iso, r_nonneg, r_printing, 'gpu')

    res2 = g_CCPI_reg_toolkit.proximal(noisy_data, 1.)
    plt.imshow(res2.as_array())
    plt.colorbar()
    plt.show()

    plt.imshow(np.abs(res1.as_array()-res2.as_array()))
    plt.colorbar()
    plt.title("Difference CIL_FGP_TV vs CCPi_FGP_TV")
    plt.show()        
    
    
    
   
