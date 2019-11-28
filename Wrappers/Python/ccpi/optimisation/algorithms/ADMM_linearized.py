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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from ccpi.optimisation.algorithms import Algorithm

class ADMM_linearized(Algorithm):
        
    ''' 
        Quick comments:
    
        ADMM :   min_{x} f(x) + g(y), subject to Ax + By = b
    
        Case: A = Id, B = -K, b = 0   ==> min_x f(Kx) + g(x)  
                
        (Linearized) ADMM : The quadratic term in the augmented Lagrangian is linearized
                            for the x-update
                            
        Main algorithmic diff is that in ADMM computes two proximal, where in the PDHG
        one proxi, one prox conjugate                            
    
    
    '''

    def __init__(self, **kwargs):
        
        super(ADMM_linearized, self).__init__(max_iteration=kwargs.get('max_iteration',0))
        self.f        = kwargs.get('f', None)
        self.operator = kwargs.get('operator', None)
        self.g        = kwargs.get('g', None)
        self.tau      = kwargs.get('tau', None)
        self.sigma    = kwargs.get('sigma', 1.)    
        
        if self.f is not None and self.operator is not None and \
           self.g is not None:
            if self.tau is None:
                # Compute operator Norm
                normK = self.operator.norm()
                # Primal & dual stepsizes
                self.tau = 1/(self.sigma*normK**2)
            print ("Calling from creator")
            self.set_up(self.f,
                        self.g,
                        self.operator,
                        self.tau, 
                        self.sigma)        
            
    def set_up(self, f, g, operator, tau = None, sigma = None, opt = None, **kwargs):
        
        # algorithmic parameters
        self.operator = operator
        self.f = f
        self.g = g
        self.tau = tau
        self.sigma = sigma
        self.opt = opt

        self.x = self.operator.domain_geometry().allocate()
        self.z0 = self.operator.range_geometry().allocate()
        self.u0 = self.z0.copy()
        
        self.x1 = self.x.copy() 
        self.z1 = self.z0.copy() 
        self.u1 = self.u0.copy() 
        
        self.tmp1 = self.z0.copy()
        self.tmp2 = self.x.copy()
        self.tmp3 = self.z0.copy()
    
        self.update_objective()
        self.configured = True        
        
    def update(self):
        
        self.operator.direct(self.x, out = self.tmp1)
        self.tmp1 += self.u0
        self.tmp1 += -1 * self.z0
        
        self.operator.adjoint(self.tmp1, out = self.tmp2)
        
        self.tmp2 *= -1 * (self.tau/self.sigma)
        self.tmp2 += self.x
        self.f.proximal( self.tmp2, self.tau, out = self.x1)
        
        self.operator.direct(self.x1, out = self.tmp3)
        self.tmp3 += self.u0
        
        self.g.proximal(self.tmp3, self.sigma, out = self.z1)

        self.operator.direct(self.x1, out = self.u1)
        self.u1 += self.u0
        self.u1 -= self.z1

        self.u0.fill(self.u1)
        self.z0.fill(self.z1)
        self.x.fill(self.x1)      
                                
    def update_objective(self):

        self.loss.append(self.f(self.x) + self.g(self.operator.direct(self.x)))
        
        
if __name__ == '__main__':
    
        import numpy as np 
        import numpy                          
        import matplotlib.pyplot as plt
        
        from ccpi.optimisation.algorithms import PDHG
        
        from ccpi.optimisation.operators import BlockOperator, Identity, Gradient
        from ccpi.optimisation.functions import ZeroFunction, L1Norm, smoothMixedL21Norm, \
                              MixedL21Norm, BlockFunction, L2NormSquared,\
                                  KullbackLeibler, ConstantFunction
        from ccpi.framework import TestData
        import os
        import sys
        
        # user supplied input
        if len(sys.argv) > 1:
            which_noise = int(sys.argv[1])
        else:
            which_noise = 0
        print ("Applying {} noise")
        
        if len(sys.argv) > 2:
            method = sys.argv[2]
        else:
            method = '0'
        print ("method ", method)
        
        
        loader = TestData(data_dir=os.path.join(sys.prefix, 'share','ccpi'))
        data = loader.load(TestData.SIMPLE_PHANTOM_2D, size = (128,128))
        ig = data.geometry
        ag = ig
        
        # Create noisy data. 
        noises = ['gaussian', 'poisson', 's&p']
        noise = noises[which_noise]
        if noise == 's&p':
            n1 = TestData.random_noise(data.as_array(), mode = noise, salt_vs_pepper = 0.9, amount=0.2)
        elif noise == 'poisson':
            scale = 5
            eta = ig.allocate(0.01)
        #    eta.fill(np.random.normal(, 0.02, data.shape))
            n1 = TestData.random_noise( (data.as_array() + eta.as_array())/scale, mode = noise, seed = 10)*scale 
        elif noise == 'gaussian':
            n1 = TestData.random_noise(data.as_array(), mode = noise, seed = 10)
        else:
            raise ValueError('Unsupported Noise ', noise)
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
        
        #%%
        
        # Regularisation Parameter depending on the noise distribution
        if noise == 's&p':
            alpha = 0.8
        elif noise == 'poisson':
            alpha = 1
        elif noise == 'gaussian':
            alpha = 2
        
        # fidelity
        if noise == 's&p':
            f2 = L1Norm(b=noisy_data)
        #    f2 = L1Norm().centered_at(noisy_data)    
        elif noise == 'poisson':
            f2 = KullbackLeibler(b=noisy_data, eta=eta)
        elif noise == 'gaussian':
            f2 = 0.5*L2NormSquared(b=noisy_data)
#            f2 = 0.5 * L2NormSquared().centered_at(noisy_data)
            
        op1 = Gradient(ig, correlation=Gradient.CORRELATION_SPACE)
        op2 = Identity(ig, ag)

        # Create BlockOperator
        operator = BlockOperator(op1, op2, shape=(2,1) )             
          
        normK = operator.norm()
        sigma = 1
        tau = 1/(sigma*normK**2)    

        # Create functions      
        g = BlockFunction(alpha*MixedL21Norm(), f2) 
        f = ZeroFunction()        
                        
        # Setup and run the PDHG algorithm
        admm = ADMM_linearized(f = f, g = g,operator = operator, tau = tau, sigma = sigma)
        admm.max_iteration = 3000
        admm.update_objective_interval = 500
        admm.run(3000, very_verbose=True)            
    
    
    
    
    