#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 23:24:00 2020

@author: evangelos
"""


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
        Linearized Alternating Direction Method of Multipliers (ADMM)
    
        General form of ADMM : min_{x} f(x) + g(y), subject to Ax + By = b
    
        Case: A = Id, B = -K, b = 0   ==> min_x f(Kx) + g(x)  
                
        The quadratic term in the augmented Lagrangian is linearized for the x-update.
                            
        Main algorithmic difference is that in ADMM we compute two proximal subproblems, 
        where in the PDHG a proximal and proximal conjugate.

        Reference (Section 8) : https://link.springer.com/content/pdf/10.1007/s10107-018-1321-1.pdf                          

            x^{k} = prox_{\tau f } (x^{k-1} - tau/sigma A^{T}(Ax^{k-1} - z^{k-1} + u^{k-1} )                
            
            z^{k} = prox_{\sigma g} (Ax^{k} + u^{k-1})
            
            u^{k} = u^{k-1} + Ax^{k} - z^{k}
                
    '''

    def __init__(self, f, g, operator, \
                       tau = None, sigma = 1., 
                       x_init = None):
        
        '''Initialisation of the algorithm

        :param operator: a Linear Operator
        :param f: Convex function with "simple" proximal
        :param g: Convex function with "simple" proximal 
        :param sigma: Positive step size parameter 
        :param tau: Positive step size parameter
        :param x_init: Initial guess ( Default x_init = 0)'''        
        
        super(ADMM_linearized, self).__init__()
        
        self.set_up(f = f, g = g, operator = operator, tau = tau, sigma = sigma, x_init = x_init)        
                    
    def set_up(self, f, g, operator, tau = None, sigma = 1., x_init = None):

        print("{} setting up".format(self.__class__.__name__, ))
        
        if sigma is None and tau is None:
            raise ValueError('Need tau <= sigma / ||K||^2')

        self.f = f
        self.g = g
        self.operator = operator

        self.tau = tau
        self.sigma = sigma

        if self.tau is None:
            normK = self.operator.norm()
            self.tau = self.sigma / normK ** 2
            
        if x_init is None:
            self.x = self.operator.domain_geometry().allocate()
        else:
            self.x = x_init.copy()
         
        # allocate space for operator direct & adjoint    
        self.tmp_dir = self.operator.range_geometry().allocate()
        self.tmp_adj = self.operator.domain_geometry().allocate()            
            
        self.z = self.operator.range_geometry().allocate() 
        self.u = self.operator.range_geometry().allocate() 

        self.update_objective()
        self.configured = True  
        
        print("{} configured".format(self.__class__.__name__, ))
        
    def update(self):

        self.tmp_dir += self.u
        self.tmp_dir -= self.z          
        self.operator.adjoint(self.tmp_dir, out = self.tmp_adj)          
        
        # TODO replace with axpby (doesn'twork)
        self.tmp_adj *= -(self.tau/self.sigma)
        self.x += self.tmp_adj     
        # apply proximal of f        
        self.f.proximal(self.x, self.tau, out = self.x)
        
        self.operator.direct(self.x, out = self.tmp_dir)  
        self.u += self.tmp_dir
        
        # apply proximal of g   
        self.g.proximal(self.u, self.sigma, out = self.z)

        # update 
        self.u += self.tmp_dir
        self.u -= self.z

    def update_objective(self):
        
        self.loss.append(self.f(self.x) +  self.g(self.operator.direct(self.x)) ) 
        
if __name__ == '__main__':
    
        import numpy as np 
        import numpy                          
        import matplotlib.pyplot as plt
        
        from ccpi.optimisation.algorithms import PDHG
        
        from ccpi.optimisation.operators import BlockOperator, Identity, Gradient
        from ccpi.optimisation.functions import ZeroFunction, L1Norm, SmoothMixedL21Norm, \
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
        elif noise == 'poisson':
            f2 = KullbackLeibler(b=noisy_data, eta=eta)
        elif noise == 'gaussian':
            f2 = 0.5*L2NormSquared(b=noisy_data)
            
        #setup and run ADMM            
            
        operator = Gradient(ig)       
        g = alpha * MixedL21Norm()
                        
        # Setup and run the PDHG algorithm
        admm = ADMM_linearized(f = f2, g = g, operator = operator)
        admm.max_iteration = 3000
        admm.update_objective_interval = 500
        admm.run(3000, verbose=True)   
               
        # Setup and run the PDHG algorithm
        # Note here we use the proximal conjugate of the KL divergence
        
        # Create operators
        op1 = Gradient(ig)
        op2 = Identity(ig, ag)
    
        # Create BlockOperator
        operator = BlockOperator(op1, op2, shape=(2,1) ) 
    
        # Create functions      
        f = BlockFunction(g, f2) 
        h = ZeroFunction()
            
        
        sigma = 1
        tau = 1/(sigma * operator.norm()**2 )
        
        # Setup and run the PDHG algorithm
        pdhg = PDHG(f = f, g = h, operator=operator)
        pdhg.max_iteration = 3000
        pdhg.update_objective_interval = 500
        pdhg.run(3000, very_verbose = True)    
        
        # Show results
        plt.figure(figsize=(8,10))
        plt.subplot(2,2,1)
        plt.imshow(data.as_array())
        plt.title('Ground Truth')
        plt.colorbar()
        plt.subplot(2,2,2)
        plt.imshow(noisy_data.as_array())
        plt.title('Noisy Data')
        plt.colorbar()
        plt.subplot(2,2,3)
        plt.imshow(admm.get_output().as_array())
        plt.title('ADMM: TV Reconstruction')
        plt.colorbar()
        plt.subplot(2,2,4)
        plt.imshow(pdhg.get_output().as_array())
        plt.title('PDHG: TV Reconstruction')
        plt.colorbar() 
        plt.show()
                
        plt.figure(figsize=(10,5))            
        plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), data.as_array()[int(ig.shape[0]/2),:], label = 'GTruth')
        plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), admm.get_output().as_array()[int(ig.shape[0]/2),:], label = 'ADMM reconstruction')
        plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), pdhg.get_output().as_array()[int(ig.shape[0]/2),:], label = 'PDHG reconstruction')
                
        plt.legend()
        plt.title('Middle Line Profiles')
        plt.show()
    
    
    
    
            
        
        