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

from cil.optimisation.algorithms import Algorithm
import warnings
import numpy



class PDHG(Algorithm):
    r'''Primal Dual Hybrid Gradient
    
    Problem: 
    
    .. math::
    
      \min_{x} f(Kx) + g(x)
        
    :param operator: Linear Operator = K
    :param f: Convex function with "simple" proximal of its conjugate. 
    :param g: Convex function with "simple" proximal 
    :param sigma: Step size parameter for Primal problem
    :param tau: Step size parameter for Dual problem
        
    Remark: Convergence is guaranted provided that
        
    .. math:: 
    
      \tau \sigma \|K\|^{2} <1
        
            
    Reference:
        
        
        (a) A. Chambolle and T. Pock (2011), "A first-order primal–dual algorithm for convex
        problems with applications to imaging", J. Math. Imaging Vision 40, 120–145.        
        
        
        (b) E. Esser, X. Zhang and T. F. Chan (2010), "A general framework for a class of first
        order primal–dual algorithms for convex optimization in imaging science",
        SIAM J. Imaging Sci. 3, 1015–1046.
    '''

    def __init__(self, f=None, g=None, operator=None, tau=None, sigma=1.,initial=None, use_axpby=True, **kwargs):
        '''PDHG algorithm creator

        Optional parameters

        :param operator: a Linear Operator
        :param f: Convex function with "simple" proximal of its conjugate. 
        :param g: Convex function with "simple" proximal 
        :param sigma: Step size parameter for Primal problem
        :param tau: Step size parameter for Dual problem
        :param initial: Initial guess ( Default initial = 0)
        '''
        super(PDHG, self).__init__(**kwargs)
        if kwargs.get('x_init', None) is not None:
            if initial is None:
                warnings.warn('The use of the x_init parameter is deprecated and will be removed in following version. Use initial instead',
                   DeprecationWarning, stacklevel=4)
                initial = kwargs.get('x_init', None)
            else:
                raise ValueError('{} received both initial and the deprecated x_init parameter. It is not clear which one we should use.'\
                    .format(self.__class__.__name__))
        self._use_axpby = use_axpby

        if f is not None and operator is not None and g is not None:
            self.set_up(f=f, g=g, operator=operator, tau=tau, sigma=sigma, initial=initial, **kwargs)

    def set_up(self, f, g, operator, tau=None, sigma=1., initial=None, **kwargs):
        '''initialisation of the algorithm

        :param operator: a Linear Operator
        :param f: Convex function with "simple" proximal of its conjugate. 
        :param g: Convex function with "simple" proximal 
        :param sigma: Step size parameter for Primal problem
        :param tau: Step size parameter for Dual problem
        :param initial: Initial guess ( Default initial = 0)'''

        print("{} setting up".format(self.__class__.__name__, ))
        
        # can't happen with default sigma
        if sigma is None and tau is None:
            raise ValueError('Need sigma*tau||K||^2<1')

        # algorithmic parameters
        self.f = f
        self.g = g
        self.operator = operator

        self.tau = tau
        self.sigma = sigma

        if self.tau is None:
            # Compute operator Norm
            normK = self.operator.norm()
            # Primal & dual stepsizes
            self.tau = 1 / (self.sigma * normK ** 2)
        
        if initial is None:
            self.x_old = self.operator.domain_geometry().allocate()
        else:
            self.x_old = initial.copy()

        self.x = operator.domain_geometry().allocate(0)        
        self.y = self.operator.range_geometry().allocate(0)
        self.y_old = self.operator.range_geometry().allocate(0)
        
        self.x_tmp = self.x_old.copy()
        
        self.y_tmp = self.y_old.copy()
        
        self.xbar = self.x_old.copy()
                
        # relaxation parameter, default value is 1.0
        self.theta = kwargs.get('theta',1.0)

        # Strongly convex case either F or G or both
        self.gamma = kwargs.get('gamma', None)
        
        self.configured = True
        print("{} configured".format(self.__class__.__name__, ))

    def update_previous_solution(self):
        # swap the pointers to current and previous solution
        tmp = self.x_old
        self.x_old = self.x
        self.x = tmp
        tmp = self.y_old
        self.y_old = self.y
        self.y = tmp
        
    def update(self):
        
        # Gradient ascent for the dual variable
        self.operator.direct(self.xbar, out=self.y_tmp)
        
        # self.y_tmp *= self.sigma
        # self.y_tmp += self.y_old
        if self._use_axpby:
            self.y_tmp.axpby(self.sigma, 1 , self.y_old, self.y_tmp)
        else:
            self.y_tmp *= self.sigma
            self.y_tmp += self.y_old

        # self.y = self.f.proximal_conjugate(self.y_old, self.sigma)
        self.f.proximal_conjugate(self.y_tmp, self.sigma, out=self.y)
        
        # Gradient descent for the primal variable
        self.operator.adjoint(self.y, out=self.x_tmp)
        # self.x_tmp *= -1*self.tau
        # self.x_tmp += self.x_old
        if self._use_axpby:
            self.x_tmp.axpby(-self.tau, 1. , self.x_old, self.x_tmp)
        else:
            self.x_tmp *= -1*self.tau
            self.x_tmp += self.x_old

        self.g.proximal(self.x_tmp, self.tau, out=self.x)

        # Update
        self.x.subtract(self.x_old, out=self.xbar)
        # self.xbar *= self.theta
        # self.xbar += self.x
        if self._use_axpby:
            self.xbar.axpby(self.theta, 1 , self.x, self.xbar)
        else:
            self.xbar *= self.theta
            self.xbar += self.x
  
        if self.gamma is not None:
            self.theta = float(1 / numpy.sqrt(1 + 2 * self.gamma * self.tau))
            self.tau *= self.theta
            self.sigma /= self.theta            
                

    def update_objective(self):

        p1 = self.f(self.operator.direct(self.x)) + self.g(self.x)
        d1 = -(self.f.convex_conjugate(self.y) + self.g.convex_conjugate(-1*self.operator.adjoint(self.y)))

        self.loss.append([p1, d1, p1-d1])
        
    @property
    def objective(self):
        '''alias of loss'''
        return [x[0] for x in self.loss]

    @property
    def dual_objective(self):
        return [x[1] for x in self.loss]
    
    @property
    def primal_dual_gap(self):
        return [x[2] for x in self.loss]


if __name__ == "__main__":

    # Import libraries
    from cil.utilities import dataexample, noise
    from cil.optimisation.operators import GradientOperator
    from cil.optimisation.functions import MixedL21Norm, L2NormSquared
    from cil.utilities.display import show2D
    from cil.io import NEXUSDataWriter, NEXUSDataReader
    import pickle
    import matplotlib.pyplot as plt
    import os

    # Load data
    data = dataexample.SHAPES.get()

    # Add gaussian noise
    noisy_data = noise.gaussian(data, seed = 10, var = 0.02)     

    ig = noisy_data.geometry

    alpha = 0.2
    K = GradientOperator(ig)
    F = alpha * MixedL21Norm()
    G = 0.5 * L2NormSquared(b=noisy_data)

    normK = K.norm()
    sigma = 1./normK
    tau = 1./normK

    pdhg_noaccel = PDHG(f = F, g = G, operator = K, 
                   update_objective_interval=1, 
                   max_iteration=1000, sigma=sigma, tau=tau)
    pdhg_noaccel.run(verbose=0)

    name_recon = "pdhg_noaccel"
    writer = NEXUSDataWriter(file_name = os.getcwd() + name_recon + ".nxs",
                         data = pdhg_noaccel.solution)
    writer.write() 

    pdhg_noaccel_info = {}
    pdhg_noaccel_info['primal'] = pdhg_noaccel.objective
    pdhg_noaccel_info['dual'] = pdhg_noaccel.dual_objective
    pdhg_noaccel_info['pdgap'] = pdhg_noaccel.primal_dual_gap

    with open(os.getcwd() + 'pdhg_noaccel_info.pkl','wb') as f:
        pickle.dump(pdhg_noaccel_info, f) 

    pdhg_accel = PDHG(f = F, g = G, operator = K, 
                    update_objective_interval=1, max_iteration=1000, 
                      gamma = 0.2, sigma=sigma, tau=tau)
    pdhg_accel.run(verbose=0)  

    # Load pdhg_noaccel_info
    pdhg_noaccel_info = pickle.load( open( os.getcwd() + 'pdhg_noaccel_info.pkl', "rb" ) )         

    plt.figure()
    plt.loglog(pdhg_accel.objective, label="Accelerate")
    plt.loglog(pdhg_noaccel_info['primal'], label="No accelerate")
    plt.legend()
    plt.title("Primal")
    plt.show()

    plt.figure()
    plt.loglog(pdhg_accel.primal_dual_gap, label="Accelerate")
    plt.loglog(pdhg_noaccel_info['pdgap'], label="No accelerate")
    plt.legend()
    plt.title("PD - GAP")
    plt.show() 

    plt.figure()
    plt.loglog(pdhg_accel.dual_objective, label="Accelerate")
    plt.loglog(pdhg_noaccel_info['dual'], label="No accelerate")
    plt.legend()
    plt.title("Dual")
    plt.show()        

    reader_pdhg = NEXUSDataReader(file_name = os.getcwd() + "pdhg_noaccel" + ".nxs")
    pdhg_accel_solution = reader_pdhg.load_data()
        
    show2D([pdhg_noaccel.solution,
            pdhg_accel_solution, (pdhg_noaccel.solution - pdhg_accel_solution).abs()], num_cols=1, origin="upper")



