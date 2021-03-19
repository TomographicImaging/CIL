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

class LADMM(Algorithm):
        
    ''' 
        LADMM is the Linearized Alternating Direction Method of Multipliers (LADMM)
    
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

    def __init__(self, f=None, g=None, operator=None, \
                       tau = None, sigma = 1., 
                       initial = None, **kwargs):
        
        '''Initialisation of the algorithm

        :param operator: a Linear Operator
        :param f: Convex function with "simple" proximal
        :param g: Convex function with "simple" proximal 
        :param sigma: Positive step size parameter 
        :param tau: Positive step size parameter
        :param initial: Initial guess ( Default initial_guess = 0)'''        
        
        super(LADMM, self).__init__(**kwargs)
        if kwargs.get('x_init', None) is not None:
            if initial is None:
                warnings.warn('The use of the x_init parameter is deprecated and will be removed in following version. Use initial instead',
                   DeprecationWarning, stacklevel=4)
                initial = kwargs.get('x_init', None)
            else:
                raise ValueError('{} received both initial and the deprecated x_init parameter. It is not clear which one we should use.'\
                    .format(self.__class__.__name__))

        self.set_up(f = f, g = g, operator = operator, tau = tau,\
             sigma = sigma, initial=initial)        
                    
    def set_up(self, f, g, operator, tau = None, sigma=1., \
        initial=None):

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
            
        if initial is None:
            self.x = self.operator.domain_geometry().allocate()
        else:
            self.x = initial.copy()
         
        # allocate space for operator direct & adjoint    
        self.tmp_dir = self.operator.range_geometry().allocate()
        self.tmp_adj = self.operator.domain_geometry().allocate()            
            
        self.z = self.operator.range_geometry().allocate() 
        self.u = self.operator.range_geometry().allocate() 

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