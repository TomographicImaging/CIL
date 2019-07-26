# -*- coding: utf-8 -*-
# Copyright 2019 Science Technology Facilities Council
# Copyright 2019 University of Manchester
#
# This work is part of the Core Imaging Library developed by Science Technology
# Facilities Council and University of Manchester
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ccpi.optimisation.algorithms import Algorithm
from ccpi.optimisation.functions import ZeroFunction
import numpy
from numbers import Number

class ADMM(Algorithm):
    
    r'''Alternating Directions Method of Multipliers Algorithm 
    
    Problem : 
    
    .. math::
    
      \min_{x} f(x) + g(x)
    
    |
    
    Parameters :
        
      :parameter x_init : Initial guess ( Default x_init = 0)
      :parameter f : Differentiable function
      :parameter g : Convex function with " simple " proximal operator


    Reference:
      
        Beck, A. and Teboulle, M., 2009. A fast iterative shrinkage-thresholding 
        algorithm for linear inverse problems. 
        SIAM journal on imaging sciences,2(1), pp.183-202.
    '''
    
    
    def __init__(self, **kwargs):
        
        '''creator 
        
        initialisation can be done at creation time if all 
        proper variables are passed or later with set_up'''
        
        super(ADMM, self).__init__()

        self.x_init = kwargs.get('x_init', None)
        tau_f = kwargs.get('tau_f', -1.)
        tau_g = kwargs.get('tau_g', -1.)
        f = kwargs.get('f', None)
        g = kwargs.get('g', None)

        if self.x_init is not None and \
            tau_f > 0. and tau_g > 0. and \
            f is not None and g is not None:
            print (self.__class__.__name__ , 'set_up called from creator.')
            self.set_up(self.x_init, f, g, tau_f, tau_g)
    def check_tau(self, tau):
        if isinstance(tau, Number):
            if tau < 0. :
                raise ValueError('Tau is expected to be positive. Got', tau)
        else:
            raise TypeError('Tau is expected to be a number. Got', type(tau))
        return tau
    def set_up(self, x_init, f, g, tau_f, tau_g,  **kwargs):
        self.tau_f = self.check_tau(tau_f)
        self.tau_g = self.check_tau(tau_g)
        self.f = f
        self.g = g
        
        # allocate memory
        self.u = x_init * 0.
        self.y = x_init.copy()
        self.x = x_init.copy()
        self.x_previous = x_init.copy()
        self.v = self.x.add(self.u)
        self.g.proximal(self.v, self.tau_g, out=self.v)

        
        self.update_objective()
        self.configured = True

    def update(self):
        # update solution
        # x_{k+1} = prox_tf (v_k -u_k)
        self.v -= self.u
        self.f.proximal(self.v, self.tau_f, out=self.x)

        # v_{k+1} = prox_tg (x_{k+1} +u_k)
        self.x.add(self.u, out = self.v)
        self.g.proximal(self.v, self.tau_g, out=self.v)

        # u_{k+1} = u_k + x_{k+1} -v_{k+1}
        self.u += self.x
        self.u -= self.v

        self.x_previous.fill(self.x)



    def update_objective(self):
        self.loss.append( self.f(self.x) + self.g(self.x) )