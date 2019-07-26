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
        self.tau = kwargs.get('tau', -1.)
        f = kwargs.get('f', None)
        g = kwargs.get('g', None)

        if self.x_init is not None and self.tau > 0. and \
            f is not None and g is not None:
            print (self.__name__ , 'set_up called from creator.')
            self.set_up(x_init, tau, f, g)

    def set_up(self, x_init, tau, f, g,  **kwargs):
        self.u = x_init * 0.
        self.y = x_init.copy()

    def update_objective(self):
