# -*- coding: utf-8 -*-
#========================================================================
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
#
#=========================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ccpi.optimisation.algorithms import Algorithm
from ccpi.optimisation.functions import IndicatorBox

from numpy import inf

class SIRT(Algorithm):

    r'''Simultaneous Iterative Reconstruction Technique
    
    Problem: 
    
    .. math::  
    
      A x = b
    
    :param x_init: Initial guess
    :param operator: Linear operator for the inverse problem
    :param data: Acquired data to reconstruct       
    :param constraint: Function proximal method
                e.g.  :math:`x\in[0, 1]`, :code:`IndicatorBox` to enforce box constraints
                        Default is :code:`None`).
    '''
    def __init__(self, x_init=None, operator=None, data=None, lower=None, upper=None, constraint=None, **kwargs):
        '''SIRT algorithm creator

       Optional parameters:

      :param x_init: Initial guess
      :param operator: Linear operator for the inverse problem
      :param data: Acquired data to reconstruct 
      :param lower: Scalar specifying lower bound constraint on pixel values, default -inf
      :param upper: Scalar specifying upper bound constraint on pixel values, default +inf
      :param constraint: More general constraint, given as Function proximal method
                   e.g.  :math:`x\in[0, 1]`, :code:`IndicatorBox` to enforce box constraints
                         Default is :code:`None`). constraint takes priority over lower and upper.'''
        super(SIRT, self).__init__(**kwargs)

        if x_init is not None and operator is not None and data is not None:
            self.set_up(x_init=x_init, operator=operator, data=data, lower=lower, upper=upper, constraint=constraint)

    def set_up(self, x_init, operator, data, lower=None, upper=None, constraint=None):
        '''initialisation of the algorithm

        :param x_init: Initial guess
        :param operator: Linear operator for the inverse problem
        :param data: Acquired data to reconstruct
        :param lower: Scalar specifying lower bound constraint on pixel values, default -inf
        :param upper: Scalar specifying upper bound constraint on pixel values, default +inf
        :param constraint: More general constraint, given as Function proximal method
                   e.g.  :math:`x\in[0, 1]`, :code:`IndicatorBox` to enforce box constraints
                         Default is :code:`None`). constraint takes priority over lower and upper.'''
        print("{} setting up".format(self.__class__.__name__, ))
        
        self.x = x_init.copy()
        self.operator = operator
        self.data = data
        
        self.r = data.copy()
        
        self.relax_par = 1.0
        
        # Set constraints. If "constraint" is given, it should be an indicator
        # function, and in that case "lower" and "upper" inputs are ignored. If
        # "constraint" is not given, then "lower" and "upper" are looked at, 
        # and if at least one is not None, then an IndicatorBox is set up which 
        # provides the proximal mapping to enforce lower and upper bounds.
        self.constraint = constraint
        if constraint is None:
            if lower is not None or upper is not None:
                if lower is None:
                    lower=-inf
                if upper is None:
                    upper=inf
                self.constraint=IndicatorBox(lower=lower,upper=upper)
                
        # Set up scaling matrices D and M.
        self.M = 1/self.operator.direct(self.operator.domain_geometry().allocate(value=1.0))        
        self.D = 1/self.operator.adjoint(self.operator.range_geometry().allocate(value=1.0))
        self.configured = True
        print("{} configured".format(self.__class__.__name__, ))



    def update(self):
        
        self.r = self.data - self.operator.direct(self.x)
        
        self.x += self.relax_par * (self.D*self.operator.adjoint(self.M*self.r))
        
        if self.constraint is not None:
            self.x = self.constraint.proximal(self.x, None)
            # self.constraint.proximal(self.x,None, out=self.x)

    def update_objective(self):
        self.loss.append(self.r.squared_norm())
