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

import numpy
from ccpi.optimisation.algorithms import Algorithm

class GradientDescent(Algorithm):
    ''' 
    
        Gradient Descent algorithm
        
        
    '''

    def __init__(self, x_init=None, objective_function=None, step_size =None, **kwargs):
        '''GradientDescent algorithm creator
        
        initialisation can be done at creation time if all 
        proper variables are passed or later with set_up
        
        :param x_init: initial guess
        :param objective_function: objective function to be minimised
        :param step_size: step size for gradient descent iteration
        :param alpha: optional parameter to start the backtracking algorithm, default 1e6
        :param beta: optional parameter defining the reduction of step, default 0.5.
                    It's value can be in (0,1)
        :param rtol: optional parameter defining the relative tolerance comparing the 
                     current objective function to 0, default 1e-5, see numpy.isclose
        :param atol: optional parameter defining the absolute tolerance comparing the 
                     current objective function to 0, default 1e-8, see numpy.isclose
        '''
        super(GradientDescent, self).__init__(**kwargs)

        self.alpha = kwargs.get('alpha' , 1e6)
        self.beta = kwargs.get('beta', 0.5)
        self.rtol = kwargs.get('rtol', 1e-5)
        self.atol = kwargs.get('atol', 1e-8)
        if x_init is not None and objective_function is not None :
            self.set_up(x_init=x_init, objective_function=objective_function, step_size=step_size)
    
    def set_up(self, x_init, objective_function, step_size):
        '''initialisation of the algorithm
        
        :param x_init: initial guess
        :param objective_function: objective function to be minimised
        :param step_size: step size'''
        print("{} setting up".format(self.__class__.__name__, ))
            
        self.x = x_init.copy()
        self.objective_function = objective_function
        

        if step_size is None:
            self.k = 0
            self.update_step_size = True
            self.x_armijo = x_init.copy()
            # self.rate = self.armijo_rule() * 2
            # print (self.rate)
        else:
            self.step_size = step_size
            self.update_step_size = False
        
        
        self.update_objective()
        self.iteration = 0

        self.x_update = x_init.copy()

        self.configured = True
        print("{} configured".format(self.__class__.__name__, ))

    def update(self):
        '''Single iteration'''
        
        self.objective_function.gradient(self.x, out=self.x_update)
        
        if self.update_step_size:
            # the next update and solution are calculated within the armijo_rule
            self.step_size = self.armijo_rule()
        else:
            self.x_update *= -self.step_size
            self.x += self.x_update
        
    

    def update_objective(self):
        self.loss.append(self.objective_function(self.x))

    def armijo_rule(self):
        '''Applies the Armijo rule to calculate the step size (step_size)

        https://projecteuclid.org/download/pdf_1/euclid.pjm/1102995080

        The Armijo rule runs a while loop to find the appropriate step_size by starting
        from a very large number (alpha). The step_size is found by dividing alpha by 2 
        in an iterative way until a certain criterion is met. To avoid infinite loops, we 
        add a maximum number of times the while loop is run. 

        This rule would allow to reach a minimum step_size of 10^-alpha. 
        
        if
        alpha = numpy.power(10,gamma)
        delta = 3
        step_size = numpy.power(10, -delta)
        with armijo rule we can get to step_size from initial alpha by repeating the while loop k times 
        where 
        alpha / 2^k = step_size
        10^gamma / 2^k = 10^-delta 
        2^k = 10^(gamma+delta)
        k = gamma+delta / log10(2) \\approx 3.3 * (gamma+delta)

        if we would take by default delta = gamma
        kmax = numpy.ceil ( 2 * gamma / numpy.log10(2) )
        kmax = numpy.ceil (2 * numpy.log10(alpha) / numpy.log10(2))

        '''
        f_x = self.objective_function(self.x)
        if not hasattr(self, 'x_update'):
            self.x_update = self.objective_function.gradient(self.x)
        
        while self.k < self.kmax:
            # self.x - alpha * self.x_update
            self.x_update.multiply(self.alpha, out=self.x_armijo)
            self.x.subtract(self.x_armijo, out=self.x_armijo)
            
            f_x_a = self.objective_function(self.x_armijo)
            sqnorm = self.x_update.squared_norm()
            if f_x_a - f_x <= - ( self.alpha/2. ) * sqnorm:
                self.x.fill(self.x_armijo)
                break
            else:
                self.k += 1.
                # we don't want to update kmax
                self._alpha *= self.beta

        if self.k == self.kmax:
            raise ValueError('Could not find a proper step_size in {} loops. Consider increasing alpha.'.format(self.kmax))
        return self.alpha
    
    @property
    def alpha(self):
        return self._alpha
    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha
        self.kmax = numpy.ceil (2 * numpy.log10(alpha) / numpy.log10(2))
        self.k = 0
        
    def should_stop(self):
        return self.max_iteration_stop_cryterion() or \
            numpy.isclose(self.get_last_objective(), 0., rtol=self.rtol, 
                atol=self.atol, equal_nan=False)


