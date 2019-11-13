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
from __future__ import unicode_literals
<<<<<<< HEAD

from ccpi.optimisation.algorithms import Algorithm, StochasticAlgorithm
=======
import numpy
from ccpi.optimisation.algorithms import Algorithm
>>>>>>> armijo_rule

class GradientDescent(Algorithm):
    ''' 
    
        Gradient Descent algorithm
        
        
    '''

    def __init__(self, x_init=None, objective_function=None, rate=None, **kwargs):
        '''GradientDescent algorithm creator
        
        initialisation can be done at creation time if all 
        proper variables are passed or later with set_up
        
        :param x_init: initial guess
        :param objective_function: objective function to be minimised
        :param rate: step rate
        :param alpha: optional parameter to start the backtracking algorithm
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
            self.set_up(x_init=x_init, objective_function=objective_function, rate=rate)
    
    def should_stop(self):
        '''stopping criterion, currently only based on number of iterations'''
        return self.iteration >= self.max_iteration
    
    def set_up(self, x_init, objective_function, rate):
        '''initialisation of the algorithm
        
        :param x_init: initial guess
        :param objective_function: objective function to be minimised
        :param rate: step rate'''
        print("{} setting up".format(self.__class__.__name__, ))
            
        self.x = x_init.copy()
        self.objective_function = objective_function
        

        if rate is None:
            self.k = 0
            self.update_rate = True
            self.x_armijo = x_init.copy()
            # self.rate = self.armijo_rule() * 2
            # print (self.rate)
        else:
            self.rate = rate
            self.update_rate = False
        
        
        self.update_objective()
        self.iteration = 0

        self.x_update = x_init.copy()

        self.configured = True
        print("{} configured".format(self.__class__.__name__, ))

    def update(self):
        '''Single iteration'''
        
        self.objective_function.gradient(self.x, out=self.x_update)
        
        if self.update_rate:
            # the next update and solution are calculated within the armijo_rule
            self.rate = self.armijo_rule() * 2.
        else:
            self.x_update *= -self.rate
            self.x += self.x_update
        
    

    def update_objective(self):
        self.loss.append(self.objective_function(self.x))

    def armijo_rule(self):
        '''Applies the Armijo rule to calculate the step size (rate)

        https://projecteuclid.org/download/pdf_1/euclid.pjm/1102995080
        '''
        f_x = self.objective_function(self.x)
        if not hasattr(self, 'x_update'):
            self.x_update = self.objective_function.gradient(self.x)
        while True:
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
                self.alpha *= self.beta
        return self.alpha

    def should_stop(self):
        return self.max_iteration_stop_cryterion() or \
            numpy.isclose(self.get_last_objective(), 0., rtol=self.rtol, atol=self.atol, equal_nan=False)


class StochasticGradientDescent(StochasticAlgorithm, GradientDescent):
    def __init__(self, **kwargs):
        super(StochasticGradientDescent, self).__init__(**kwargs)
        
    def notify_new_subset(self, subset_id, number_of_subsets):
        self.objective_function.notify_new_subset(subset_id, number_of_subsets)