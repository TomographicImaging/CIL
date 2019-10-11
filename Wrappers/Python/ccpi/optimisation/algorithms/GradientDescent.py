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

from ccpi.optimisation.algorithms import Algorithm

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
        '''
        super(GradientDescent, self).__init__(**kwargs)

        
        if x_init is not None and objective_function is not None and rate is not None:
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
        self.rate = rate

        self.loss.append(objective_function(x_init))
        self.iteration = 0

        try:
            self.memopt = self.objective_function.memopt
        except AttributeError as ae:
            self.memopt = False
        if self.memopt:
            self.x_update = x_init.copy()

        self.configured = True
        print("{} configured".format(self.__class__.__name__, ))

    def update(self):
        '''Single iteration'''
        if self.memopt:
            self.objective_function.gradient(self.x, out=self.x_update)
            self.x_update *= -self.rate
            self.x += self.x_update
        else:
            self.x += -self.rate * self.objective_function.gradient(self.x)

    def update_objective(self):
        self.loss.append(self.objective_function(self.x))
