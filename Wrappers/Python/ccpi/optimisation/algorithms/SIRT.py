#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:39:35 2019
 @author: jakob
"""

 # -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library developed by
#   Visual Analytics and Imaging System Group of the Science Technology
#   Facilities Council, STFC

 #   Copyright 2018 Edoardo Pasca

 #   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

 #       http://www.apache.org/licenses/LICENSE-2.0

 #   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""
Created on Thu Feb 21 11:11:23 2019
 @author: ofn77899
"""

 from ccpi.optimisation.algorithms import Algorithm
from ccpi.framework import ImageData, AcquisitionData

 #from collections.abc import Iterable
class SIRT(Algorithm):

     '''Simultaneous Iterative Reconstruction Technique
     Parameters:
      x_init: initial guess
      operator: operator for forward/backward projections
      data: data to operate on
      constraint: Function with prox-method, for example IndicatorBox to 
                  enforce box constraints.
    '''
    def __init__(self, **kwargs):
        super(SIRT, self).__init__()
        self.x          = kwargs.get('x_init', None)
        self.operator   = kwargs.get('operator', None)
        self.data       = kwargs.get('data', None)
        self.constraint = kwargs.get('data', None)
        if self.x is not None and self.operator is not None and \
           self.data is not None:
            print ("Calling from creator")
            self.set_up(x_init  =kwargs['x_init'],
                               operator=kwargs['operator'],
                               data    =kwargs['data'])

     def set_up(self, x_init, operator , data, constraint=None ):

         self.x = x_init.copy()
        self.operator = operator
        self.data = data
        self.constraint = constraint

         self.r = data.copy()

         self.relax_par = 1.0

         # Set up scaling matrices D and M.
        im1 = ImageData(geometry=self.x.geometry)
        im1.array[:] = 1.0
        self.M = 1/operator.direct(im1)
        aq1 = AcquisitionData(geometry=self.M.geometry)
        aq1.array[:] = 1.0
        self.D = 1/operator.adjoint(aq1)


     def update(self):

         self.r = self.data - self.operator.direct(self.x)

         self.x += self.relax_par * (self.D*self.operator.adjoint(self.M*self.r))

         if self.constraint != None:
            self.x = self.constraint.prox(self.x,None)

     def update_objective(self):
        self.loss.append(self.r.squared_norm()) 
        