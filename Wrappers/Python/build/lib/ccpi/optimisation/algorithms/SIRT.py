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

from ccpi.optimisation.algorithms import Algorithm

class SIRT(Algorithm):

    '''Simultaneous Iterative Reconstruction Technique

    Parameters:
      x_init: initial guess
      operator: operator for forward/backward projections
      data: data to operate on
      constraint: Function with prox-method, for example IndicatorBox to 
                  enforce box constraints, default is None).
    '''
    def __init__(self, **kwargs):
        super(SIRT, self).__init__()

        x_init     = kwargs.get('x_init', None)
        operator   = kwargs.get('operator', None)
        data       = kwargs.get('data', None)
        constraint = kwargs.get('constraint', None)

        if x_init is not None and operator is not None and data is not None:
            print(self.__class__.__name__, "set_up called from creator")
            self.set_up(x_init=x_init, operator=operator, data=data, constraint=constraint)

    def set_up(self, x_init, operator, data, constraint=None):
        
        self.x = x_init.copy()
        self.operator = operator
        self.data = data
        self.constraint = constraint
        
        self.r = data.copy()
        
        self.relax_par = 1.0
        
        # Set up scaling matrices D and M.
        self.M = 1/self.operator.direct(self.operator.domain_geometry().allocate(value=1.0))        
        self.D = 1/self.operator.adjoint(self.operator.range_geometry().allocate(value=1.0))
        self.update_objective()
        self.configured = True


    def update(self):
        
        self.r = self.data - self.operator.direct(self.x)
        
        self.x += self.relax_par * (self.D*self.operator.adjoint(self.M*self.r))
        
        if self.constraint is not None:
            self.x = self.constraint.proximal(self.x, None)
            # self.constraint.proximal(self.x,None, out=self.x)

    def update_objective(self):
        self.loss.append(self.r.squared_norm())
