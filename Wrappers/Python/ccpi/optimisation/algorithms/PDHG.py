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
from ccpi.framework import ImageData, DataContainer
import numpy as np
import numpy
import time
from ccpi.optimisation.operators import BlockOperator
from ccpi.framework import BlockDataContainer
from ccpi.optimisation.functions import FunctionOperatorComposition


class PDHG(Algorithm):
    '''Primal Dual Hybrid Gradient'''

    def __init__(self, **kwargs):
        super(PDHG, self).__init__(max_iteration=kwargs.get('max_iteration',0))
        f        = kwargs.get('f', None)
        operator = kwargs.get('operator', None)
        g        = kwargs.get('g', None)
        tau      = kwargs.get('tau', None)
        sigma    = kwargs.get('sigma', 1.)

        if f is not None and operator is not None and g is not None:
            print(self.__class__.__name__ , "set_up called from creator")
            self.set_up(f=f, g=g, operator=operator, tau=tau, sigma=sigma)

    def set_up(self, f, g, operator, tau=None, sigma=1.):

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


        self.x_old = self.operator.domain_geometry().allocate()
        self.x_tmp = self.x_old.copy()
        self.x = self.x_old.copy()
    
        self.y_old = self.operator.range_geometry().allocate()
        self.y_tmp = self.y_old.copy()
        self.y = self.y_old.copy()

        self.xbar = self.x_old.copy()

        # relaxation parameter
        self.theta = 1
        self.update_objective()
        self.configured = True

    def update(self):
        
        # Gradient descent, Dual problem solution
        self.operator.direct(self.xbar, out=self.y_tmp)
        self.y_tmp *= self.sigma
        self.y_tmp += self.y_old

        # self.y = self.f.proximal_conjugate(self.y_old, self.sigma)
        self.f.proximal_conjugate(self.y_tmp, self.sigma, out=self.y)
        
        # Gradient ascent, Primal problem solution
        self.operator.adjoint(self.y, out=self.x_tmp)
        self.x_tmp *= -1*self.tau
        self.x_tmp += self.x_old

        self.g.proximal(self.x_tmp, self.tau, out=self.x)

        # Update
        self.x.subtract(self.x_old, out=self.xbar)
        self.xbar *= self.theta
        self.xbar += self.x

        self.x_old.fill(self.x)
        self.y_old.fill(self.y)

    def update_objective(self):

        p1 = self.f(self.operator.direct(self.x)) + self.g(self.x)
        d1 = -(self.f.convex_conjugate(self.y) + self.g.convex_conjugate(-1*self.operator.adjoint(self.y)))

        self.loss.append([p1, d1, p1-d1])