#  Copyright 2020 United Kingdom Research and Innovation
#  Copyright 2020 The University of Manchester
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Authors:
# CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt

from cil.optimisation.algorithms import Algorithm
import logging

log = logging.getLogger(__name__)


class LADMM(Algorithm):

    r'''
        LADMM is the Linearized Alternating Direction Method of Multipliers (LADMM)

        General form of ADMM : :math:`min_{x} f(x) + g(y)`, subject to :math:`Ax + By = b`

        Case: :math:`A = Id, B = -K, b = 0   ==> min_x f(Kx) + g(x)`

        The quadratic term in the augmented Lagrangian is linearized for the x-update.

        Main algorithmic difference is that in ADMM we compute two proximal subproblems,
        where in the PDHG a proximal and proximal conjugate.

        Reference (Section 8) : https://link.springer.com/content/pdf/10.1007/s10107-018-1321-1.pdf
        
        
            .. math:: x^{k} = prox_{\tau f } (x^{k-1} - \tau/\sigma A^{T}(Ax^{k-1} - z^{k-1} + u^{k-1} )

            .. math:: z^{k} = prox_{\sigma g} (Ax^{k} + u^{k-1})

            .. math:: u^{k} = u^{k-1} + Ax^{k} - z^{k}

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

        self.set_up(f = f, g = g, operator = operator, tau = tau,\
             sigma = sigma, initial=initial)

    def set_up(self, f, g, operator, tau = None, sigma=1., initial=None):
        log.info("%s setting up", self.__class__.__name__)

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

        log.info("%s configured", self.__class__.__name__)

    def update(self):

        self.tmp_dir += self.u
        self.tmp_dir -= self.z
        self.operator.adjoint(self.tmp_dir, out = self.tmp_adj)

        self.x.sapyb(1, self.tmp_adj, -(self.tau/self.sigma), out=self.x)

        # apply proximal of f
        tmp = self.f.proximal(self.x, self.tau)
        self.operator.direct(tmp, out=self.tmp_dir)
        # store the result in x
        self.x.fill(tmp)
        del tmp

        self.u += self.tmp_dir

        # apply proximal of g
        self.g.proximal(self.u, self.sigma, out = self.z)

        # update
        self.u -= self.z

    def update_objective(self):

        self.loss.append(self.f(self.x) +  self.g(self.operator.direct(self.x)) )
