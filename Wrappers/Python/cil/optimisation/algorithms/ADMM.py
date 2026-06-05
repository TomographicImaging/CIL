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
    r"""
    LADMM is the Linearized Alternating Direction Method of Multipliers (LADMM).

    The general form of ADMM is given by the following optimization problem: 
    
    .. math::
    
        \min_{x} f(x) + g(y), \text{ subject to } Ax + By = b

    In CIL, we have implemented the case where :math:`A = Id`, :math:`B = -K`, :math:`b = 0`  which gives 
    
    .. math::
        
        \min_x f(Kx) + g(x).
        
    The algorithm is given by the following iteration, for :math:`k\geq 1`:
    
    .. math::

        \begin{cases}
            x_{k} = \mathrm{prox}_{\tau f} \left(x_{k-1} - \dfrac{\tau}{\sigma} A_{T}\left(Ax_{k-1} - z_{k-1} + u_{k-1} \right)  \right)\\
            z_{k} = \mathrm{prox}_{\sigma g} \left(Ax_{k} + u_{k-1}\right) \\
            u_{k} = u_{k-1} + Ax_{k} - z_{k}
        \end{cases}
        
    where :math:`\mathrm{prox}_{\tau f}` is the proximal operator of :math:`f` and :math:`\mathrm{prox}_{\sigma g}` is the proximal operator of :math:`g`.
    
    
    Parameters
    ------------
    operator:  CIL Linear Operator
        Operator :math:`K` in the objective function
    f: CIL Function
        Convex function with "simple" proximal
    g: CIL Function
        Convex function with "simple" proximal
    sigma: float, positive, defaults to 1.
        Positive step size parameter
    tau: float, positive, defaults to :math:`\frac{\sigma}{\|K\|^{2}}`
        Positive step size parameter
    initial: DataContainer, defaults to DataContainer filled with zeros
        Initial guess 
            
    
    Note
    ----
    This implementation of ADMM minimises the same objective function as the Primal-Dual Hybrid Gradient (PDHG) method.
    The main algorithmic difference is that in ADMM we compute the proximal of :math:`f` and :math:`g` 
    where in the PDHG this is a proximal-conjugate and proximal.
    
    
    Note
    -----
    Reference (Section 8) : O’Connor, D., Vandenberghe, L. On the equivalence of the primal-dual hybrid gradient method and Douglas–Rachford splitting. Math. Program. 179, 85–108 (2020). https://doi.org/10.1007/s10107-018-1321-1

    """


    def __init__(self, f=None, g=None, operator=None, \
                       tau = None, sigma = 1.,
                       initial = None, **kwargs):

        """Initialisation of the algorithm."""

        super(LADMM, self).__init__(**kwargs)

        self.set_up(f = f, g = g, operator = operator, tau = tau,\
             sigma = sigma, initial=initial)

    def set_up(self, f, g, operator, tau = None, sigma=1., initial=None):
        """Set up of the algorithm."""
        log.info("%s setting up", self.__class__.__name__)

        self.f = f
        self.g = g
        self.operator = operator

        self.tau = tau
        self.sigma = sigma

        if self.tau is None:
            normK = self.operator.norm()
            self.tau = self.sigma / normK ** 2

        if initial is None:
            self.x = self.operator.domain_geometry().allocate(0)
        else:
            self.x = initial.copy()

        # allocate space for operator direct & adjoint
        self.tmp_dir = self.operator.range_geometry().allocate(0)
        self.tmp_adj = self.operator.domain_geometry().allocate(0)

        self.z = self.operator.range_geometry().allocate(0)
        self.u = self.operator.range_geometry().allocate(0)

        self.configured = True

        log.info("%s configured", self.__class__.__name__)

    def update(self):
        """Performs a single iteration of the LADMM algorithm"""
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
        """Update the objective function value"""
        self.loss.append(self.f(self.x) +  self.g(self.operator.direct(self.x)) )
