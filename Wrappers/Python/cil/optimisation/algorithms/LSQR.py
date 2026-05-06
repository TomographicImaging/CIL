#  Copyright 2019 United Kingdom Research and Innovation
#  Copyright 2019 The University of Manchester
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
# CIL Developers and contributers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt


from cil.optimisation.algorithms import Algorithm
import numpy
import logging
import warnings 
import math

log = logging.getLogger(__name__)

class LSQR(Algorithm):

    
    r"""
    Least Squares with QR factorisation (LSQR) algorithm.

    The LSQR algorithm is used to solve large-scale linear systems and least-squares problems, particularly when the matrix is sparse or implicitly defined.

    Solves the problem:

    .. math::

        \min_x \|Ax - b\|_2^2

    Optionally, with Tikhonov regularisation:

    .. math::

        \min_x \|Ax - b\|_2^2 + \alpha^2 \|x\|_2^2

    Parameters
    ----------
    operator : Operator
        Linear operator representing the forward model.
    initial : DataContainer, optional
        Initial guess for the solution. If not provided, a zero-initialised container is used.
    data : DataContainer
        Measured data (right-hand side of the equation).
    alpha : float, optional
        Non-negative regularisation parameter. If zero, standard LSQR is used.

    Reference
    ---------
    https://web.stanford.edu/group/SOL/software/lsqr/
    """

    def __init__(self, initial=None, operator=None, data=None, alpha=0, **kwargs):
        """
        Initialise the LSQR algorithm.

        Parameters
        ----------
        initial : DataContainer, optional
            Initial guess for the solution.
        operator : Operator
            Linear operator representing the forward model.
        data : DataContainer
            Measured data.
        alpha : float, optional
            Regularisation parameter. Default is 0 (no regularisation).
        """


        
        super(LSQR, self).__init__(**kwargs)

        if initial is None and operator is not None:
            initial = operator.domain_geometry().allocate(0)
        self.regalpha = alpha 

        if initial is not None and operator is not None and data is not None:
            self.set_up(initial=initial, operator=operator, data=data)
        else:
            raise ValueError(' You must initialise LSQR with an `operator` and `data`')


    def set_up(self, initial, operator, data):
        """
        Set up the LSQR algorithm with the problem definition.

        Parameters
        ----------
        initial : DataContainer
            Initial guess for the solution.
        operator : Operator
            Linear operator representing the forward model.
        data : DataContainer
            Measured data.
        """
        log.info("%s setting up", self.__class__.__name__)
        self.x = initial.copy() #1 domain
        self.operator = operator

        # Initialise Golub-Kahan bidiagonalisation (GKB)
        
        #self.u = data - self.operator.direct(self.x)
        self.u = self.operator.direct(self.x) #1 range 
        self.u.sapyb(-1, data, 1, out=self.u)
        self.beta = self.u.norm()
        self.u /= self.beta
        
        self.v = self.operator.adjoint(self.u) #2 domain 
        self.alpha = self.v.norm()
        self.v /= self.alpha

        self.rhobar = self.alpha
        self.phibar = self.beta
        self.normr = self.beta
        self.regalphasq = self.regalpha**2

        self.d = self.v.copy() #3 domain 
        self.tmp_range = data.geometry.allocate(None) #2 range
        self.tmp_domain = self.x.geometry.allocate(None) #4 domain
        
        self.res2 = 0

        self.configured = True
        log.info("%s configured", self.__class__.__name__)


    def update(self):
        """Perform a single iteration of the LSQR algorithm."""
        # Update u in GKB
        self.operator.direct(self.v, out=self.tmp_range)
        self.tmp_range.sapyb(1.,  self.u,-self.alpha, out=self.u)
        self.beta = self.u.norm()
        self.u /= self.beta

        # Update v in GKB
        self.operator.adjoint(self.u, out=self.tmp_domain)
        self.v.sapyb(-self.beta, self.tmp_domain, 1., out=self.v)
        self.alpha = self.v.norm()
        self.v /= self.alpha

        # Eliminate diagonal from regularisation
        if self.regalphasq > 0:
            rhobar1 = math.sqrt(self.rhobar * self.rhobar + self.regalphasq)
            c1 = self.rhobar / rhobar1
            s1 = self.regalpha / rhobar1
            psi = s1 * self.phibar
            self.phibar = c1 * self.phibar
        else:
            rhobar1 = self.rhobar
            psi = 0

        # Eliminate lower bidiagonal part
        rho = math.sqrt(rhobar1 ** 2 + self.beta ** 2)
        c = rhobar1 / rho
        s = self.beta / rho
        theta = s * self.alpha
        self.rhobar = -c * self.alpha
        phi = c * self.phibar
        self.phibar = s * self.phibar

        # Update image x
        self.x.sapyb(1, self.d, phi/rho, out=self.x)

        # Update d
        self.d.sapyb(-theta/rho, self.v, 1, out=self.d)

        # Estimate residual norm 
        self.res2 += psi ** 2
        self.normr = math.sqrt(self.phibar ** 2 + self.res2)
        

    def update_objective(self):
        """
        Update the objective function value (residual norm squared).
        """

        if self.normr is numpy.nan:
            raise StopIteration()
        self.loss.append(self.normr**2)


