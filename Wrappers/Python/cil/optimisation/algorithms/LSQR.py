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
# CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt
# Maike Meier and Mariam Demir, SCD STFC

from cil.optimisation.algorithms import Algorithm
import numpy
import logging
import warnings 
import math

log = logging.getLogger(__name__)

class LSQR(Algorithm):

    r''' Least Squares QR (LSQR) algorithm
    
    The Least Squares QR (LSQR) algorithm is commonly used for solving large systems of linear equations, due to its fast convergence.

    Problem:

    .. math::

      \min_x || A x - b ||^2_2
      
    An optional regularisation parameter alpha can be included to instead solve the Tikhonov regularised problem

    .. math::

      \min_x { || A x - b ||^2_2 + alpha^2 || x ||_2^2 }

        
    Parameters
    ------------
    operator : Operator
        Linear operator for the inverse problem
    initial : (optional) DataContainer in the domain of the operator, default is a DataContainer filled with zeros. 
        Initial guess 
    data : DataContainer in the range of the operator 
        Acquired data to reconstruct
    alpha : (optional) non-negative float, default 0
        Regularisation parameter that includes Tikhonov regularisation in the objective, default is zero. In case of zero the algorithm is standard LSQR.


    Reference
    ---------
    https://web.stanford.edu/group/SOL/software/lsqr/
    '''
    def __init__(self, initial=None, operator=None, data=None, alpha=None, **kwargs):
        '''initialisation of the algorithm
        '''
        #We are deprecating tolerance 
        self.tolerance=kwargs.pop("tolerance", None)
        if self.tolerance is not None:
            warnings.warn( stacklevel=2, category=DeprecationWarning, message="Passing tolerance directly to CGLS is being deprecated. Instead we recommend using the callback functionality: https://tomographicimaging.github.io/CIL/nightly/optimisation/#callbacks and in particular the CGLSEarlyStopping callback replicated the old behaviour")
        else:
            self.tolerance = 0
        
        super(LSQR, self).__init__(**kwargs)

        if initial is None and operator is not None:
            initial = operator.domain_geometry().allocate(0)
        if alpha is None:
            self.regalpha = 0
        else:
            self.regalpha = alpha 

        if initial is not None and operator is not None and data is not None:
            self.set_up(initial=initial, operator=operator, data=data)


    def set_up(self, initial, operator, data):
        r'''Initialisation of the algorithm
        Parameters
        ------------
        operator : Operator
            Linear operator for the inverse problem
        initial : (optional) DataContainer in the domain of the operator, default is a DataContainer filled with zeros. 
            Initial guess 
        data : DataContainer in the range of the operator 
            Acquired data to reconstruct

        '''
        
        log.info("%s setting up", self.__class__.__name__)
        self.x = initial #1 domain
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
        self.tmp_range = data.copy() #2 range
        self.tmp_domain = self.x.copy() #4 domain
        
        self.res2 = 0

        self.configured = True
        log.info("%s configured", self.__class__.__name__)


    def update(self):
        '''single iteration'''

        # Update u in GKB
        self.operator.direct(self.v, out=self.tmp_range)
        self.tmp_range.sapyb(1.,  self.u,-self.alpha, out=self.u)
        self.beta = self.u.norm()
        self.u /= self.beta
        print(self.beta)

        # Update v in GKB
        self.operator.adjoint(self.u, out=self.tmp_domain)
        self.v.sapyb(-self.beta, self.tmp_domain, 1., out=self.v)
        self.alpha = self.v.norm()
        self.v /= self.alpha
        print(self.alpha)

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
        if self.normr is numpy.nan:
            raise StopIteration()
        self.loss.append(self.normr**2)


