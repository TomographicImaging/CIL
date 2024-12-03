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
#
# 

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
      
      
    Parameters
    ------------
    operator : Operator
        Linear operator for the inverse problem
    initial : (optional) DataContainer in the domain of the operator, default is a DataContainer filled with zeros. 
        Initial guess 
    data : DataContainer in the range of the operator 
        Acquired data to reconstruct

    Reference
    ---------
    https://web.stanford.edu/group/SOL/software/lsqr/
    '''
    def __init__(self, initial=None, operator=None, data=None, **kwargs):
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
        self.x = initial.copy()
        self.operator = operator

        self.u = data - self.operator.direct(self.x)
        self.beta = self.u.norm()
        self.u = self.u/self.beta
        
        self.v = self.operator.adjoint(self.u)
        self.alpha = self.v.norm()
        self.v = self.v/self.alpha

        self.c = 1
        self.s = 0
        self.phibar = self.beta
        self.normr = self.beta
        
        self.d = operator.domain_geometry().allocate(0)

        self.configured = True
        log.info("%s configured", self.__class__.__name__)


    def update(self):
        '''single iteration'''

        # update u
        self.u = self.operator.direct(self.v) - self.alpha * self.u
        self.beta = self.u.norm()
        self.u = self.u/self.beta

        # update scalars
        theta = -self.s * self.alpha
        rhobar = self.c * self.alpha
        rho = math.sqrt((rhobar**2 + self.beta**2))

        self.c = rhobar/rho
        self.s = (-1*self.beta)/rho
        phi = self.c*self.phibar
        self.phibar = self.s*self.phibar

        # update d
        self.d.sapyb(-theta, self.v, 1, out=self.d)
        self.d /= rho

        #update image x
        self.x.sapyb(1, self.d, phi, out=self.x)

        # estimate residual norm
        self.normr = abs(self.s) * self.normr

        # update v
        self.v = self.operator.adjoint(self.u) - self.beta * self.v
        self.alpha = self.v.norm()
        self.v = self.v/self.alpha


    def update_objective(self):
        if self.normr is numpy.nan:
            raise StopIteration()
        self.loss.append(self.normr)

    def should_stop(self): # TODO: Deprecated, remove when CGLS tolerance is removed
        return self.flag() or super().should_stop()

    def flag(self): # TODO: Deprecated, remove when CGLS tolerance is removed
        flag = False

        if flag:
            self.update_objective()
            print('Tolerance is reached: {}'.format(self.tolerance))

        return flag