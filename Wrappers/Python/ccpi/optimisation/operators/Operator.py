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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ccpi.optimisation.operators.ScaledOperator import ScaledOperator

class Operator(object):
    '''Operator that maps from a space X -> Y'''
    def __init__(self, **kwargs):
        self.__norm = None

    def is_linear(self):
        '''Returns if the operator is linear'''
        return False
    def direct(self,x, out=None):
        '''Returns the application of the Operator on x'''
        raise NotImplementedError
    def norm(self, **kwargs):
        '''Returns the norm of the Operator
        
        Calling norm triggers the calculation of the norm of the operator. Normally this
        is a computationally expensive task, therefore we store the result of norm into 
        a member of the class. If the calculation has already run, following calls to 
        norm just return the saved member. 
        It is possible to force recalculation by setting the optional force parameter. Notice that
        norm doesn't take notice of how many iterations or of the initialisation of the PowerMethod, 
        so in case you want to recalculate by setting a higher number of iterations or changing the
        starting point or both you need to set :code:`force=True`

        :param iterations: number of iterations to run
        :type iterations: int, optional, default = 25
        :param x_init: starting point for the iteration in the operator domain
        :type x_init: same type as domain, a subclass of :code:`DataContainer`, optional, default None
        :parameter force: forces the recalculation of the norm
        :type force: boolean, default :code:`False`
        '''
        if self.__norm is None or kwargs.get('force', False):
            self.__norm = self.calculate_norm(**kwargs)
        return self.__norm
    def calculate_norm(self, **kwargs):
        '''Calculates the norm of the Operator'''
        raise NotImplementedError
    def range_geometry(self):
        '''Returns the range of the Operator: Y space'''
        raise NotImplementedError
    def domain_geometry(self):
        '''Returns the domain of the Operator: X space'''
        raise NotImplementedError
    def __rmul__(self, scalar):
        '''Defines the multiplication by a scalar on the left

        returns a ScaledOperator'''
        return ScaledOperator(self, scalar)
