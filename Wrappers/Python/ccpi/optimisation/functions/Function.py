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

import warnings
from ccpi.optimisation.functions.ScaledFunction import ScaledFunction

class Function(object):
    '''Abstract class representing a function
    
    Members:
    L is the Lipschitz constant of the gradient of the Function 
    '''
    def __init__(self):
        self.L = None

    def __call__(self,x, out=None):
        '''Evaluates the function at x '''
        raise NotImplementedError

    def gradient(self, x, out=None):
        '''Returns the gradient of the function at x, if the function is differentiable'''
        raise NotImplementedError

    def proximal(self, x, tau, out=None):
        '''This returns the proximal operator for the function at x, tau'''
        raise NotImplementedError

    def convex_conjugate(self, x, out=None):
        '''This evaluates the convex conjugate of the function at x'''
        raise NotImplementedError

    def proximal_conjugate(self, x, tau, out = None):
        '''This returns the proximal operator for the convex conjugate of the function at x, tau'''
        raise NotImplementedError

    def grad(self, x):
        '''Alias of gradient(x,None)'''
        warnings.warn('''This method will disappear in following 
        versions of the CIL. Use gradient instead''', DeprecationWarning)
        return self.gradient(x, out=None)

    def prox(self, x, tau):
        '''Alias of proximal(x, tau, None)'''
        warnings.warn('''This method will disappear in following 
        versions of the CIL. Use proximal instead''', DeprecationWarning)
        return self.proximal(x, tau, out=None)

    def __rmul__(self, scalar):
        '''Defines the multiplication by a scalar on the left

        returns a ScaledFunction'''
        return ScaledFunction(self, scalar)

