# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library developed by
#   Visual Analytics and Imaging System Group of the Science Technology
#   Facilities Council, STFC

#   Copyright 2018-2019 Jakob Jorgensen, Daniil Kazantsev and Edoardo Pasca

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import warnings

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
        return self.proximal(x, out=None)

