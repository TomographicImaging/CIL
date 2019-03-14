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
    alpha is scaling parameter of the function. 
    '''
    def __init__(self):
        self.L = None
    def __call__(self,x, out=None):
        raise NotImplementedError
    def call_adjoint(self, x, out=None):
        raise NotImplementedError
    def convex_conjugate(self, x, out=None):
        raise NotImplementedError
    def proximal_conjugate(self, x, tau, out = None):
        raise NotImplementedError
    def grad(self, x):
        warnings.warn('''This method will disappear in following 
        versions of the CIL. Use gradient instead''', DeprecationWarning)
        return self.gradient(x, out=None)
    def prox(self, x, tau):
        warnings.warn('''This method will disappear in following 
        versions of the CIL. Use proximal instead''', DeprecationWarning)
        return self.proximal(x, out=None)
    def gradient(self, x, out=None):
        raise NotImplementedError
    def proximal(self, x, tau, out=None):
        raise NotImplementedError