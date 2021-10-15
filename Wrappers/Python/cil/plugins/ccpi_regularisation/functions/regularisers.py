# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library (CIL) developed by CCPi 
#   (Collaborative Computational Project in Tomographic Imaging), with 
#   substantial contributions by UKRI-STFC and University of Manchester.

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

try:
    from ccpi.filters import regularisers
    from ccpi.filters.cpu_regularisers import TV_ENERGY
except ImportError as ie:
    raise ImportError(ie , "\n\n", 
                      "This plugin requires the additional package ccpi-regularisation\n" +
                      "Please install it via conda as ccpi-regulariser from the ccpi channel\n"+
                      "Minimal version is 20.04")


from cil.framework import DataContainer
from cil.optimisation.functions import Function
import numpy as np
import warnings
from numbers import Number

class RegulariserFunction(Function):
    def proximal(self, x, tau, out=None):
        arr = x.as_array()
        if arr.dtype in [np.complex, np.complex64]:
            # do real and imag part indep
            in_arr = np.asarray(arr.real, dtype=np.float32, order='C')
            res, info = self.proximal_numpy(in_arr, tau, out)
            arr.real = res[:]
            in_arr = np.asarray(arr.imag, dtype=np.float32, order='C')
            res, info = self.proximal_numpy(in_arr, tau, out)
            arr.imag = res[:]
            self.info = info
            if out is not None:
                out.fill(arr)
            else:
                out = x.copy()
                out.fill(arr)
                return out
        else:
            arr = np.asarray(x.as_array(), dtype=np.float32, order='C')
            res, info = self.proximal_numpy(arr, tau, out)
            self.info = info
            if out is not None:
                out.fill(res)
            else:
                out = x.copy()
                out.fill(res)
                return out
    def proximal_numpy(self, xarr, tau, out=None):
        raise NotImplementedError('Please implement proximal_numpy')

class TV_Base(RegulariserFunction):
    def __call__(self,x):
        in_arr = np.asarray(x.as_array(), dtype=np.float32, order='C')
        EnergyValTV = TV_ENERGY(in_arr, in_arr, self.alpha, 2)
        return 0.5*EnergyValTV[0]

    def convex_conjugate(self,x):     
        return 0.0


class FGP_TV(TV_Base):
    def __init__(self, alpha=1, max_iteration=100, tolerance=1e-6, isotropic=True, nonnegativity=True, printing=False, device='cpu'):

        if isotropic == True:
            self.methodTV = 0
        else:
            self.methodTV = 1

        if nonnegativity == True:
            self.nonnegativity = 1
        else:
            self.nonnegativity = 0

        self.alpha = alpha
        self.max_iteration = max_iteration
        self.tolerance = tolerance
        self.nonnegativity = nonnegativity
        self.device = device # string for 'cpu' or 'gpu'

    def proximal_numpy(self, in_arr, tau, out = None):
        res , info = regularisers.FGP_TV(\
              in_arr,\
              self.alpha * tau,\
              self.max_iteration,\
              self.tolerance,\
              self.methodTV,\
              self.nonnegativity,\
              self.device)
        return res, info
    
    def __rmul__(self, scalar):
        '''Define the multiplication with a scalar
        
        this changes the regularisation parameter in the plugin'''
        if not isinstance (scalar, Number):
            raise NotImplemented
        else:
            self.alpha *= scalar
            return self
        
class TGV(RegulariserFunction):

    def __init__(self, alpha=1, alpha1=1, alpha2=1, iter_TGV=100, LipshitzConstant=12, tolerance=1e-6, device='cpu' ):
        # Default values
        # https://github.com/vais-ral/CCPi-Regularisation-Toolkit/blob/413c6001003c6f1272aeb43152654baaf0c8a423/src/Core/regularisers_CPU/TGV_core.c#L25-L32
        self.alpha = alpha
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.iter_TGV = iter_TGV
        self.LipshitzConstant = LipshitzConstant
        self.tolerance = tolerance
        self.device = device
        
    def __call__(self,x):
        warnings.warn("{}: the __call__ method is not implemented. Returning NaN.".format(self.__class__.__name__))
        return np.nan
    
    def proximal_numpy(self, in_arr, tau, out = None):
        res , info = regularisers.TGV(in_arr,
              self.alpha * tau,
              self.alpha1,
              self.alpha2,
              self.iter_TGV,
              self.LipshitzConstant,
              self.tolerance,
              self.device)
                
        # info: return number of iteration and reached tolerance
        # https://github.com/vais-ral/CCPi-Regularisation-Toolkit/blob/master/src/Core/regularisers_CPU/TGV_core.c#L168
        # Stopping Criteria  || u^k - u^(k-1) ||_{2} / || u^{k} ||_{2}    
        return res, info
    
    def convex_conjugate(self, x):
        warnings.warn("{}: the convex_conjugate method is not implemented. Returning NaN.".format(self.__class__.__name__))
        return np.nan
        
    def __rmul__(self, scalar):
        '''Define the multiplication with a scalar
        
        this changes the regularisation parameter in the plugin'''
        if not isinstance (scalar, Number):
            raise NotImplemented
        else:
            self.alpha *= scalar
            return self


class FGP_dTV(RegulariserFunction):
    def __init__(self, reference, alpha=1, max_iteration=100,
                 tolerance=1e-6, eta=0.01, isotropic=True, nonnegativity=True, device='cpu'):

        if isotropic == True:
            self.methodTV = 0
        else:
            self.methodTV = 1

        if nonnegativity == True:
            self.nonnegativity = 1
        else:
            self.nonnegativity = 0

        self.alpha = alpha
        self.max_iteration = max_iteration
        self.tolerance = tolerance
        self.device = device # string for 'cpu' or 'gpu'
        self.reference = np.asarray(reference.as_array(), dtype=np.float32)
        self.eta = eta
        
    def __call__(self,x):
        warnings.warn("{}: the __call__ method is not implemented. Returning NaN.".format(self.__class__.__name__))
        return np.nan

    def proximal_numpy(self, in_arr, tau, out = None):
        res , info = regularisers.FGP_dTV(\
                in_arr,\
                self.reference,\
                self.alpha * tau,\
                self.max_iteration,\
                self.tolerance,\
                self.eta,\
                self.methodTV,\
                self.nonnegativity,\
                self.device)
        return res, info

    def convex_conjugate(self, x):
        warnings.warn("{}: the convex_conjugate method is not implemented. Returning NaN.".format(self.__class__.__name__))
        return np.nan
    
    def __rmul__(self, scalar):
        '''Define the multiplication with a scalar
        
        this changes the regularisation parameter in the plugin'''
        if not isinstance (scalar, Number):
            raise NotImplemented
        else:
            self.alpha *= scalar
            return self

class TNV(RegulariserFunction):
    
    def __init__(self,alpha=1, iterationsTNV=100, tolerance=1e-6):
        
        # set parameters
        self.alpha = alpha
        self.iterationsTNV = iterationsTNV
        self.tolerance = tolerance
        
    def __call__(self,x):
        warnings.warn("{}: the __call__ method is not implemented. Returning NaN.".format(self.__class__.__name__))
        return np.nan
    
    def proximal_numpy(self, in_arr, tau, out = None):
        res = regularisers.TNV(in_arr, 
              self.alpha * tau,
              self.iterationsTNV,
              self.tolerance)
        return res, []

    def convex_conjugate(self, x):
        warnings.warn("{}: the convex_conjugate method is not implemented. Returning NaN.".format(self.__class__.__name__))
        return np.nan

    def __rmul__(self, scalar):
        '''Define the multiplication with a scalar
        
        this changes the regularisation parameter in the plugin'''
        if not isinstance (scalar, Number):
            raise NotImplemented
        else:
            self.alpha *= scalar
            return self
