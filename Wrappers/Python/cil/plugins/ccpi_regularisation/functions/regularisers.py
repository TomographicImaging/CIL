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


from cil.framework import DataOrder
from cil.optimisation.functions import Function
import numpy as np
import warnings
from numbers import Number

class RegulariserFunction(Function):
    def proximal(self, x, tau, out=None):
        '''Generic proximal method for a RegulariserFunction
        
        :param x: image to be regularised
        :type x: an ImageData
        :param tau: 
        :type tau: Number
        :param out: a placeholder for the result
        :type out: same as x: ImageData'''

        self.check_input(x)
        arr = x.as_array()
        if arr.dtype in [np.complex, np.complex64]:
            # do real and imag part indep
            in_arr = np.asarray(arr.real, dtype=np.float32, order='C')
            res, info = self.proximal_numpy(in_arr, tau)
            arr.real = res[:]
            in_arr = np.asarray(arr.imag, dtype=np.float32, order='C')
            res, info = self.proximal_numpy(in_arr, tau)
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
            res, info = self.proximal_numpy(arr, tau)
            self.info = info
            if out is not None:
                out.fill(res)
            else:
                out = x.copy()
                out.fill(res)
                return out
    def proximal_numpy(self, xarr, tau):
        raise NotImplementedError('Please implement proximal_numpy')

    def check_input(self, input):
        pass

class TV_Base(RegulariserFunction):
    def __call__(self,x):
        in_arr = np.asarray(x.as_array(), dtype=np.float32, order='C')
        EnergyValTV = TV_ENERGY(in_arr, in_arr, self.alpha, 2)
        return 0.5*EnergyValTV[0]

    def convex_conjugate(self,x):     
        return 0.0


class FGP_TV(TV_Base):
    def __init__(self, alpha=1, max_iteration=100, tolerance=0, isotropic=True, nonnegativity=True, device='cpu'):
        '''Creator of FGP_TV Function


        :param alpha: regularisation parameter
        :type alpha: number, default 1
        :param isotropic: Whether it uses L2 (isotropic) or L1 (unisotropic) norm
        :type isotropic: boolean, default True, can range between 1 and 2
        :param nonnegativity: Whether to add the non-negativity constraint
        :type nonnegativity: boolean, default True
        :param max_iteration: max number of sub iterations. The algorithm will iterate up to this number of iteration or up to when the tolerance has been reached
        :type max_iteration: integer, default 100
        :param tolerance: minimum difference between previous iteration of the algorithm that determines the stop of the iteration earlier than num_iter
        :type tolerance: float, default 1e-6
        :param device: determines if the code runs on CPU or GPU
        :type device: string, default 'cpu', can be 'gpu' if GPU is installed
        '''
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

    def proximal_numpy(self, in_arr, tau):
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
    def check_input(self, input):
        if input.geometry.length > 3:
            raise ValueError('{} cannot work on more than 3D. Got {}'.format(self.__class__.__name__, input.geometry.length))
        
class TGV(RegulariserFunction):

    def __init__(self, alpha=1, gamma=1, max_iteration=100, tolerance=0, device='cpu' , **kwargs):
        '''Creator of Total Generalised Variation Function 

        :param alpha: regularisation parameter
        :type alpha: number, default 1
        :param gamma: ratio of TGV terms
        :type gamma: number, default 1, can range between 1 and 2
        :param max_iteration: max number of sub iterations. The algorithm will iterate up to this number of iteration or up to when the tolerance has been reached
        :type max_iteration: integer, default 100
        :param tolerance: minimum difference between previous iteration of the algorithm that determines the stop of the iteration earlier than num_iter
        :type tolerance: float, default 1e-6
        :param device: determines if the code runs on CPU or GPU
        :type device: string, default 'cpu', can be 'gpu' if GPU is installed
        
        '''
        
        self.alpha = alpha
        self.gamma = gamma
        self.max_iteration = max_iteration
        self.tolerance = tolerance
        self.device = device

        if kwargs.get('iter_TGV', None) is not None:
            # raise ValueError('iter_TGV parameter has been superseded by num_iter. Use that instead.')
            self.num_iter = kwargs.get('iter_TGV')
        
    def __call__(self,x):
        warnings.warn("{}: the __call__ method is not implemented. Returning NaN.".format(self.__class__.__name__))
        return np.nan
    @property
    def gamma(self):
        return self.__gamma
    @gamma.setter
    def gamma(self, value):
        if value <= 2 and value >= 1:
            self.__gamma = value
    @property
    def alpha2(self):
        return self.alpha1 * self.gamma
    @property
    def alpha1(self):
        return 1.
    
    def proximal_numpy(self, in_arr, tau):
        res , info = regularisers.TGV(in_arr,
              self.alpha * tau,
              self.alpha1,
              self.alpha2,
              self.max_iteration,
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

        # f = TGV()
        # f = alpha * f

    def check_input(self, input):
        if len(input.dimension_labels) == 2:
            self.LipshitzConstant = 12
        elif len(input.dimension_labels) == 3:
            self.LipshitzConstant = 16 # Vaggelis to confirm
        else:
            raise ValueError('{} cannot work on more than 3D. Got {}'.format(self.__class__.__name__, input.geometry.length))
        

class FGP_dTV(RegulariserFunction):
    '''Creator of FGP_dTV Function

        :param reference: reference image
        :type reference: ImageData
        :param alpha: regularisation parameter
        :type alpha: number, default 1
        :param max_iteration: max number of sub iterations. The algorithm will iterate up to this number of iteration or up to when the tolerance has been reached
        :type max_iteration: integer, default 100
        :param tolerance: minimum difference between previous iteration of the algorithm that determines the stop of the iteration earlier than num_iter
        :type tolerance: float, default 1e-6
        :param eta: smoothing constant to calculate gradient of the reference
        :type eta: number, default 0.01
        :param isotropic: Whether it uses L2 (isotropic) or L1 (unisotropic) norm
        :type isotropic: boolean, default True, can range between 1 and 2
        :param nonnegativity: Whether to add the non-negativity constraint
        :type nonnegativity: boolean, default True
        :param device: determines if the code runs on CPU or GPU
        :type device: string, default 'cpu', can be 'gpu' if GPU is installed
        '''
    def __init__(self, reference, alpha=1, max_iteration=100,
                 tolerance=0, eta=0.01, isotropic=True, nonnegativity=True, device='cpu'):

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

    def proximal_numpy(self, in_arr, tau):
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

    def check_input(self, input):
        if input.geometry.length > 3:
            raise ValueError('{} cannot work on more than 3D. Got {}'.format(self.__class__.__name__, input.geometry.length))

class TNV(RegulariserFunction):
    
    def __init__(self,alpha=1, max_iteration=100, tolerance=0):
        '''Creator of TNV Function

        :param alpha: regularisation parameter
        :type alpha: number, default 1
        :param max_iteration: max number of sub iterations. The algorithm will iterate up to this number of iteration or up to when the tolerance has been reached
        :type max_iteration: integer, default 100
        :param tolerance: minimum difference between previous iteration of the algorithm that determines the stop of the iteration earlier than num_iter
        :type tolerance: float, default 1e-6
        '''
        # set parameters
        self.alpha = alpha
        self.max_iteration = max_iteration
        self.tolerance = tolerance
        
    def __call__(self,x):
        warnings.warn("{}: the __call__ method is not implemented. Returning NaN.".format(self.__class__.__name__))
        return np.nan
    
    def proximal_numpy(self, in_arr, tau):
        if in_arr.ndim != 3:
            # https://github.com/vais-ral/CCPi-Regularisation-Toolkit/blob/413c6001003c6f1272aeb43152654baaf0c8a423/src/Python/src/cpu_regularisers.pyx#L584-L588
            raise ValueError('Only 3D data is supported. Passed data has {} dimensions'.format(in_arr.ndim))
        res = regularisers.TNV(in_arr, 
              self.alpha * tau,
              self.max_iteration,
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

    def check_input(self, input):
        '''TNV requires 2D+channel data with the first dimension as the channel dimension'''
        DataOrder.check_order_for_engine('cil', input.geometry)
        if ( input.geometry.channels == 1 ) or ( not input.geometry.length == 3) :
            raise ValueError('TNV requires 2D+channel data. Got {}'.format(input.geometry.dimension_labels))
        