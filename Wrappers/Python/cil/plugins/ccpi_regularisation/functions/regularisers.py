# -*- coding: utf-8 -*-
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

try:
    from ccpi.filters import regularisers
    from ccpi.filters.cpu_regularisers import TV_ENERGY
except ImportError as ie:
    raise ImportError(ie , "\n\n", 
                      "This plugin requires the additional package ccpi-regularisation\n" +
                      "Please install it via conda as ccpi-regulariser from the ccpi channel\n"+
                      "Minimal version is 20.04")


from cil.framework import DataOrder
from cil.framework import DataContainer
from cil.optimisation.functions import Function
import numpy as np
import warnings
from numbers import Number

class RegulariserFunction(Function):
    def proximal(self, x, tau, out=None):
 
        r""" Generic proximal method for a RegulariserFunction

        .. math:: \mathrm{prox}_{\tau f}(x) := \argmin_{z} f(x) + \frac{1}{2}\|z - x \|^{2}
        
        Parameters
        ----------

        x : DataContainer
            Input of the proximal operator
        tau : Number
            Positive parameter of the proximal operator
        out : DataContainer
            Output :class:`Datacontainer` in which the result is placed.

        Note
        ----    
        
        If the :class:`ImageData` contains complex data, rather than the default `float32`, the regularisation
        is run independently on the real and imaginary part.

        """

        self.check_input(x)
        arr = x.as_array()
        if np.iscomplexobj(arr):
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

    r""" Total Variation regulariser

    .. math:: TV(u) = \alpha \|\nabla u\|_{2,1}

    Parameters
    ----------
    
    strong_convexity_constant : Number 
                              Positive parameter that allows Total variation regulariser to be strongly convex. Default = 0.

    Note
    ----

    By definition, Total variation is a convex function. However,
    adding a strongly convex term makes it a strongly convex function.
    Then, we say that `TV` is a :math:`\gamma>0` strongly convex function i.e., 

    .. math:: TV(u) = \alpha \|\nabla u\|_{2,1} + \frac{\gamma}{2}\|u\|^{2}

    """

    def __init__(self, strong_convexity_constant = 0):

        self.strong_convexity_constant = strong_convexity_constant

    def __call__(self,x):
        in_arr = np.asarray(x.as_array(), dtype=np.float32, order='C')
        EnergyValTV = TV_ENERGY(in_arr, in_arr, self.alpha, 2)
        if self.strong_convexity_constant>0:
            return 0.5*EnergyValTV[0] + (self.strong_convexity_constant/2)*x.squared_norm()
        else:
            return 0.5*EnergyValTV[0]

    def convex_conjugate(self,x):     
        return 0.0


class FGP_TV(TV_Base):

    r""" Fast Gradient Projection Total Variation (FGP_TV)

        The :class:`FGP_TV` computes the proximal operator of the Total variation regulariser

        .. math:: \mathrm{prox}_{\tau (\alpha TV)}(x) = \underset{z}{\mathrm{argmin}} \,\alpha\,\mathrm{TV}(z) + \frac{1}{2}\|z - x\|^{2} .
        
        The algorithm used for the proximal operator of TV is the Fast Gradient Projection algorithm 
        applied to the _dual problem_ of the above problem, see :cite:`BeckTeboulle_b`, :cite:`BeckTeboulle_a`.


        Parameters
        ----------

        alpha : :obj:`Number` (positive), default = 1.0 .
                Total variation regularisation parameter. 

        max_iteration : :obj:`int`. Default = 100 .
                Maximum number of iterations for the Fast Gradient Projection algorithm.

        isotropic : :obj:`boolean`. Default = True .
                    Isotropic or Anisotropic definition of the Total variation regulariser.

                    .. math:: |x|_{2} = \sqrt{x_{1}^{2} + x_{2}^{2}},\, (\mbox{isotropic})

                    .. math:: |x|_{1} = |x_{1}| + |x_{2}|\, (\mbox{anisotropic})

        nonnegativity : :obj:`boolean`. Default = True .
                        Non-negativity constraint for the solution of the FGP algorithm.

        tolerance : :obj:`float`, Default = 0 .
                    Stopping criterion for the FGP algorithm.
                    
                    .. math:: \|x^{k+1} - x^{k}\|_{2} < \mathrm{tolerance}

        device : :obj:`str`, Default = 'cpu' .
                FGP_TV algorithm runs on `cpu` or `gpu`.

        strong_convexity_constant : :obj:`float`, default = 0
                A strongly convex term weighted by the :code:`strong_convexity_constant` (:math:`\gamma`) parameter is added to the Total variation. 
                Now the :code:`TotalVariation` function is :math:`\gamma` - strongly convex and the proximal operator is

                .. math:: \underset{u}{\mathrm{argmin}} \frac{1}{2\tau}\|u - b\|^{2} + \mathrm{TV}(u) + \frac{\gamma}{2}\|u\|^{2} \Leftrightarrow

                .. math:: \underset{u}{\mathrm{argmin}} \frac{1}{2\frac{\tau}{1+\gamma\tau}}\|u - \frac{b}{1+\gamma\tau}\|^{2} + \mathrm{TV}(u) 


        Examples
        --------

        .. math:: \underset{u\qeq0}{\mathrm{argmin}} \frac{1}{2}\|u - b\|^{2} + \alpha TV(u)


        >>> G = alpha * FGP_TV(max_iteration=100, device='gpu')
        >>> sol = G.proximal(b)

        Note
        ----

        The :class:`FGP_TV` regularisation does not incorparate information on the :class:`ImageGeometry`, i.e., pixel/voxel size.
        Therefore a rescaled parameter should be used to match the same solution computed using :class:`~cil.optimisation.functions.TotalVariation`.

        >>> G1 = (alpha/ig.voxel_size_x) * FGP_TV(max_iteration=100, device='gpu')
        >>> G2 = alpha * TotalVariation(max_iteration=100, lower=0.)
        
        
        See Also
        --------
        :class:`~cil.optimisation.functions.TotalVariation`


        """


    def __init__(self, alpha=1, max_iteration=100, tolerance=0, isotropic=True, nonnegativity=True, device='cpu', strong_convexity_constant=0):
        
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
        self.device = device 

        super(FGP_TV, self).__init__(strong_convexity_constant=strong_convexity_constant)

    def _fista_on_dual_rof(self, in_arr, tau):
        
        r""" Implements the Fast Gradient Projection algorithm on the dual problem 
        of the Total Variation Denoising problem (ROF).

        """    

        res , info = regularisers.FGP_TV(\
              in_arr,\
              self.alpha * tau,\
              self.max_iteration,\
              self.tolerance,\
              self.methodTV,\
              self.nonnegativity,\
              self.device)

        return res, info

    def proximal_numpy(self, in_arr, tau):

        if self.strong_convexity_constant>0:

            strongly_convex_factor = (1 + tau * self.strong_convexity_constant)
            in_arr /= strongly_convex_factor
            tau /= strongly_convex_factor
        
        solution = self._fista_on_dual_rof(in_arr, tau)

        if self.strong_convexity_constant>0:
            in_arr *= strongly_convex_factor
            tau *= strongly_convex_factor

        return solution
                
    def __rmul__(self, scalar):
        '''Define the multiplication with a scalar
        
        this changes the regularisation parameter in the plugin'''
        if not isinstance (scalar, Number):
            raise NotImplemented
        else:
            self.alpha *= scalar
            return self
    def check_input(self, input):
        if len(input.shape) > 3:
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
        :param tolerance: minimum difference between previous iteration of the algorithm that determines the stop of the iteration earlier than max_iteration. If set to 0 only the max_iteration will be used as stop criterion.
        :type tolerance: float, default 0
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
        if len(input.shape) == 2:
            self.LipshitzConstant = 12
        elif len(input.shape) == 3:
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
        :param tolerance: minimum difference between previous iteration of the algorithm that determines the stop of the iteration earlier than max_iteration. If set to 0 only the max_iteration will be used as stop criterion.
        :type tolerance: float, default 0
        :param eta: smoothing constant to calculate gradient of the reference
        :type eta: number, default 0.01
        :param isotropic: Whether it uses L2 (isotropic) or L1 (anisotropic) norm
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
        if len(input.shape) > 3:
            raise ValueError('{} cannot work on more than 3D. Got {}'.format(self.__class__.__name__, input.geometry.length))

class TNV(RegulariserFunction):
    
    def __init__(self,alpha=1, max_iteration=100, tolerance=0):
        '''Creator of TNV Function

        :param alpha: regularisation parameter
        :type alpha: number, default 1
        :param max_iteration: max number of sub iterations. The algorithm will iterate up to this number of iteration or up to when the tolerance has been reached
        :type max_iteration: integer, default 100
        :param tolerance: minimum difference between previous iteration of the algorithm that determines the stop of the iteration earlier than max_iteration. If set to 0 only the max_iteration will be used as stop criterion.
        :type tolerance: float, default 0
        '''
        # set parameters
        self.alpha = alpha
        self.max_iteration = max_iteration
        self.tolerance = tolerance
        
    def __call__(self,x):
        warnings.warn("{}: the __call__ method is not implemented. Returning NaN.".format(self.__class__.__name__))
        return np.nan
    
    def proximal_numpy(self, in_arr, tau):
        # remove any dimension of size 1
        new_shape = [ i for i in input.shape if i!=1]
        in_arr.shape = tuple(new_shape)
            
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
        if isinstance(input, DataContainer):
            DataOrder.check_order_for_engine('cil', input.geometry)
            if ( input.geometry.channels == 1 ) or ( not input.geometry.length == 3) :
                raise ValueError('TNV requires 2D+channel data. Got {}'.format(input.geometry.dimension_labels))
        else:
            # if it is not a CIL DataContainer we assume that the data is passed in the correct order
            new_shape = [ i for i in input.shape if i!=1]
            if len(input.shape) != 3:
                raise ValueError('TNV requires 3D data (with channel as first axis). Got {}'.format(input.shape))
        

