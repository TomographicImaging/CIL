#  Copyright 2024 United Kingdom Research and Innovation
#  Copyright 2024 The University of Manchester
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
# This work has been supported by the Royal Academy of Engineering and the
# Office of the Chief Science Adviser for National Security under the UK
# Intelligence Community Postdoctoral Research Fellowship programme.
#
# Francis M Watson, University of Manchester 2024
#


import numpy as np
from cil.optimisation.functions import Function
from cil.framework import DataContainer
from typing import Optional
import warnings
import logging

log = logging.getLogger(__name__)

class FunctionOfAbs(Function):
    r'''A function which acts on the absolute value of the complex-valued input, 

    .. math::  G(z) = H(abs(z))

    This function is initialised with another CIL function, :math:`H` in the above formula. When this function is called, first the absolute value of the input is taken, and then the input is passed to the provided function.
    
    Included in this class is the proximal map for FunctionOfAbs.  From this, the proximal conjugate is also available from the parent CIL Function class, which is valid for this function. 
    In the case that :math:`H` is lower semi-continuous, convex, non-decreasing and finite at the origin, (and thus `assume_lower_semi` is set to `True` in the `init`) the convex conjugate is also defined.  
    The gradient is not defined for this function.
    


    Parameters
    ----------
    function : Function
        Function acting on a real input, :math:`H` in the above formula.
    assume_lower_semi : bool, default False
        If True, assume that the function is lower semi-continuous, convex, non-decreasing and finite at the origin.
        This allows the convex conjugate to be calculated as the monotone conjugate, which is less than or equal to the convex conjugate.
        If False, the convex conjugate returned as 0. This is to ensure compatibility with Algorithms such as PDHG.

    Reference
    ---------
    For further details see https://doi.org/10.48550/arXiv.2410.22161
    
    '''

    def __init__(self, function: Function, assume_lower_semi: bool=False):
        self._function = function
        self._lower_semi = assume_lower_semi
                
        super().__init__(L=function.L)

    def __call__(self, x: DataContainer) -> float:
        call_abs = _take_abs_input(self._function.__call__)
        return call_abs(self._function, x)

    def proximal(self, x: DataContainer, tau: float, out: Optional[DataContainer]=None) -> DataContainer:
        r'''Returns the proximal map of function :math:`\tau G`  evaluated at x

        .. math:: \text{prox}_{\tau G}(x) = \underset{z}{\text{argmin}} \frac{1}{2}\|z - x\|^{2} + \tau G(z)

        This is accomplished by calculating a bounded proximal map and making a change of phase,
        :math:`prox_G(z) = prox^+_H(r) \circ \Phi` where  :math:`z = r \circ \Phi`, :math:`r = abs(z)`, :math:`\Phi = \exp(i angl(z))`,
        and :math:`\circ` is element-wise product.  Also define :math:`prox^+` to be the proximal map of :math:`H`  in which the minimisation carried out over the positive orthant.


        Parameters
        ----------
        x : DataContainer
            The input to the function
        tau: scalar
            The scalar multiplying the function in the proximal map
        out: return DataContainer, if None a new DataContainer is returned, default None.
            DataContainer to store the result of the proximal map
        

        Returns
        -------
        DataContainer, the proximal map of the function at x with scalar :math:`\tau`.

        '''
        prox_abs = _abs_and_project(self._function.proximal)
        return prox_abs(self._function, x, tau=tau, out=out)

    def convex_conjugate(self, x: DataContainer) -> float:
        r'''
        Evaluation of the function G* at x, where G* is the convex conjugate of function G,

        .. math:: G^{*}(x^{*}) = \underset{x}{\sup} \langle x^{*}, x \rangle - G(x)

        If :math:`H` is lower semi-continuous, convex, non-decreasing 
        finite at the origin, then :math:`G^*(z*) = H^+(|z*|)`, where the monotone conjugate :math:`g^+` is

        .. math:: H^+(z^*) =sup {(z, z^*) - H(z) : z >= O}

        The monotone conjugate will therefore be less than or equal to the convex conjugate, 
        since it is taken over a smaller set.  It is not available directly, but may coincide with
        the convex conjugate, which is therefore the best estimate we have.  This is only valid for
        real x. In other cases, a general convex conjugate is not available or defined.      


        For reference see:  Convex Analysis, R. Tyrrell Rocakfellar, pp110-111.
        
        
        Parameters
        ----------
        x : DataContainer
            The input to the function

        Returns
        -------
        float:

        '''

        if self._lower_semi:
            conv_abs = _take_abs_input(self._function.convex_conjugate)
            return conv_abs(self._function, x)
        else:
            warnings.warn('Convex conjugate is not defined for this function, returning 0 for compatibility with optimisation algorithms')
            return 0.0

    def gradient(self, x):
        '''Gradient of the function at x is not defined for this function.
        '''
        raise NotImplementedError('Gradient not available for this function')




def _take_abs_input(func):
    '''Decorator for function to act on abs of input of a method'''

    def _take_abs_decorator(self, x: DataContainer, *args, **kwargs):
        real_dtype, _ = _get_real_complex_dtype(x)
        
        rgeo = x.geometry.copy()
        rgeo.dtype = real_dtype
        r = rgeo.allocate(0)
        r.fill(np.abs(x.as_array()).astype(real_dtype))
        fval = func(r, *args, **kwargs)
        return fval
    return _take_abs_decorator


def _abs_and_project(func):
    '''Decorator for function to act on abs of input, 
    with return being projected to the angle of the input.
    Requires function return to have the same shape as input,
    such as prox.'''
    
        
    def _abs_project_decorator(self, x: DataContainer, *args, **kwargs):
        
        real_dtype, complex_dtype = _get_real_complex_dtype(x)
            
                    
        rgeo = x.geometry.copy()
        rgeo.dtype = real_dtype
        r = rgeo.allocate(None)
        r.fill(np.abs(x.as_array()).astype(real_dtype))
        Phi = np.exp((1j*np.angle(x.array)))
        out = kwargs.pop('out', None)
        
        fvals = func(r, *args, **kwargs)
        fvals_np = fvals.as_array()
        
        # Douglas-Rachford splitting to find solution in positive orthant
        if np.any(fvals_np < 0):
            log.info('AbsFunctions: projection to +ve orthant triggered')
            cts = 0
            y = r.copy()
            fvals_np = fvals.as_array()
            while np.any(fvals_np < 0):
                tmp = fvals_np  - 0.5*y.as_array() + 0.5*r.as_array()
                tmp[tmp < 0] = 0.
                y += DataContainter(tmp, y.geometry) - fvals
                fvals = func(y, *args, **kwargs)
                cts += 1
                if cts > 10:
                    fvals_np = fvals.as_array()
                    fvals_np[fvals_np < 0] = 0.
                    break

        if out is None: 
            out_geom = x.geometry.copy()
            out = out_geom.allocate(None)
        if np.isreal(x.as_array()).all():
            out.fill( np.real(fvals_np.astype(complex_dtype)*Phi))
        else:
            out.fill( fvals_np.astype(complex_dtype)*Phi)
        return out
    return _abs_project_decorator


def _get_real_complex_dtype(x: DataContainer):
    '''An internal function to find the type of x and set the corresponding real and complex data types '''
    
    x_dtype = x.as_array().dtype
    if np.issubdtype(x_dtype, np.complexfloating):
        complex_dtype = x_dtype
        complex_example = np.array([1 + 1j], dtype=x_dtype)
        real_dtype = np.real(complex_example).dtype
    else: 
        real_dtype = x_dtype
        complex_example = 1j*np.array([1], dtype=x_dtype)
        complex_dtype = complex_example.dtype
    return real_dtype, complex_dtype            
            