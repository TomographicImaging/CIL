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
        If False, the convex conjugate is not implemented.
    precision : str, default 'double'
        Precision of the calculation, 'single' or 'double'
        
   


    Reference
    ---------
    For further details see https://doi.org/10.48550/arXiv.2410.22161
    
    '''

    def __init__(self, function: Function, assume_lower_semi: bool=False, precision: str='double'):
        self._function = function
        self._lower_semi = assume_lower_semi
        
        if precision == 'single':
            self.real_dtype = np.float32
            self.complex_dtype = np.complex64
        elif precision == 'double':
            self.real_dtype = np.float64
            self.complex_dtype = np.complex128
        else:
            raise ValueError('Precision must be `single` or `double`')
        
        super().__init__(L=function.L)

    def __call__(self, x: DataContainer):
        call_abs = self._take_abs_input(self._function.__call__)
        return call_abs(self._function, x)

    def proximal(self, x: DataContainer, tau: float, out: Optional[DataContainer]=None):
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
        prox_abs = self._abs_and_project(self._function.proximal)
        return prox_abs(self._function, x, tau=tau, out=out)

    def convex_conjugate(self, x: DataContainer):
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
            conv_abs = self._take_abs_input(self._function.convex_conjugate)
            return conv_abs(self._function, x)
        else:
            raise NotImplementedError(
                'Convex conjugate not available for this function. If you are sure your function is lower semi-continuous, convex, non-decreasing and finite at the origin, set `assume_lower_semi=True`')

    def _take_abs_input(self, func: Function):
        '''Decorator for function to act on abs of input of a method'''
 
        def _take_abs_decorator(self2, x, *args, **kwargs):
            rgeo = x.geometry.copy()
            rgeo.dtype = self.real_dtype
            r = rgeo.allocate(0)
            r.fill(np.abs(x.array).astype(self.real_dtype))
            # func(self, r, *args, **kwargs) for the abstract class implementation
            fval = func(r, *args, **kwargs)
            return fval
        return _take_abs_decorator

    def _abs_and_project(self, func: Function):
        '''Decorator for function to act on abs of input, 
        with return being projected to the angle of the input.
        Requires function return to have the same shape as input,
        such as prox.'''
        
            
        def _abs_project_decorator(self2, x, *args, **kwargs):
            rgeo = x.geometry.copy()
            rgeo.dtype = self.real_dtype
            r = rgeo.allocate(None)
            r.fill( np.abs(x.array).astype(self.real_dtype))
            Phi = np.exp((1j*np.angle(x.array)))
            out = kwargs.pop('out', None)
            
            # func(self, r, *args, **kwargs) for the abstract class implementation
            fvals = func(r, *args, **kwargs)

            # Douglas-Rachford splitting to find solution in positive orthant
            if np.any(fvals.array < 0):
                log.info('AbsFunctions: projection to +ve orthant triggered')
                cts = 0
                y = r.copy()
                while np.any(fvals.array < 0):
                    tmp = fvals.array - 0.5*y.array + 0.5*r.array
                    tmp[tmp < 0] = 0.
                    y.array += tmp - fvals.array
                    fvals = func(y, *args, **kwargs)
                    cts += 1
                    if cts > 10:
                        fvals.array[fvals.array < 0] = 0.
                        break

            if out is not None:
                out.array = fvals.array.astype(self.complex_dtype)*Phi
                
            else:
                out = x.geometry.allocate(None)
                out.array = fvals.array.astype(self.complex_dtype)*Phi
            return out
        return _abs_project_decorator

    def gradient(self, x):
        '''Gradient of the function at x is not defined for this function.
        '''
        raise NotImplementedError('Gradient not available for this function')
