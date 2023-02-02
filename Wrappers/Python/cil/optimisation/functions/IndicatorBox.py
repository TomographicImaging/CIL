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

from cil.optimisation.functions import Function
import numpy as np
import numba


class IndicatorBox(Function):
    
    
    r'''Indicator function for box constraint
            
      .. math:: 
         
         f(x) = \mathbb{I}_{[a, b]} = \begin{cases}  
                                            0, \text{ if } x \in [a, b] \\
                                            \infty, \text{otherwise}
                                     \end{cases}
    
    '''
    
    def __init__(self,lower=-np.inf,upper=np.inf):
        '''creator

        Parameters:
        -----------
        lower : float, DataContainer or numpy array, default ``-np.inf``
          Lower bound
        upper : float, DataContainer or numpy array, default ``np.inf``
          upper bound
        
        If passed a DataContainer or numpy array, the bounds can be set to different values for each element.
        '''
        super(IndicatorBox, self).__init__()
        
        # We set lower and upper to either a float or a numpy array        
        self.lower = _get_as_nparray_or_number(lower)
        self.upper = _get_as_nparray_or_number(upper)
        
    def __call__(self,x):
        
        '''Evaluates IndicatorBox at x'''
                
        
        if isinstance(self.lower, np.ndarray):
            if isinstance(self.upper, np.ndarray):
                return _array_within_limits_aa(x.as_array(), self.lower, self.upper)
            else:
                return _array_within_limits_af(x.as_array(), self.lower, self.upper)
        else:
            if isinstance(self.upper, np.ndarray):
                return _array_within_limits_fa(x.as_array(), self.lower, self.upper)
            else:
                return _array_within_limits_ff(x.as_array(), self.lower, self.upper)
    
    def gradient(self,x):
        '''IndicatorBox is not differentiable, so calling gradient will raise a ValueError'''
        return ValueError('Not Differentiable') 
    
    def convex_conjugate(self,x):
        '''Convex conjugate of IndicatorBox at x'''
        return x.maximum(0).sum()
         
    def proximal(self, x, tau, out=None):
        
        r'''Proximal operator of IndicatorBox at x

            .. math:: prox_{\tau * f}(x)
        '''
        should_return = False
        if out is None:
            should_return = True
            out = x * 0 
        outarr = out.as_array()

        if isinstance(self.lower, np.ndarray):
            if isinstance(self.upper, np.ndarray):
                _proximal_aa(x.as_array(), self.lower, self.upper, outarr)
            else:
                _proximal_af(x.as_array(), self.lower, self.upper, outarr)
        else:
            if isinstance(self.upper, np.ndarray):
                _proximal_fa(x.as_array(), self.lower, self.upper, outarr)
            else:
                np.clip(x.as_array(), self.lower, self.upper, out=outarr)

        if should_return:
            return out
            
    def proximal_conjugate(self, x, tau, out=None):
        
        r'''Proximal operator of the convex conjugate of IndicatorBox at x:

          ..math:: prox_{\tau * f^{*}}(x)
        '''

        if out is None:
            # x - tau * self.proximal(x/tau, tau):
            # use x as temporary storage variable
            x/=tau
            out = self.proximal(x, tau)
            out *= -1*tau
            # restore the values of x
            x*=tau
            out += x
            return out
        
        else:
            self.proximal(x/tau, tau, out=out)
            out *= -1*tau
            out += x

## Utilities
def _get_as_nparray_or_number(x):
    '''Returns x as a numpy array or a number'''
    try:
        return x.as_array()
    except AttributeError:
        # In this case we trust that it will be either a numpy ndarray 
        # or a number as described in the docstring
        return x

@numba.jit(nopython=True)
def _array_within_limits_ff(x, lower, upper):
    '''Returns 0 if all elements of x are within [lower, upper]'''

    for i in range(x.size):
        if x.flat[i] < lower or x.flat[i] > upper:
            return np.inf
    return 0
@numba.jit(nopython=True)
def _array_within_limits_af(x, lower, upper):
    '''Returns 0 if all elements of x are within [lower, upper]'''
    if x.size != lower.size:
        raise ValueError('x and lower must have the same size')
    for i in range(x.size):
        if x.flat[i] < lower.flat[i] or x.flat[i] > upper:
            return np.inf
    return 0

@numba.jit(nopython=True)
def _array_within_limits_aa(x, lower, upper):
    '''Returns 0 if all elements of x are within [lower, upper]'''
    if x.size != lower.size or x.size != upper.size:
        raise ValueError('x, lower and upper must have the same size')
    for i in range(x.size):
        if x.flat[i] < lower.flat[i] or x.flat[i] > upper.flat[i]:
            return np.inf
    return 0

@numba.jit(nopython=True)
def _array_within_limits_fa(x, lower, upper):
    '''Returns 0 if all elements of x are within [lower, upper]'''
    if x.size != upper.size:
        raise ValueError('x and upper must have the same size')
    for i in range(x.size):
        if x.flat[i] < lower or x.flat[i] > upper.flat[i]:
            return np.inf
    return 0

##########################################################################
@numba.jit(nopython=True)
def _proximal_aa(x, lower, upper, out):
    '''Similar to np.clip except that the clipping range can be defined by ndarrays'''
    if x.size != lower.size or x.size != upper.size:
        raise ValueError('x, lower and upper must have the same size')
    for i in numba.prange(x.size):
        if x.flat[i] < lower.flat[i]:
            out.flat[i] = lower.flat[i]
        else:
            out.flat[i] = x.flat[i]

        if out.flat[i] > upper.flat[i]:
            out.flat[i] = upper.flat[i]
            
@numba.jit(nopython=True)
def _proximal_af(x, lower, upper, out):
    '''Similar to np.clip except that the clipping range can be defined by ndarrays'''
    if x.size != lower.size :
        raise ValueError('x, lower and upper must have the same size')
    for i in numba.prange(x.size):
        if x.flat[i] < lower.flat[i]:
            out.flat[i] = lower.flat[i]
        else:
            out.flat[i] = x.flat[i]

        if out.flat[i] > upper:
            out.flat[i] = upper


@numba.jit(nopython=True)
def _proximal_fa(x, lower, upper, out):
    '''Similar to np.clip except that the clipping range can be defined by ndarrays'''
    if x.size != upper.size:
        raise ValueError('x, lower and upper must have the same size')
    for i in numba.prange(x.size):
        if x.flat[i] < lower:
            out.flat[i] = lower
        else:
            out.flat[i] = x.flat[i]

        if out.flat[i] > upper.flat[i]:
            out.flat[i] = upper.flat[i]
