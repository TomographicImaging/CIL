# Copyright 2022 United Kingdom Research and Innovation
# Copyright 2022 The University of Manchester

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author(s): 
# Evangelos Papoutsellis (UKRI)
# Edoardo Pasca (UKRI)
# Gemma Fardell (UKRI)

from cil.optimisation.functions import Function
import numpy as np
import numba
from cil.utilities import multiprocessing as cil_mp


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

        Parameters
        ----------
        
            lower : float, DataContainer or numpy array, default ``-np.inf``
                Lower bound
            upper : float, DataContainer or numpy array, default ``np.inf``
                upper bound
        
        If passed a ``DataContainer`` or ``numpy array``, the bounds can be set to different values for each element.

        To suppress the evaluation of the function, set ``suppress_evaluation`` to ``True``. This will return 0 for any input.

        Example:
        --------

        .. code-block:: python

          ib = IndicatorBox(lower=0, upper=1)
          ib.set_suppress_evaluation(True)
          ib.evaluate(x) # returns 0
        '''
        super(IndicatorBox, self).__init__()
        
        # We set lower and upper to either a float or a numpy array        
        self.lower = _get_as_nparray_or_number(lower)
        self.upper = _get_as_nparray_or_number(upper)

        # default is to evaluate the function
        self._suppress_evaluation = False

    @property
    def suppress_evaluation(self):
        return self._suppress_evaluation

    def set_suppress_evaluation(self, value):
        '''Suppresses the evaluation of the function
        
        Parameters
        ----------

            value : bool
                If True, the function evaluation on any input will return 0, without calculation.
        '''
        if not isinstance(value, bool):
            raise ValueError('Value must be boolean')
        self._suppress_evaluation = value

    def __call__(self,x):
        '''Evaluates IndicatorBox at x
        
        Parameters
        ----------
        
            x : DataContainer
            
        Evaluates the IndicatorBox at x. If ``suppress_evaluation`` is ``True``, returns 0.  
        '''
        if not self.suppress_evaluation:
            return self.evaluate(x)    
        return 0.0

    def evaluate(self,x):
        
        '''Evaluates IndicatorBox at x'''
        
        num_threads = numba.get_num_threads()
        numba.set_num_threads(cil_mp.NUM_THREADS)
        breaking = np.zeros(numba.get_num_threads(), dtype=np.uint8)
                
        if isinstance(self.lower, np.ndarray):
            if isinstance(self.upper, np.ndarray):

                _array_within_limits_aa(x.as_array(), self.lower, self.upper, breaking)

            else:

                _array_within_limits_af(x.as_array(), self.lower, self.upper, breaking)

        else:
            if isinstance(self.upper, np.ndarray):

                _array_within_limits_fa(x.as_array(), self.lower, self.upper, breaking)

            else:

                _array_within_limits_ff(x.as_array(), self.lower, self.upper, breaking)

        numba.set_num_threads(num_threads)
        return np.inf if breaking.sum() > 0 else 0.0
    
    def gradient(self,x):
        '''IndicatorBox is not differentiable, so calling gradient will raise a ``ValueError``'''
        return ValueError('Not Differentiable') 
    
    def convex_conjugate(self,x):
        '''Convex conjugate of IndicatorBox at x'''
        return _convex_conjugate(x.as_array())
         
    def proximal(self, x, tau, out=None):
        
        r'''Proximal operator of IndicatorBox at x

        .. math:: prox_{\tau * f}(x)

        Parameters
        ----------

        x : DataContainer
            Input to the proximal operator
        tau : float
            Step size. Notice it is ignored in IndicatorBox
        out : DataContainer, optional
            Output of the proximal operator. If not provided, a new DataContainer is created.

        Note
        ----

            ``tau`` is ignored but it is in the signature of the generic Function class
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

          .. math:: prox_{\tau * f^{*}}(x)

          Parameters
          ----------

            x : DataContainer
                Input to the proximal operator
            tau : float
                Step size. Notice it is ignored in IndicatorBox, see ``proximal`` for details
            out : DataContainer, optional
                Output of the proximal operator. If not provided, a new DataContainer is created.

        '''

        # x - tau * self.proximal(x/tau, tau)
        should_return = False
        
        if out is None:
            out = self.proximal(x, tau)
            should_return = True
        else:
            self.proximal(x, tau, out=out)
        
        out.sapyb(-1., x, 1., out=out)

        if should_return:
            return out
        
## Utilities
def _get_as_nparray_or_number(x):
    '''Returns x as a numpy array or a number'''
    try:
        return x.as_array()
    except AttributeError:
        # In this case we trust that it will be either a numpy ndarray 
        # or a number as described in the docstring
        return x

@numba.jit(nopython=True, parallel=True)
def _array_within_limits_ff(x, lower, upper, breaking):
    '''Returns 0 if all elements of x are within [lower, upper]'''
    arr = x.ravel()
    for i in numba.prange(x.size):
        j = numba.np.ufunc.parallel._get_thread_id()
        
        if breaking[j] == 0 and (arr[i] < lower or arr[i] > upper):
            breaking[j] = 1

@numba.jit(nopython=True, parallel=True)
def _array_within_limits_af(x, lower, upper, breaking):
    '''Returns 0 if all elements of x are within [lower, upper]'''
    if x.size != lower.size:
        raise ValueError('x and lower must have the same size')
    arr = x.ravel()
    loarr = lower.ravel()
    for i in numba.prange(x.size):
        j = numba.np.ufunc.parallel._get_thread_id()
        
        if breaking[j] == 0 and (arr[i] < loarr[i] or xarr[i] > upper):
            breaking[j] = 1

@numba.jit(parallel=True, nopython=True)
def _array_within_limits_aa(x, lower, upper, breaking):
    '''Returns 0 if all elements of x are within [lower, upper]'''
    if x.size != lower.size or x.size != upper.size:
        raise ValueError('x, lower and upper must have the same size')
    arr = x.ravel()
    uparr = upper.ravel()
    loarr = lower.ravel()
    for i in numba.prange(x.size):
        j = numba.np.ufunc.parallel._get_thread_id()
        
        if breaking[j] == 0 and (arr[i] < loarr[i] or arr[i] > uparr[i]):
            breaking[j] = 1

@numba.jit(nopython=True, parallel=True)
def _array_within_limits_fa(x, lower, upper, breaking):
    '''Returns 0 if all elements of x are within [lower, upper]'''
    if x.size != upper.size:
        raise ValueError('x and upper must have the same size')
    arr = x.ravel()
    uparr = upper.ravel()
    for i in numba.prange(x.size):
        j = numba.np.ufunc.parallel._get_thread_id()
        
        if breaking[j] == 0 and (arr[i] < lower or arr[i] > uparr[i]):
            breaking[j] = 1

##########################################################################

@numba.jit(nopython=True, parallel=True)
def _proximal_aa(x, lower, upper, out):
    '''Similar to np.clip except that the clipping range can be defined by ndarrays'''
    if x.size != lower.size or x.size != upper.size:
        raise ValueError('x, lower and upper must have the same size')
    for i in numba.prange(x.size):
        if x.flat[i] < lower.flat[i]:
            out.flat[i] = lower.flat[i]
        elif out.flat[i] > upper.flat[i]:
            out.flat[i] = upper.flat[i]
        else:
            out.flat[i] = x.flat[i]

        
            
@numba.jit(nopython=True, parallel=True)
def _proximal_af(x, lower, upper, out):
    '''Similar to np.clip except that the clipping range can be defined by ndarrays'''
    if x.size != lower.size :
        raise ValueError('x, lower and upper must have the same size')
    for i in numba.prange(x.size):
        if x.flat[i] < lower.flat[i]:
            out.flat[i] = lower.flat[i]
        elif out.flat[i] > upper:
            out.flat[i] = upper
        else:
            out.flat[i] = x.flat[i]

@numba.jit(nopython=True, parallel=True)
def _proximal_fa(x, lower, upper, out):
    '''Similar to np.clip except that the clipping range can be defined by ndarrays'''
    if x.size != upper.size:
        raise ValueError('x, lower and upper must have the same size')
    for i in numba.prange(x.size):
        if x.flat[i] < lower:
            out.flat[i] = lower
        elif out.flat[i] > upper.flat[i]:
            out.flat[i] = upper.flat[i]
        else:
            out.flat[i] = x.flat[i]

@numba.jit(nopython=True)
def _convex_conjugate(x):
    '''Convex conjugate of IndicatorBox
    
    im.maximum(0).sum()
    '''
    acc = np.zeros((numba.get_num_threads()), dtype=np.uint32)
    for i in numba.prange(x.size):
        j = numba.np.ufunc.parallel._get_thread_id()
        if x.flat[i] > 0:
            acc[j] += x.flat[i]
    return np.sum(acc)
