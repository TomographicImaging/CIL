#  Copyright 2019 United Kingdom Research and Innovation
#  Copyright 2019 The University of Manchester
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

from cil.optimisation.functions import Function
import numpy as np
import numba
from cil.utilities import multiprocessing as cil_mp
import logging


class IndicatorBox(Function):
    r'''Indicator function for box constraint

      .. math::

         f(x) = \mathbb{I}_{[a, b]} = \begin{cases}
                                            0, \text{ if } x \in [a, b] \\
                                            \infty, \text{otherwise}
                                     \end{cases}

    Parameters
    ----------
    lower : float, DataContainer or numpy array, default None
        Lower bound. If set to None, it is equivalent to ``-np.inf``.
    upper : float, DataContainer or numpy array, default None
        Upper bound. If set to None, it is equivalent to ``np.inf``.
    accelerated : bool, default True
        Specifies whether to use the accelerated version or not, using numba or
        numpy backends respectively.


    If ``lower`` or ``upper`` are passed a ``DataContainer`` (or derived class
    such as ``ImageData`` or ``AcquisitionData``) or a ``numpy array``, the bounds
    can be set to different values for each element.

    In order to save computing time it is possible to suppress the evaluation of
    the function. This is achieved by setting ``suppress_evaluation`` to ``True``.
    ``IndicatorBox`` evaluated on any input will then return 0.

    If ``accelerated`` is set to ``True`` (default), the Numba backend is used.
    Otherwise, the Numpy backend is used. An optional parameter to set the number of
    threads used by Numba can be set with ``set_num_threads``. Setting the number of
    threads when ``accelerate`` is set to ``False`` will not have any effect.
    The default number of threads is defined in the ``cil.utilities.multiprocessing``
    module, and it is equivalent to half of the CPU cores available.

    Example:
    --------

    In order to save computing time it is possible to suppress the evaluation of the
    function.

    .. code-block:: python

        ib = IndicatorBox(lower=0, upper=1)
        ib.set_suppress_evaluation(True)
        ib.evaluate(x) # returns 0


    Example:
    --------

    Set the number of threads used in accelerated mode.

    .. code-block:: python


        num_threads = 4
        ib = IndicatorBox(lower=0, upper=1)
        ib.set_num_threads(num_threads)
    '''

    def __new__(cls, lower=None, upper=None, accelerated=True):
        if accelerated:
            logging.info("Numba backend is used.")
            return super(IndicatorBox, cls).__new__(IndicatorBox_numba)
        else:
            logging.info("Numpy backend is used.")
            return super(IndicatorBox, cls).__new__(IndicatorBox_numpy)

    def __init__(self, lower=None, upper=None, accelerated=True):
        '''__init__'''
        super(IndicatorBox, self).__init__()

        # We set lower and upper to either a float or a numpy array
        self.lower = -np.inf if lower is None else _get_as_nparray_or_number(
            lower)
        self.upper = np.inf if upper is None else _get_as_nparray_or_number(
            upper)

        self.orig_lower = lower
        self.orig_upper = upper
        # default is to evaluate the function
        self._suppress_evaluation = False

        # optional parameter to track the number of threads used by numba
        self._num_threads = None

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

    def __call__(self, x):
        '''Evaluates IndicatorBox at x

        Parameters
        ----------

            x : DataContainer

        Evaluates the IndicatorBox at x. If ``suppress_evaluation`` is ``True``, returns 0.
        '''
        if not self.suppress_evaluation:
            return self.evaluate(x)
        return 0.0

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
            out = x.copy()
        else:
            out.fill(x)
        outarr = out.as_array()

        # calculate the proximal
        self._proximal(outarr)

        out.fill(outarr)
        if should_return:
            return out

    def gradient(self, x, out=None):
        '''IndicatorBox is not differentiable, so calling gradient will raise a ``ValueError``'''
        raise NotImplementedError('The IndicatorBox is not differentiable')

    def _proximal(self, outarr):
        raise NotImplementedError('Implement this in the derived class')

    @property
    def num_threads(self):
        '''Get the optional number of threads parameter to use for the accelerated version.

        Defaults to the value set in the CIL multiprocessing module.'''
        return cil_mp.NUM_THREADS if self._num_threads is None else self._num_threads

    def set_num_threads(self, value):
        '''Set the optional number of threads parameter to use for the accelerated version.

        This is discarded if ``accelerated=False``.'''
        self._num_threads = value


class IndicatorBox_numba(IndicatorBox):

    def evaluate(self, x):
        '''Evaluates IndicatorBox at x'''
        # set the number of threads to the number of threads defined by the user
        # or default to what set in the CIL multiprocessing module
        num_threads = numba.get_num_threads()
        numba.set_num_threads(self.num_threads)
        breaking = np.zeros(numba.get_num_threads(), dtype=np.uint8)

        if isinstance(self.lower, np.ndarray):
            if isinstance(self.upper, np.ndarray):

                _array_within_limits_aa(x.as_array(), self.lower, self.upper,
                                        breaking)

            else:

                _array_within_limits_af(x.as_array(), self.lower, self.upper,
                                        breaking)

        else:
            if isinstance(self.upper, np.ndarray):

                _array_within_limits_fa(x.as_array(), self.lower, self.upper,
                                        breaking)

            else:

                _array_within_limits_ff(x.as_array(), self.lower, self.upper,
                                        breaking)

        # reset the number of threads to the original value
        numba.set_num_threads(num_threads)
        return np.inf if breaking.sum() > 0 else 0.0

    def convex_conjugate(self, x):
        '''Convex conjugate of IndicatorBox at x'''
        # set the number of threads to the number of threads defined by the user
        # or default to what set in the CIL multiprocessing module
        num_threads = numba.get_num_threads()
        numba.set_num_threads(self.num_threads)

        acc = np.zeros((numba.get_num_threads()), dtype=np.uint32)
        _convex_conjugate(x.as_array(), acc)

        # reset the number of threads to the original value
        numba.set_num_threads(num_threads)

        return np.sum(acc)

    def _proximal(self, outarr):
        if self.orig_lower is not None and self.orig_upper is not None:
            if isinstance(self.lower, np.ndarray):
                if isinstance(self.upper, np.ndarray):
                    _proximal_aa(outarr, self.lower, self.upper)
                else:
                    _proximal_af(outarr, self.lower, self.upper)

            else:
                if isinstance(self.upper, np.ndarray):
                    _proximal_fa(outarr, self.lower, self.upper)
                else:
                    np.clip(outarr, self.lower, self.upper, out=outarr)

        elif self.orig_lower is None:
            if isinstance(self.upper, np.ndarray):
                _proximal_na(outarr, self.upper)
            else:
                np.clip(outarr, None, self.upper, out=outarr)

        elif self.orig_upper is None:
            if isinstance(self.lower, np.ndarray):
                _proximal_an(outarr, self.lower)
            else:
                np.clip(outarr, self.lower, None, out=outarr)


class IndicatorBox_numpy(IndicatorBox):

    def evaluate(self, x):
        '''Evaluates IndicatorBox at x'''

        if (np.all(x.as_array() >= self.lower)
                and np.all(x.as_array() <= self.upper)):
            val = 0
        else:
            val = np.inf
        return val

    def convex_conjugate(self, x):
        '''Convex conjugate of IndicatorBox at x'''
        return x.maximum(0).sum()

    def _proximal(self, outarr):
        np.clip(outarr,
                None if self.orig_lower is None else self.lower,
                None if self.orig_upper is None else self.upper,
                out=outarr)


## Utilities
def _get_as_nparray_or_number(x):
    '''Returns x as a numpy array or a number'''
    try:
        return x.as_array()
    except AttributeError:
        # In this case we trust that it will be either a numpy ndarray
        # or a number as described in the docstring
        logging.info('Assuming that x is a numpy array or a number')
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

        if breaking[j] == 0 and (arr[i] < loarr[i] or arr[i] > upper):
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
def _proximal_aa(x, lower, upper):
    '''Slightly faster than using np.clip'''
    if x.size != lower.size or x.size != upper.size:
        raise ValueError('x, lower and upper must have the same size')
    arr = x.ravel()
    loarr = lower.ravel()
    uparr = upper.ravel()
    for i in numba.prange(x.size):
        if arr[i] < loarr[i]:
            arr[i] = loarr[i]
        if arr[i] > uparr[i]:
            arr[i] = uparr[i]


@numba.jit(nopython=True, parallel=True)
def _proximal_af(x, lower, upper):
    '''Slightly faster than using np.clip'''
    if x.size != lower.size:
        raise ValueError('x, lower and upper must have the same size')
    arr = x.ravel()
    loarr = lower.ravel()
    for i in numba.prange(x.size):
        if arr[i] < loarr[i]:
            arr[i] = loarr[i]
        if arr[i] > upper:
            arr[i] = upper


@numba.jit(nopython=True, parallel=True)
def _proximal_fa(x, lower, upper):
    '''Slightly faster than using np.clip'''
    if x.size != upper.size:
        raise ValueError('x, lower and upper must have the same size')
    arr = x.ravel()
    uparr = upper.ravel()
    for i in numba.prange(x.size):
        if arr[i] < lower:
            arr[i] = lower
        if arr[i] > uparr[i]:
            arr[i] = uparr[i]


@numba.jit(nopython=True, parallel=True)
def _proximal_na(x, upper):
    '''Slightly faster than using np.clip'''
    if x.size != upper.size:
        raise ValueError('x and upper must have the same size')
    arr = x.ravel()
    uparr = upper.ravel()
    for i in numba.prange(x.size):
        if arr[i] > uparr[i]:
            arr[i] = uparr[i]


@numba.jit(nopython=True, parallel=True)
def _proximal_an(x, lower):
    '''Slightly faster than using np.clip'''
    if x.size != lower.size:
        raise ValueError('x and lower must have the same size')
    arr = x.ravel()
    loarr = lower.ravel()
    for i in numba.prange(x.size):
        if arr[i] < loarr[i]:
            arr[i] = loarr[i]


@numba.jit(nopython=True, parallel=True)
def _convex_conjugate(x, acc):
    '''Convex conjugate of IndicatorBox

    im.maximum(0).sum()
    '''
    arr = x.ravel()
    j = 0
    for i in numba.prange(x.size):
        j = numba.np.ufunc.parallel._get_thread_id()

        if arr[i] > 0:
            acc[j] += arr[i]
