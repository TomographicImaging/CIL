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


from cil.optimisation.functions import SumFunction
from cil.optimisation.utilities import Sampler
import numbers
from abc import ABC, abstractmethod
import numpy as np


class ApproximateGradientSumFunction(SumFunction, ABC):
    r"""ApproximateGradientSumFunction represents the following sum 

    .. math:: \sum_{i=0}^{n-1} f_{i} = (f_{0} + f_{2} + ... + f_{n-1})

    where there are :math:`n` functions. The gradient method from a CIL function is overwritten and calls an approximate gradient method. 

    It is an abstract base class and any child classes must implement an `approximate_gradient` function.

    Parameters:
    -----------
    functions : `list`  of functions
                A list of functions: :code:`[f_{0}, f_{2}, ..., f_{n-1}]`. Each function is assumed to be smooth function with an implemented :func:`~Function.gradient` method. Each function must have the same domain. The number of functions must be strictly greater than 1. 
    sampler: An instance of a CIL Sampler class ( :meth:`~optimisation.utilities.sampler`) or another class which has a `next` function implemented to output integers in {0,...,n-1}.
        This sampler is called each time gradient is called and  sets the internal `function_num` passed to the `approximate_gradient` function.  The `num_indices` must match the number of functions provided. Default is `Sampler.random_with_replacement(len(functions))`. 


    Note
    -----
    We provide two ways of keeping track the amount of data you have seen: 
        - `data_passes_indices` a list of lists the length of which should be the number of iterations currently run. Each entry corresponds to the indices of the function numbers seen in that iteration. 
        - `data_passes` is a list of floats the length of which should be the number of iterations currently run. Each entry corresponds to the proportion of data seen up to this iteration. Warning: if your functions do not contain an equal `amount` of data, for example your data was not partitioned into equal batches, then you must first use the `set_data_partition_weights" function for this to be accurate.   



    Note
    ----
    The :meth:`~ApproximateGradientSumFunction.gradient` returns the approximate gradient depending on an index provided by the  :code:`sampler` method. 

    Example
    -------
    Consider the objective is to minimise: 

    .. math:: \sum_{i=0}^{n-1} f_{i}(x) = \sum_{i=0}^{n-1}\|A_{i} x - b_{i}\|^{2}

    >>> list_of_functions = [LeastSquares(Ai, b=bi)] for Ai,bi in zip(A_subsets, b_subsets))   
    >>> f = ApproximateGradientSumFunction(list_of_functions) 

    >>> list_of_functions = [LeastSquares(Ai, b=bi)] for Ai,bi in zip(A_subsets, b_subsets)) 
    >>> sampler = Sampler.random_shuffle(len(list_of_functions))
    >>> f = ApproximateGradientSumFunction(list_of_functions, sampler=sampler)   


    """

    def __init__(self, functions, sampler=None):

        if sampler is None:
            sampler = Sampler.random_with_replacement(len(functions))

        if not isinstance(functions, list):
            raise TypeError("Input to functions should be a list of functions")
        if not hasattr(sampler, "next"):
            raise ValueError('The provided sampler must have a `next` method')

        self.sampler = sampler

        self._partition_weights = [1 / len(functions)] * len(functions)

        self._data_passes_indices = []

        super(ApproximateGradientSumFunction, self).__init__(*functions)

    def __call__(self, x):
        r"""Returns the value of the sum of functions at :math:`x`.

        .. math:: (f_{0} + f_{1} + ... + f_{n-1})(x) = f_{0}(x) + f_{1}(x) + ... + f_{n-1}(x)

        Parameters
        ----------
        x : DataContainer

        --------
        float
            the value of the SumFunction at x


        """
        return super(ApproximateGradientSumFunction, self).__call__(x)

    def full_gradient(self, x, out=None):
        r"""Returns the value of the  full gradient of the sum of functions at :math:`x`.

        .. math:: \nabla_x(f_{0} + f_{1} + ... + f_{n-1})(x) = \nabla_xf_{0}(x) + \nabla_xf_{1}(x) + ... + \nabla_xf_{n-1}(x)

        Parameters
        ----------
        x : DataContainer
        out: return DataContainer, if `None` a new DataContainer is returned, default `None`.

        Returns
        --------
        DataContainer
            the value of the gradient of the sum function at x or nothing if `out`  
        """

        return super(ApproximateGradientSumFunction, self).gradient(x, out=out)

    @abstractmethod
    def approximate_gradient(self, x, function_num,   out=None):
        """ Computes the approximate gradient for each selected function at :code:`x` given a `function_number` in {0,...,len(functions)-1}.

        Parameters
        ----------
        x : DataContainer
        out: return DataContainer, if `None` a new DataContainer is returned, default `None`.
        function_num: `int` 
            Between 0 and the number of functions in the list  
        Returns
        --------
        DataContainer
            the value of the approximate gradient of the sum function at :code:`x` given a `function_number` in {0,...,len(functions)-1} or nothing if `out`  
        """
        pass

    def gradient(self, x, out=None):
        """ Selects a random function using the `sampler` and then calls the approximate gradient at :code:`x`

        Parameters
        ----------
        x : DataContainer
        out: return DataContainer, if `None` a new DataContainer is returned, default `None`.

        Returns
        --------
        DataContainer
            the value of the approximate gradient of the sum function at :code:`x`  or nothing if `out`  
        """

        self.function_num = self.sampler.next()

        if self.function_num > self.num_functions:
            raise IndexError(
                'The sampler has outputted an index larger than the number of functions to sample from. Please ensure your sampler samples from {0,1,...,len(functions)-1} only.')

        if isinstance(self.function_num, numbers.Number):
            return self.approximate_gradient(x, self.function_num, out=out)
        raise ValueError("Batch gradient is not yet implemented")

    def _update_data_passes_indices(self, indices):
        """ Internal function that updates the list of lists containing the function indices seen at each iteration. 

        Parameters
        ----------
        indices: list
            List of indices seen in a given iteration

        """
        self._data_passes_indices.append(indices)

    def set_data_partition_weights(self, weights):
        """ Setter for the partition weights used to calculate the data passes  

        Parameters
        ----------
        weights: list of positive floats that sum to one. 
            The proportion of the data held in each function. Equivalent to the proportions that you partitioned your data into. 

        """
        if len(weights) != len(self.functions):
            raise ValueError(
                'The provided weights must be a list the same length as the number of functions')

        if abs(sum(weights) - 1) > 1e-6:
            raise ValueError('The provided weights must sum to one')

        if any(np.array(weights) < 0):
            raise ValueError(
                'The provided weights must be greater than or equal to zero')

        self._partition_weights = weights

    @property
    def data_passes_indices(self):
        return self._data_passes_indices

    @property
    def data_passes(self):
        data_passes = []
        for el in self.data_passes_indices:
            try:
                data_passes.append(data_passes[-1])
            except IndexError:
                data_passes.append(0)
            for i in el:
                data_passes[-1] += self._partition_weights[i]
        return data_passes
