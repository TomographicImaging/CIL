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
# - CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt
# - Daniel Deidda (National Physical Laboratory, UK)
# - Claire Delplancke (Electricite de France, Research and Development)
# - Ashley Gillman (Australian e-Health Res. Ctr., CSIRO, Brisbane, Queensland, Australia)
# - Zeljko Kerata (Department of Computer Science, University College London, UK)
# - Evgueni Ovtchinnikov (STFC - UKRI)
# - Georg Schramm (Department of Imaging and Pathology, Division of Nuclear Medicine, KU Leuven, Leuven, Belgium)



from cil.optimisation.functions import SumFunction
from cil.optimisation.utilities import Sampler
import numbers
from abc import ABC, abstractmethod
import numpy as np


class ApproximateGradientSumFunction(SumFunction, ABC):
    r"""ApproximateGradientSumFunction represents the following sum 

    .. math:: \sum_{i=0}^{n-1} f_{i} = (f_{0} + f_{2} + ... + f_{n-1})

    where there are :math:`n` functions. This function class has two ways of calling gradient:
    - `full_gradient` calculates the gradient of the sum :math:`\sum_{i=0}^{n-1} \nabla f_{i}`
    - `gradient` calls an `approximate_gradient` function which may be less computationally expensive to calculate than the full gradient
    
    

    This class is an abstract class.

    Parameters:
    -----------
    functions : `list`  of functions
                A list of functions: :math:`[f_{0}, f_{2}, ..., f_{n-1}]`. Each function is assumed to be smooth with an implemented :func:`~Function.gradient` method. All functions must have the same domain. The number of functions (equivalently the length of the list) must be strictly greater than 1. 
    sampler: An instance of a CIL Sampler class ( :meth:`~optimisation.utilities.sampler`) or of another class which has a :code:`__next__` function implemented to output integers in :math:`{0,...,n-1}`.
        This sampler is called each time :code:`gradient` is called and sets the internal :code:`function_num` passed to the :code:`approximate_gradient` function.  Default is :code:`Sampler.random_with_replacement(len(functions))`. 

    Note
    -----
    We ensure that the approximate gradient is of a similar order of magnitude to the full gradient calculation. For example, in the :code:`SGFunction` we approximate the full gradient by :math:`n\nabla f_i` for an index :math:`i` given by the sampler. 
    The multiplication by :math:`n` is a choice to more easily allow comparisons between stochastic and non-stochastic methods and between stochastic methods with varying numbers of subsets. 

    Note
    -----
    Each time :code:`gradient` is called the class keeps track of which functions have been used to calculate the gradient. This may be useful for debugging or plotting after using this function in an iterative algorithm:  
    - :code:`data_passes_indices` is a list of lists. Each time :code:`gradient` is called a list is appended with with the indices of the functions have been used to calculate the gradient.  
    - :code:`data_passes` is a list. Each time :code:`gradient` is called an entry is appended with  the proportion of the data used when calculating the approximate gradient  since the class was initialised (a full gradient calculation would be 1 full data pass). Warning: if your functions do not contain an equal `amount` of data, for example your data was not partitioned into equal batches, then you must first use the :code:`set_data_partition_weights` function for this to be accurate.   



    Note
    ----
    The :meth:`~ApproximateGradientSumFunction.gradient` returns the approximate gradient depending on an index provided by the  :code:`sampler` method. 

    Example
    -------
    This class is an abstract base class, so we give an example using the SGFunction child class. 
    
    Consider the objective is to minimise: 

    .. math:: \sum_{i=0}^{n-1} f_{i}(x) = \sum_{i=0}^{n-1}\|A_{i} x - b_{i}\|^{2}

    >>> list_of_functions = [LeastSquares(Ai, b=bi)] for Ai,bi in zip(A_subsets, b_subsets))   
    >>> f = ApproximateGradientSumFunction(list_of_functions) 

    >>> list_of_functions = [LeastSquares(Ai, b=bi)] for Ai,bi in zip(A_subsets, b_subsets)) 
    >>> sampler = Sampler.sequential(len(list_of_functions))
    >>> f = SGFunction(list_of_functions, sampler=sampler)   
    >>> f.full_gradient(x)
    This will return :math:`\sum_{i=0}^{n-1} \nabla f_{i}(x)`
    >>> f.gradient(x)
    As per the approximate gradient implementation in the SGFunction this will return :math:`\nabla f_{0}`. The choice of the `0` index is because we chose a `sequential` sampler and this is the first time we called `gradient`. 
    >>> f.gradient(x)
    This will return :math:`\nabla f_{1}` because we chose  a `sequential` sampler and this is the second time we called `gradient`. 



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
            The value of the gradient of the sum function at x or nothing if `out`  
        """

        return super(ApproximateGradientSumFunction, self).gradient(x, out=out)

    @abstractmethod
    def approximate_gradient(self, x, function_num,   out=None):
        """ Returns the approximate gradient at a given point :code:`x` given a `function_number` in {0,...,len(functions)-1}.

        Parameters
        ----------
        x : DataContainer
        out: return DataContainer, if `None` a new DataContainer is returned, default `None`.
        function_num: `int` 
            Between 0 and the number of functions in the list  
        Returns
        --------
        DataContainer
            the value of the approximate gradient of the sum function at :code:`x` given a `function_number` in {0,...,len(functions)-1} 
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
            the value of the approximate gradient of the sum function at :code:`x`   
        """

        self.function_num = self.sampler.next()
        
        self._update_data_passes_indices([self.function_num])
        

        
        return self.approximate_gradient(x, self.function_num, out=out)
        

    def _update_data_passes_indices(self, indices):
        """ Internal function that updates the list of lists containing the function indices used to calculate the approximate gradient. 
        Parameters
        ----------
        indices: list
            List of indices used to calculate the approximate gradient in a given iteration

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
        """ The property `data_passes_indices` is a list of lists. Each time `gradient` is called a list is appended with with the indices of the functions have been used to calculate the gradient.  """
        return self._data_passes_indices

    @property
    def data_passes(self):
        """ The property `data_passes` is a list of floats. Each time `gradient` is called an entry is appended to this list with  the proportion of the data used when calculating the approximate gradient  since the class was initialised (a full gradient calculation would be 1 full data pass). Warning: if your functions do not contain an equal `amount` of data, for example your data was not partitioned into equal batches, then you must first use the `set_data_partition_weights" function for this to be accurate.   """
        data_passes = []
        for el in self._data_passes_indices:
            try:
                data_passes.append(data_passes[-1])
            except IndexError:
                data_passes.append(0)
            for i in el:
                data_passes[-1] += self._partition_weights[i]
        return data_passes
