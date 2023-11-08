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


from cil.optimisation.functions import SumFunction
#from cil.optimisation.utilities import Sampler TODO: after sampler merged in 
import numbers
from abc import ABC, abstractmethod

class ApproximateGradientSumFunction(SumFunction, ABC): 
    r"""ApproximateGradientSumFunction represents the following sum 

    .. math:: \sum_{i=1}^{n} F_{i} = (F_{1} + F_{2} + ... + F_{n})

    where :math:`n` is the number of functions. It is an abstract base class and any child classes must implement an `approximate_gradient` function.

    Parameters:
    -----------
    functions : `list`  of functions
                A list of functions: :code:`[F_{1}, F_{2}, ..., F_{n}]`. Each function is assumed to be smooth function with an implemented :func:`~Function.gradient` method. Each function must have the same domain. The number of functions must be strictly greater than 1. 
    sampler: An instance of one of the :meth:`~optimisation.utilities.sampler` classes which has a `next` function implemented and a `num_indices` property.
        This sampler is called each time gradient is called and  sets the internal `function_num` passed to the `approximate_gradient` function.  The `num_indices` must match the number of functions provided. Default is `Sampler.random_with_replacement(len(functions))`. 
   
         
            
    Note
    ----

    The :meth:`~ApproximateGradientSumFunction.gradient` returns the approximate gradient depending on an index provided by the  :code:`sampler` method. 

    Example
    -------

    .. math:: \sum_{i=1}^{n} F_{i}(x) = \sum_{i=1}^{n}\|A_{i} x - b_{i}\|^{2}

    >>> list_of_functions = [LeastSquares(Ai, b=bi)] for Ai,bi in zip(A_subsets, b_subsets))   
    >>> f = ApproximateGradientSumFunction(list_of_functions) 

    >>> list_of_functions = [LeastSquares(Ai, b=bi)] for Ai,bi in zip(A_subsets, b_subsets)) 
    >>> sampler = RandomSampling.random_shuffle(len(list_of_functions))
    >>> f = ApproximateGradientSumFunction(list_of_functions, sampler=sampler)   


    """

    def __init__(self, functions, sampler =None):
        
      #  if sampler is None:
      #      sampler=Sampler.random_with_replacement(len(functions)) #TODO: once sampler is merged in and unit test for this! 
        
        if not isinstance(functions, list):
            raise TypeError("Input to functions should be a list of functions")
        if not hasattr(sampler, "next"):
            raise ValueError('The provided sampler must have a `next` method')
        if not hasattr(sampler, "num_indices"):
            raise ValueError('The provided sampler must store the `num_indices` it samples from')
        if sampler.num_indices !=len(functions):
            raise ValueError('The sampler should choose from the same number of indices as there are functions passed to this approximate gradient method')
        
        self.sampler = sampler

        self.num_functions = len(functions)

        super(ApproximateGradientSumFunction, self).__init__(*functions)

    def __call__(self, x):
        r"""Returns the value of the sum of functions at :math:`x`.
        
        .. math:: (F_{1} + F_{2} + ... + F_{n})(x) = F_{1}(x) + F_{2}(x) + ... + F_{n}(x)
                
        """ 
        return super(ApproximateGradientSumFunction, self).__call__(x)

    def full_gradient(self, x, out=None):
        r"""Returns the value of the  full gradient of the sum of functions at :math:`x`.
        
        .. math:: \nabla_x(F_{1} + F_{2} + ... + F_{n})(x) = \nabla_xF_{1}(x) + \nabla_xF_{2}(x) + ... + \nabla_xF_{n}(x)
                
        """ 
        return super(ApproximateGradientSumFunction, self).gradient(x, out=out)

    @abstractmethod
    def approximate_gradient(self, x, function_num,   out=None):
        """ Computes the approximate gradient for each selected function at :code:`x`."""
        pass

    def gradient(self, x, out=None):
        """ Selects a random function using the `sampler` and then calls the approximate gradient at :code:`x`."""
        self.function_num = self.sampler.next()

        if isinstance(self.function_num, numbers.Number):
            return self.approximate_gradient(x, self.function_num, out=out)
        else:
            raise ValueError("Batch gradient is not yet implemented")
