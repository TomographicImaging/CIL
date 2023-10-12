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

import numbers


class ApproximateGradientSumFunction(SumFunction):

    r"""ApproximateGradientSumFunction represents the following sum 

    .. math:: \sum_{i=1}^{n} F_{i} = (F_{1} + F_{2} + ... + F_{n})

    where :math:`n` is the number of functions.

    Parameters:
    -----------
    functions : list(functions) #TODO: do we want this to be a list of functions or a BlockFunction? Perhaps it could be a list here and a BlockFunction for SGFunction? 
                A list of functions: :code:`[F_{1}, F_{2}, ..., F_{n}]`. Each function is assumed to be smooth function with an implemented :func:`~Function.gradient` method.
    sampler: An instance of the :meth:`~framework.sampler` class which has a next function which gives the next subset to calculate the gradient for. 

    Note
    ----

    The :meth:`~ApproximateGradientSumFunction.gradient` computes the `gradient` of only one function of a batch of functions 
    depending on the :code:`sampler` method. 

    Example
    -------

    .. math:: \sum_{i=1}^{n} F_{i}(x) = \sum_{i=1}^{n}\|A_{i} x - b_{i}\|^{2}

    >>> list_of_functions = [LeastSquares(Ai, b=bi)] for Ai,bi in zip(A_subsets, b_subsets))   
    >>> f = ApproximateGradientSumFunction(list_of_functions) 

    >>> list_of_functions = [LeastSquares(Ai, b=bi)] for Ai,bi in zip(A_subsets, b_subsets)) 
    >>> sampler = RandomSampling.random_shuffle(len(list_of_functions))
    >>> f = ApproximateGradientSumFunction(list_of_functions, sampler=sampler)   


    """

    def __init__(self, functions, sampler):

        self.sampler = sampler

        self.num_functions = len(functions)

        super(ApproximateGradientSumFunction, self).__init__(*functions)

    def __call__(self, x):
        r""" Computes the full sum at :code:`x`. It is the sum of the outputs for each function.  """
        return super(ApproximateGradientSumFunction, self).__call__(x)

    def full_gradient(self, x, out=None):
        r""" Computes the full gradient at :code:`x`. It is the sum of all the gradients for each function. """
        return super(ApproximateGradientSumFunction, self).gradient(x, out=out)

    def approximate_gradient(self, function_num, x,  out=None):
        """ Computes the approximate gradient for each selected function at :code:`x`."""
        raise NotImplemented

    def gradient(self, x, out=None):
        """ Selects a random function and uses this to calculate the approximate gradient at :code:`x`."""
        self.function_num = next(self.sampler)

        # single function
        if isinstance(self.function_num, numbers.Number):
            return self.approximate_gradient(self.function_num, x, out=out)
        else:
            raise ValueError("Batch gradient is not yet implemented")


        
