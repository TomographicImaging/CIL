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
from cil.optimisation.utilities import Sampler 
import numbers
from abc import ABC, abstractmethod

class ApproximateGradientSumFunction(SumFunction, ABC): 
    r"""ApproximateGradientSumFunction represents the following sum 

    .. math:: \sum_{i=1}^{n} F_{i} = (F_{1} + F_{2} + ... + F_{n})

    where :math:`n` is the number of functions. The gradient method from a CIL function is overwritten and calls an approximate gradient method. 
    
    It is an abstract base class and any child classes must implement an `approximate_gradient` function.

    Parameters:
    -----------
    functions : `list`  of functions
                A list of functions: :code:`[F_{1}, F_{2}, ..., F_{n}]`. Each function is assumed to be smooth function with an implemented :func:`~Function.gradient` method. Each function must have the same domain. The number of functions must be strictly greater than 1. 
    sampler: An instance of a CIL Sampler class ( :meth:`~optimisation.utilities.sampler`) or another class which has a `next` function implemented to output integers in {1,...,n}.
        This sampler is called each time gradient is called and  sets the internal `function_num` passed to the `approximate_gradient` function.  The `num_indices` must match the number of functions provided. Default is `Sampler.random_with_replacement(len(functions))`. 
   
         
            
    #TODO: NOTE about batch sizes 
    
    
    Note
    ----
    The :meth:`~ApproximateGradientSumFunction.gradient` returns the approximate gradient depending on an index provided by the  :code:`sampler` method. 

    Example
    -------
    Consider the objective is to minimise: 
    
    .. math:: \sum_{i=1}^{n} F_{i}(x) = \sum_{i=1}^{n}\|A_{i} x - b_{i}\|^{2}

    >>> list_of_functions = [LeastSquares(Ai, b=bi)] for Ai,bi in zip(A_subsets, b_subsets))   
    >>> f = ApproximateGradientSumFunction(list_of_functions) 

    >>> list_of_functions = [LeastSquares(Ai, b=bi)] for Ai,bi in zip(A_subsets, b_subsets)) 
    >>> sampler = RandomSampling.random_shuffle(len(list_of_functions))
    >>> f = ApproximateGradientSumFunction(list_of_functions, sampler=sampler)   


    """

    def __init__(self, functions, sampler =None):
        
        if sampler is None:
            sampler=Sampler.random_with_replacement(len(functions)) 
        
        if not isinstance(functions, list):
            raise TypeError("Input to functions should be a list of functions")
        if not hasattr(sampler, "next"):
            raise ValueError('The provided sampler must have a `next` method')
        
        self.sampler = sampler
        
        self._data_passes=[]
        

        super(ApproximateGradientSumFunction, self).__init__(*functions)

    def __call__(self, x):
        r"""Returns the value of the sum of functions at :math:`x`.
        
        .. math:: (F_{1} + F_{2} + ... + F_{n})(x) = F_{1}(x) + F_{2}(x) + ... + F_{n}(x)
        
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
        
        .. math:: \nabla_x(F_{1} + F_{2} + ... + F_{n})(x) = \nabla_xF_{1}(x) + \nabla_xF_{2}(x) + ... + \nabla_xF_{n}(x)
                
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
        """ Computes the approximate gradient for each selected function at :code:`x` given a `function_number` in {1,...,len(functions)}.
        
        Parameters
        ----------
        x : DataContainer
        out: return DataContainer, if `None` a new DataContainer is returned, default `None`.
        function_num: `int` 
            Between 1 and the number of functions in the list  
        Returns
        --------
        DataContainer
            the value of the approximate gradient of the sum function at :code:`x` given a `function_number` in {1,...,len(functions)} or nothing if `out`  
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

        if self.function_num>self.num_functions:
            raise IndexError(
                'The sampler has outputted an index larger than the number of functions to sample from. Please ensure your sampler samples from {1,2,...,len(functions)} only.')
        
        if isinstance(self.function_num, numbers.Number):
            return self.approximate_gradient(x, self.function_num, out=out)
        raise ValueError("Batch gradient is not yet implemented")


    def _update_data_passes(self, value):
        """ Internal function that updates the list which stores the data passes
        
        Parameters
        ----------
        value: float
            The additional proportion of the data that has been seen 

        """
        try:
            self._data_passes.append(
                self._data_passes[-1] + value) #Need to do some rounding 
        except IndexError:
            self._data_passes.append(value)
        
    @property
    def data_passes(self):
        return self._data_passes
    
    @property #TODO: is this in the parent class?? 
    def num_functions(self):
        return len(self.functions)