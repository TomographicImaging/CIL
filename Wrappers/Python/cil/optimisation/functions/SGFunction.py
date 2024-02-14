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

from .ApproximateGradientSumFunction import ApproximateGradientSumFunction
from .Function import SumFunction

class SGFunction(ApproximateGradientSumFunction):

    """
    Stochastic gradient function, a child class of `ApproximateGradientSumFunction`, which defines from a list of functions, :math:`{f_1,...,f_n}` a `SumFunction`, :math:`f_1+...+f_n` where each time the `gradient` is called, the `sampler` provides an index, :math:`i \in {1,...,n}` 
   and the gradient function returns the approximate gradient :math:`n\nabla_xf_i(x)`. This can be used with the `cil.optimisation.algorithms` algorithm GD to give a stochastic gradient descent algorithm. 
   
    Parameters:
    -----------
    functions : `list`  of functions
                A list of functions: :code:`[F_{1}, F_{2}, ..., F_{n}]`. Each function is assumed to be smooth function with an implemented :func:`~Function.gradient` method. Each function must have the same domain. The number of functions must be strictly greater than 1. 
    sampler: An instance of a CIL Sampler class ( :meth:`~optimisation.utilities.sampler`) or another class which has a `next` function implemented to output integers in {1,...,n}.
        This sampler is called each time gradient is called and  sets the internal `function_num` passed to the `approximate_gradient` function.  The `num_indices` must match the number of functions provided. Default is `Sampler.random_with_replacement(len(functions))`. 
    """
  
    def __init__(self, functions, sampler=None):
        super(SGFunction, self).__init__(functions, sampler)    
        

    def approximate_gradient(self, x, function_num,  out=None):
        
        r""" Returns the gradient of the function at index `function_num` at :code:`x`. 
        
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
        
        
        self._update_data_passes(1/self.num_functions) # Also store indices of functions you have seen in a list of lists. 

        # compute gradient of randomly selected(function_num) function
        if out is None:
            out = self.functions[function_num].gradient(x)
        else:
            self.functions[function_num].gradient(x, out = out) 

        # scale wrt number of functions 
        out*=self.num_functions 
        
        return out         





           