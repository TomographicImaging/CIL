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
        Initialize the SGFunction.

        Parameters:
        ----------
        functions: list or BlockFunction 
            A list of functions. #TODO: write it a bit clearer what the sum function requires :) 
        sampler: callable or None, optional
            A callable object that selects the function or batch of functions to compute the gradient. 
            
     """
  
    def __init__(self, functions, sampler):
        if isinstance(functions, list):
            super(SGFunction, self).__init__(functions, sampler)    
        elif isinstance(functions, SumFunction):
            super(SGFunction, self).__init__(functions.functions, sampler) #TODO: is this the right thing to do? 
        else:
            raise TypeError("Input to functions should be a list of functions or a SumFunction")
    
    

    def approximate_gradient(self, function_num, x, out=None):
        
        """ Returns the gradient of the selected function or batch of functions at :code:`x`. 
            The function or batch of functions is selected using the :meth:`~ApproximateGradientSumFunction.next_function`.
        """     

        # flag to return or in-place computation
        should_return=False

        # compute gradient of randomly selected(function_num) function
        if out is None:
            out = self.functions[function_num].gradient(x)
            should_return=True
        else:
            self.functions[function_num].gradient(x, out = out) 

        # scale wrt number of functions 
        out*=self.num_functions # TODO: need to document this decision somewhere 
        
        if should_return:
            return out         





           