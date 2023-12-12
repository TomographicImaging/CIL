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

class SAGFunction(ApproximateGradientSumFunction):

    """
    Stochastic gradient function, a child class of `ApproximateGradientSumFunction`, which defines from a list of functions, :math:`{f_1,...,f_n}` a `SumFunction`, :math:`f_1+...+f_n` where each time the `gradient` is called, the `sampler` provides an index, :math:`i \in {1,...,n}` 
   and the gradient function returns the approximate gradient :math:`n\nabla_xf_i(x)`. This can be used with the `cil.optimisation.algorithms` algorithm GD to give a stochastic gradient descent algorithm. 
   
   Parameters:
    -----------
    functions : `list`  of functions
                A list of functions: :code:`[F_{1}, F_{2}, ..., F_{n}]`. Each function is assumed to be smooth function with an implemented :func:`~Function.gradient` method. Each function must have the same domain. The number of functions must be strictly greater than 1. 
    sampler: An instance of one of the :meth:`~optimisation.utilities.sampler` classes which has a `next` function implemented and a `num_indices` property.
        This sampler is called each time gradient is called and  sets the internal `function_num` passed to the `approximate_gradient` function.  The `num_indices` must match the number of functions provided. Default is `Sampler.random_with_replacement(len(functions))`. 
    """
  
    def __init__(self, functions, sampler=None):
        self.set_up_done=False
            
        super(SAGFunction, self).__init__(functions, sampler)    
        
    
    

    def approximate_gradient(self, x, function_num,  out=None):
        
        """ Returns the gradient of the selected function or batch of functions at :code:`x`. 
            The function num is selected using the :meth:`~ApproximateGradientSumFunction.next_function`.
        
        Parameters
        ----------
        x: element in the domain of the `functions`
        
        function_num: `int` 
            Between 1 and the number of functions in the list  
        
        
        
        """     
        if not self.set_up_done:
            self._set_up(x)
        
        self.stoch_grad_at_iterate=self.functions[function_num].gradient(x)
        self.data_passes.append(round(self.data_passes[-1] + 1./self.num_functions,4))
        self.stochastic_grad_difference= self.stoch_grad_at_iterate.sapyb(1., self.list_stored_gradients[function_num], -1.)
        
        # flag to return or in-place computation
        should_return=False
        
        # compute gradient of randomly selected(function_num) function
        if out is None:
            res = x*0. # for CIL/SIRF compatibility
            # due to the convention that we follow: without the 1/n factor
            self.stochastic_grad_difference.sapyb(1., self.full_gradient_at_iterate, 1., out=res)
            should_return = True
        else:
            # due to the convention that we follow: without the 1/n factor
            self.stochastic_grad_difference.sapyb(1., self.full_gradient_at_iterate, 1., out=out) 

        self.list_stored_gradients[function_num].fill(self.stoch_grad_at_iterate)
        self.full_gradient_at_iterate.sapyb(1., self.stochastic_grad_difference, 1., out=self.full_gradient_at_iterate)
        
        if should_return:
            return res        


    def _set_up(self, x):

        r"""Initialize subset gradients :math:`v_{i}` and full gradient that are stored in memory.
        The initial point is 0 by default.
        """
        if self.warm_start:
            self.list_stored_gradients = [fi.gradient(x) for fi in self.functions]
            self.full_gradient_at_iterate =  np.sum(self.list_stored_gradients) 
            self.data_passes = [1]     
        else:
            self.list_stored_gradients = [x*0.]*len(self.functions)
            self.full_gradient_at_iterate =  x*0         
            self.data_passes=[]   
        self.stoch_grad_at_iterate = x * 0.0 # for CIL/SIRF compatibility
        self.stochastic_grad_difference = x * 0.0 # for CIL/SIRF compatibility
        self.set_up_done= True 


           