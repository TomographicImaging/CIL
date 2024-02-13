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
import numpy as np


class SAGFunction(ApproximateGradientSumFunction):

    """
    Stochastic average gradient (SAG) function, a child class of `ApproximateGradientSumFunction`, which defines from a list of functions, :math:`{f_1,...,f_n}` a `SumFunction`, :math:`f_1+...+f_n` where each time the `gradient` is called, the `sampler` provides an index, :math:`i \in {1,...,n}` 
   and the gradient function returns the approximate gradient. This can be used with the `cil.optimisation.algorithms` algorithm GD to give a stochastic optimisation method.  
   By incorporating a memory of previous gradient values the SAG method can achieve a faster convergence rate than black-box stochastic gradient methods. 

    Parameters:
    -----------
    functions : `list`  of functions
                A list of functions: :code:`[F_{1}, F_{2}, ..., F_{n}]`. Each function is assumed to be smooth function with an implemented :func:`~Function.gradient` method. Each function must have the same domain. The number of functions must be strictly greater than 1. 
    sampler: An instance of one of the :meth:`~optimisation.utilities.sampler` classes which has a `next` function implemented and a `num_indices` property.
        This sampler is called each time gradient is called and  sets the internal `function_num` passed to the `approximate_gradient` function.  The `num_indices` must match the number of functions provided. Default is `Sampler.random_with_replacement(len(functions))`. 
    warm_start: Boolean, default : False
        If `warm_start` is True then when the gradient is first called, the full gradient for each function is computed and stored. If False, the gradients are initialised with zeros. 

    References
    ----------
    Schmidt, M., Le Roux, N. and Bach, F., 2017. Minimizing finite sums with the stochastic average gradient. Mathematical Programming, 162, pp.83-112. https://doi.org/10.1007/s10107-016-1030-6

    """

    def __init__(self, functions,  sampler=None):
        self.list_stored_gradients = None
        self.full_gradient_at_iterate = None
        self._warm_start_data_pass=False

        super(SAGFunction, self).__init__(functions, sampler)

    def warm_start(self, initial):
        """A function to warm start SAG or SAGA algorithms by initialising all the gradients at an initial point.
        
        Parameters
        ----------
        initial: DataContainer,
            The initial point to warmstart the calculation 
            
        Example
        --------
        >>> stochastic_objective= SAGFunction(list_of_functions)
        >>> stochastic_objective.warm_start(initial_point)
        >>> sag_algorithm=GD(initial=initial_point,  objective_function=stochastic_objective)
        """
        self.list_stored_gradients = [
            fi.gradient(initial) for fi in self.functions]
        self.full_gradient_at_iterate = np.sum(self.list_stored_gradients)
        self._warm_start_data_pass=True
        

    def approximate_gradient(self, x, function_num,  out=None):
        """ Returns the gradient of the selected function or batch of functions at :code:`x`. 
            The function_num is selected using the sampler. 

        Parameters
        ----------
        x: element in the domain of the `functions`

        function_num: `int` 
            Between 1 and the number of functions in the list  

        """
        if self.list_stored_gradients is None:
            self.list_stored_gradients = [
                x.geometry.allocate(0) for fi in self.functions]
            self.full_gradient_at_iterate = x.geometry.allocate(0) 
            

        self.stoch_grad_at_iterate = self.functions[function_num].gradient(x)

        if self._warm_start_data_pass:
            self._update_data_passes(1.0+1./self.num_functions)
            self._warm_start_data_pass=False 
        else:
            self._update_data_passes(1./self.num_functions)

        self.stochastic_grad_difference = self.stoch_grad_at_iterate.sapyb(
            1., self.list_stored_gradients[function_num], -1.)

   
        out =self._update_approx_gradient(out)

        self.list_stored_gradients[function_num].fill(
            self.stoch_grad_at_iterate)
        self.full_gradient_at_iterate.sapyb(
            1., self.stochastic_grad_difference, 1., out=self.full_gradient_at_iterate)

        return out
    
    def _update_approx_gradient(self, out):
        """Internal function used to differentiate between the SAG and SAGA calculations"""
        if out is None:
            # due to the convention that we follow: without the 1/n factor
            out = self.stochastic_grad_difference.sapyb(
                1., self.full_gradient_at_iterate, 1.)
        else:
            # due to the convention that we follow: without the 1/n factor
            self.stochastic_grad_difference.sapyb(
                1., self.full_gradient_at_iterate, 1., out=out)

        return out 

class SAGAFunction(SAGFunction):

    """
    An accelerated version of the stochastic average gradient function, a child class of `ApproximateGradientSumFunction`, which defines from a list of functions, :math:`{f_1,...,f_n}` a `SumFunction`, :math:`f_1+...+f_n` where each time the `gradient` is called, the `sampler` provides an index, :math:`i \in {1,...,n}` 
   and the gradient function returns the approximate gradient.  This can be used with the `cil.optimisation.algorithms` algorithm GD to give a stochastic optimisation method. 
   SAGA improves on the theory behind SAG and SVRG, with better theoretical convergence rates.

   Parameters:
    -----------
    functions : `list`  of functions
                A list of functions: :code:`[F_{1}, F_{2}, ..., F_{n}]`. Each function is assumed to be smooth function with an implemented :func:`~Function.gradient` method. Each function must have the same domain. The number of functions must be strictly greater than 1. 
    sampler: An instance of one of the :meth:`~optimisation.utilities.sampler` classes which has a `next` function implemented and a `num_indices` property.
        This sampler is called each time gradient is called and  sets the internal `function_num` passed to the `approximate_gradient` function.  The `num_indices` must match the number of functions provided. Default is `Sampler.random_with_replacement(len(functions))`. 
    warm_start: Boolean, default : False
        If `warm_start` is True then when the gradient is first called, the full gradient for each function is computed and stored. If False, the gradients are initialised with zeros. 

    References
    ----------
    Defazio, A., Bach, F. and Lacoste-Julien, S., 2014. SAGA: A fast incremental gradient method with support for non-strongly convex composite objectives. Advances in neural information processing systems, 27. https://proceedings.neurips.cc/paper_files/paper/2014/file/ede7e2b6d13a41ddf9f4bdef84fdc737-Paper.pdf
    """

    def __init__(self, functions,  sampler=None):
        super(SAGAFunction, self).__init__(functions, sampler)


    def _update_approx_gradient(self, out):
            
        if out is None:
            # due to the convention that we follow: without the 1/n factor
            out= self.stochastic_grad_difference.sapyb(
                self.num_functions, self.full_gradient_at_iterate, 1.)
        else:
            # due to the convention that we follow: without the 1/n factor
            self.stochastic_grad_difference.sapyb(
                self.num_functions, self.full_gradient_at_iterate, 1., out=out)

        return out 