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
   By incorporating a memory of previous gradient values the SAG method achieves a faster convergence rate than black-box stochastic gradient methods. #TODO:check this 

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

    def __init__(self, functions,  sampler=None, warm_start=False):
        self.set_up_done = False
        self.warm_start = warm_start
        super(SAGFunction, self).__init__(functions, sampler)

    def approximate_gradient(self, x, function_num,  out=None):
        """ Returns the gradient of the selected function or batch of functions at :code:`x`. 
            The function num is selected using the sampler. 

        Parameters
        ----------
        x: element in the domain of the `functions`

        function_num: `int` 
            Between 1 and the number of functions in the list  

        """
        if not self.set_up_done:
            self._set_up(x)

        self.stoch_grad_at_iterate = self.functions[function_num].gradient(x)

        try:
            self.data_passes.append(
                self.data_passes[-1] + 1./self.num_functions)
        except IndexError:
            self.data_passes.append(1./self.num_functions)

        self.stochastic_grad_difference = self.stoch_grad_at_iterate.sapyb(
            1., self.list_stored_gradients[function_num], -1.)
        # flag to return or in-place computation
        should_return = False

        # compute gradient of randomly selected(function_num) function
        if out is None:
            res = x*0.  # for CIL/SIRF compatibility
            # due to the convention that we follow: without the 1/n factor
            self.stochastic_grad_difference.sapyb(
                1., self.full_gradient_at_iterate, 1., out=res)
            should_return = True
        else:
            # due to the convention that we follow: without the 1/n factor
            self.stochastic_grad_difference.sapyb(
                1., self.full_gradient_at_iterate, 1., out=out)
            #  print(out.array)

        self.list_stored_gradients[function_num].fill(
            self.stoch_grad_at_iterate)
        self.full_gradient_at_iterate.sapyb(
            1., self.stochastic_grad_difference, 1., out=self.full_gradient_at_iterate)

        if should_return:
            return res

    def _set_up(self, x):
        r"""Initialize subset gradients :math:`v_{i}` and full gradient that are stored in memory.
        The initial point is 0 by default.
        """
        if self.warm_start:
            self.list_stored_gradients = [
                fi.gradient(x) for fi in self.functions]
            self.full_gradient_at_iterate = np.sum(self.list_stored_gradients)
            self.data_passes = [1]
        else:
            self.list_stored_gradients = [
                x.geometry.allocate(0) for fi in self.functions]

            self.full_gradient_at_iterate = x.geometry.allocate(0)
            self.data_passes = []
        self.set_up_done = True


class SAGAFunction(SAGFunction):

    """
    TODO:
    Stochastic average gradient function, a child class of `ApproximateGradientSumFunction`, which defines from a list of functions, :math:`{f_1,...,f_n}` a `SumFunction`, :math:`f_1+...+f_n` where each time the `gradient` is called, the `sampler` provides an index, :math:`i \in {1,...,n}` 
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

    def __init__(self, functions,  sampler=None, warm_start=False):
        super(SAGAFunction, self).__init__(functions, sampler, warm_start)

    def approximate_gradient(self, x, function_num,  out=None):
        """ Returns the gradient of the selected function or batch of functions at :code:`x`. 
            The function num is selected using the sampler.

        Parameters
        ----------
        x: element in the domain of the `functions`

        function_num: `int` 
            Between 1 and the number of functions in the list  

        """
        if not self.set_up_done:
            self._set_up(x)

        self.stoch_grad_at_iterate = self.functions[function_num].gradient(x)

        try:
            self.data_passes.append(
                self.data_passes[-1] + 1./self.num_functions)
        except IndexError:
            self.data_passes.append(1./self.num_functions)

        self.stochastic_grad_difference = self.stoch_grad_at_iterate.sapyb(
            1., self.list_stored_gradients[function_num], -1.)
        # flag to return or in-place computation
        should_return = False

        # compute gradient of randomly selected(function_num) function
        if out is None:
            res = x*0.  # for CIL/SIRF compatibility
            # due to the convention that we follow: without the 1/n factor
            self.stochastic_grad_difference.sapyb(
                self.num_functions, self.full_gradient_at_iterate, 1., out=res)
            should_return = True
        else:
            # due to the convention that we follow: without the 1/n factor
            self.stochastic_grad_difference.sapyb(
                self.num_functions, self.full_gradient_at_iterate, 1., out=out)

        self.list_stored_gradients[function_num].fill(
            self.stoch_grad_at_iterate)
        self.full_gradient_at_iterate.sapyb(
            1., self.stochastic_grad_difference, 1., out=self.full_gradient_at_iterate)

        if should_return:
            return res
