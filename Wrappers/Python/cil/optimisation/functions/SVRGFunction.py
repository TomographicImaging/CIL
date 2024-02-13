# -*- coding: utf-8 -*-
#  Copyright 2023 United Kingdom Research and Innovation
#  Copyright 2023 The University of Manchester
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


from .ApproximateGradientSumFunction import ApproximateGradientSumFunction
import numpy as np
import numbers


class SVRGFunction(ApproximateGradientSumFunction):

    """
    #TODO: docstrings 
    A class representing a function for Stochastic Variance Reduced Gradient (SVRG) approximation. #TODO: REference

    Parameters
    ----------
    functions : list
        A list of functions to optimize.
    sampler : callable or None, optional
         A callable function to select the next function, see e.g. optimisation.utilities.sampler 
    update_frequency : int or None, optional
        The frequency of updating the full gradient.
    store_gradients : bool, default: `False`
        Flag indicating whether to store gradients for each function.

    """

    def __init__(self, functions, sampler=None, update_frequency=None, store_gradients=False):
        super(SVRGFunction, self).__init__(functions, sampler)

        # update_frequency for SVRG
        self.update_frequency = update_frequency
        # default update frequency for SVRG is 2*n (convex cases), see  "Accelerating Stochastic Gradient Descent using Predictive Variance Reduction" #TODO: reference
        if self.update_frequency is None:
            self.update_frequency = 2*self.num_functions

        # compute and store the gradient of each function in the finite sum
        self.store_gradients = store_gradients

        self._svrg_iter_number = 0

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
                'The sampler has outputted an index larger than the number of functions to sample from. Please ensure your sampler samples from {1,2,...,len(functions)} only.')

        if isinstance(self.function_num, numbers.Number):

            if self._svrg_iter_number == 0:
                self._stochastic_grad_difference = x.geometry.allocate(0)

                return self._update_full_gradient_and_return(x, out=out)

            elif (np.isinf(self.update_frequency) == False and (self._svrg_iter_number % (self.update_frequency)) == 0):

                return self._update_full_gradient_and_return(x, out=out)

            else:

                return self.approximate_gradient(x, self.function_num, out=out)

        raise ValueError("Batch gradient is not yet implemented")

    def approximate_gradient(self, x, function_num, out=None):

        self._svrg_iter_number += 1

        self.stoch_grad_at_iterate = self.functions[function_num].gradient(x)

        if self.store_gradients is True:
            self._stochastic_grad_difference = self.stoch_grad_at_iterate.sapyb(
                1., self._list_stored_gradients[function_num], -1.)
        else:
            self._stochastic_grad_difference = self.stoch_grad_at_iterate.sapyb(
                1., self.functions[function_num].gradient(self.snapshot), -1.)

        # update the data passes
        try:
            self.data_passes.append(
                self.data_passes[-1] + 1./self.num_functions)
        except IndexError:
            self.data_passes.append(1./self.num_functions)

        # full gradient is added to the stochastic grad difference
        if out is None:
            out = self._stochastic_grad_difference.sapyb(
                self.num_functions, self._full_gradient_at_snapshot, 1.)
        else:
            self._stochastic_grad_difference.sapyb(
                self.num_functions, self._full_gradient_at_snapshot, 1., out=out)

        return out

    def _update_full_gradient_and_return(self, x, out=None):
        """
        Updates the memory for full gradient computation. If :code:`store_gradients==True`, the gradient of all functions is computed and stored.
        """

        self._svrg_iter_number += 1

        if self.store_gradients is True:
            self._list_stored_gradients = [
                fi.gradient(x) for fi in self.functions]
            self._full_gradient_at_snapshot = np.sum(
                self._list_stored_gradients)

        else:
            self._full_gradient_at_snapshot = self.full_gradient(x)
            self.snapshot = x.copy()

        try:
            self.data_passes.append(self.data_passes[-1] + 1.0)
        except IndexError:
            self.data_passes.append(1.0)

        if out is None:
            out = self._stochastic_grad_difference.sapyb(
                self.num_functions, self._full_gradient_at_snapshot, 1.)
        else:
            self._stochastic_grad_difference.sapyb(
                self.num_functions, self._full_gradient_at_snapshot, 1., out=out)

        return out


class LSVRGFunction(SVRGFunction):
    # TODO: docstrings
    """""
    A class representing a function for Loopless Stochastic Variance Reduced Gradient (SVRG) approximation. #TODO: REference

    Parameters
    ----------
    functions : list
        A list of functions to optimize.
    sampler : callable or None, optional
        A callable function to select the next function, see e.g. optimisation.utilities.sampler 
    update_prob : float or None, optional
        The probability of updating the full gradient in loopless SVRG.
    store_gradients : bool, optional
        Flag indicating whether to store gradients for each function.

    """

    def __init__(self, functions, sampler=None, update_prob=None, store_gradients=False, seed=None):

        super(LSVRGFunction, self).__init__(
            functions, sampler=sampler, store_gradients=store_gradients)

        # update frequency based on probability
        self.update_prob = update_prob
        # default update_prob for Loopless SVRG
        if self.update_prob is None:
            self.update_prob = 1./self.num_functions

        # randomness
        self.generator = np.random.default_rng(seed=seed)

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
                'The sampler has outputted an index larger than the number of functions to sample from. Please ensure your sampler samples from {1,2,...,len(functions)} only.')

        if isinstance(self.function_num, numbers.Number):

            if self._svrg_iter_number == 0:
                self._stochastic_grad_difference = x.geometry.allocate(0)
                return self._update_full_gradient_and_return(x, out=out)

            elif self.generator.uniform() < self.update_prob:

                return self._update_full_gradient_and_return(x, out=out)

            else:

                return self.approximate_gradient(x, self.function_num, out=out)

        raise ValueError("Batch gradient is not yet implemented")
