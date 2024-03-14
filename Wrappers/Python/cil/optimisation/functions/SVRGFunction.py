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
    A class representing a function for Stochastic Variance Reduced Gradient (SVRG) approximation. 

    Parameters
    ----------
    functions : list
        A list of functions to optimize.
    sampler : callable or None, optional
         A callable function to select the next function, see e.g. optimisation.utilities.sampler 
    update_frequency : int or None, optional
        The frequency of updating the full gradient. The default is 2*len(functions)
    store_gradients : bool, default: `False`
        Flag indicating whether to store gradients for each function.

    Reference
    ---------
    Johnson, R. and Zhang, T., 2013. Accelerating stochastic gradient descent using predictive variance reduction. Advances in neural information processing systems, 26.
    
    """

    def __init__(self, functions, sampler=None, update_frequency=None, store_gradients=False):
        super(SVRGFunction, self).__init__(functions, sampler)

        # update_frequency for SVRG
        self.update_frequency = update_frequency
    
        if self.update_frequency is None:
            self.update_frequency = 2*self.num_functions

        # compute and store the gradient of each function in the finite sum
        self.store_gradients = store_gradients

        self._svrg_iter_number = 0

    def gradient(self, x, out=None):
        """ Selects a random function using the `sampler` and then calls the approximate gradient at :code:`x` or calculates a full gradient depending on the update frequency

        Parameters
        ----------
        x : DataContainer
        out: return DataContainer, if `None` a new DataContainer is returned, default `None`.

        Returns
        --------
        DataContainer
            the value of the approximate gradient of the sum function at :code:`x` 
        """


    

        if (np.isinf(self.update_frequency) == False and (self._svrg_iter_number % (self.update_frequency)) == 0):

            return self._update_full_gradient_and_return(x, out=out)

        else:

            self.function_num = self.sampler.next()

            if not isinstance(self.function_num, numbers.Number):
                raise ValueError("Batch gradient is not yet implemented")
            if self.function_num > self.num_functions:
                raise IndexError(
                    'The sampler has outputted an index larger than the number of functions to sample from. Please ensure your sampler samples from {0,1,...,len(functions)-1} only.')

            return self.approximate_gradient(x, self.function_num, out=out)

        

    def approximate_gradient(self, x, function_num, out=None):
        """ Computes the approximate gradient for each selected function at :code:`x` given a `function_number` in {0,...,len(functions)-1}.
        
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
    

        self._svrg_iter_number += 1

        self.stoch_grad_at_iterate = self.functions[function_num].gradient(x)

        if self.store_gradients is True:
            self._stochastic_grad_difference = self.stoch_grad_at_iterate.sapyb(
                1., self._list_stored_gradients[function_num], -1.)
        else:
            self._stochastic_grad_difference = self.stoch_grad_at_iterate.sapyb(
                1., self.functions[function_num].gradient(self.snapshot), -1.)

        self._update_data_passes_indices([function_num])

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
        
        Parameters
        ----------
        x : DataContainer
        out: return DataContainer, if `None` a new DataContainer is returned, default `None`.

        Returns
        --------
        DataContainer
            the value of the approximate gradient of the sum function at :code:`x` given a `function_number` in {0,...,len(functions)-1}
        """

        self._svrg_iter_number += 1

        if self.store_gradients is True:
            self._list_stored_gradients = [
                fi.gradient(x) for fi in self.functions]
            self._full_gradient_at_snapshot = sum(
                self._list_stored_gradients, start=0*x)

            self._full_gradient_at_snapshot = self.full_gradient(x)
            self.snapshot = x.copy()

        self._update_data_passes_indices(list(range(self.num_functions)))

        if out is None:
            out = self._full_gradient_at_snapshot
        else:
            out.fill( self._full_gradient_at_snapshot)

        return out


class LSVRGFunction(SVRGFunction):
    """""
    A class representing a function for Loopless Stochastic Variance Reduced Gradient (SVRG) approximation. 

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
        
    Reference
    ---------
    D. Kovalev et al., “Don’t jump through hoops and remove those loops: SVRG and Katyusha are better without the outer loop,” in Algo Learn Theo, PMLR, 2020.

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
        """ Selects a random function using the `sampler` and then calls the approximate gradient at :code:`x` or calculates a full gradient depending on the update probability 

        Parameters
        ----------
        x : DataContainer
        out: return DataContainer, if `None` a new DataContainer is returned, default `None`.

        Returns
        --------
        DataContainer
            the value of the approximate gradient of the sum function at :code:`x`
        """

        if self._svrg_iter_number == 0 or self.generator.uniform() < self.update_prob:

            return self._update_full_gradient_and_return(x, out=out)

        else:
            
            self.function_num = self.sampler.next()
            if not isinstance(self.function_num, numbers.Number):
                raise ValueError("Batch gradient is not yet implemented")
            if self.function_num > self.num_functions:
                raise IndexError(
                    'The sampler has outputted an index larger than the number of functions to sample from. Please ensure your sampler samples from {0,1,...,len(functions)-1} only.')

            return self.approximate_gradient(x, self.function_num, out=out)

        
