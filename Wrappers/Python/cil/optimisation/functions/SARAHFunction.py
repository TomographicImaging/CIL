#  Copyright 2025 United Kingdom Research and Innovation
#  Copyright 2025 The University of Manchester
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
# - CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt

from .ApproximateGradientSumFunction import ApproximateGradientSumFunction
import numbers


class SARAHFunction(ApproximateGradientSumFunction):

    r"""
    The StochAstic Recursive grAdient algoritHm (SARAH) function calculates the approximate gradient of :math:`\sum_{i=0}^{n-1}f_i`. Every `update_frequency` iterations a full gradient is calculated. On the intermediate iterations an index :math:`i_k` is sampled and the estimator is updated *recursively* from the one returned on the previous iteration:

    .. math ::
        v_k = n*\nabla f_{i_k}(x_k) - n*\nabla f_{i_k}(x_{k-1}) + v_{k-1},

    where :math:`x_{k-1}` is the point `gradient` was called with on the previous iteration and :math:`v_{k-1}` is the value it returned. 

    Note
    -----
    Compared with the literature, we multiply by :math:`n`, the number of functions, so that we return an approximate gradient of the whole sum function and not an average gradient.

    Note
    -----
    Each intermediate iteration evaluates two stochastic gradients, at :math:`x_k` and :math:`x_{k-1}`, but both use the data of a single :math:`f_{i_k}`, so `data_passes` counts :math:`1/n`. 

    Note
    -----
    Algorithm 1 of the reference restarts each outer loop from a uniformly random inner iterate; to reduce memory requirements,  this implementation continues from the last one.

    Note
    ----
    The additional memory requirement is 3 times the image size (the running gradient estimator, the previous iterate and one lot of intermediary calculations).

    Reference
    ---------
    Nguyen, L.M., Liu, J., Scheinberg, K. and Takáč, M., 2017. SARAH: A novel method for machine learning problems using stochastic recursive gradient. Proceedings of the 34th International Conference on Machine Learning, PMLR 70:2613-2621. https://proceedings.mlr.press/v70/nguyen17b.html

    Parameters
    ----------
    functions : `list`  of functions
        A list of functions: :code:`[f_{0}, f_{1}, ..., f_{n-1}]`. Each function is assumed to be smooth with an implemented :func:`~Function.gradient` method. All functions must have the same domain. The number of functions must be strictly greater than 1.
    sampler: An instance of a CIL Sampler class ( :meth:`~optimisation.utilities.sampler`) or of another class which has a `next` function implemented to output integers in {0, 1, ..., n-1}.
        This sampler is called each time gradient is called and  sets the internal `function_num` passed to the `approximate_gradient` function.  Default is `Sampler.random_with_replacement(len(functions))`.
    update_frequency : positive int or None, optional
        The interval for recomputing the full gradient and restarting the recursion, called the inner loop size :math:`m` in the reference. The default is 2*len(functions), chosen to match the default of :class:`~cil.optimisation.functions.SVRGFunction` so that the two are directly comparable. At :math:`m=1` SARAH reduces to gradient descent.

    """

    def __init__(self, functions, sampler=None, update_frequency=None):
        super(SARAHFunction, self).__init__(functions, sampler)

        if update_frequency is None:
            update_frequency = 2*self.num_functions

        if not isinstance(update_frequency, numbers.Integral) or update_frequency < 1:
            raise ValueError(
                f"`update_frequency` must be a positive integer, got {update_frequency}.")

        self.update_frequency = update_frequency

        self._sarah_iter_number = 0

        #  The running estimator v_{k-1}, and the point x_{k-1} it was calculated at.
        self._gradient_estimator = None
        self._previous_iterate = None

        self._stoch_grad = None

    def gradient(self, x, out=None):
        r""" Selects a random function using the `sampler` and then calls the approximate gradient at :code:`x` or calculates a full gradient depending on the update frequency

        Parameters
        ----------
        x : DataContainer (e.g. ImageData object)
        out: return DataContainer, if `None` a new DataContainer is returned, default `None`.

        Returns
        --------
        DataContainer (e.g. ImageData object)
            the value of the approximate gradient of the sum function at :code:`x`
        """

        #  For SARAH, every `update_frequency` a full gradient is calculated, else the estimator is updated recursively.
        if (self._sarah_iter_number % self.update_frequency) == 0:

            return self._update_full_gradient_and_return(x, out=out)

        else:

            self.function_num = self.sampler.next()
            if not isinstance(self.function_num, numbers.Number):
                raise ValueError("Batch gradient is not yet implemented")
            if self.function_num >= self.num_functions or self.function_num < 0:
                raise IndexError(
                    f"The sampler has produced the index {self.function_num} which does not match the expected range of available functions to sample from. Please ensure your sampler only selects from [0,1,...,len(functions)-1] ")
            return self.approximate_gradient(x, self.function_num, out=out)

    def approximate_gradient(self, x, function_num, out=None):
        r""" Updates the recursive gradient estimator using the gradient of the selected function, indexed by :math:`i_k`, the `function_number` in {0,...,len(functions)-1}, evaluated at both the current iterate :math:`x_k` and the previous iterate :math:`x_{k-1}`

        .. math ::
            v_k = n*\nabla f_{i_k}(x_k) - n*\nabla f_{i_k}(x_{k-1}) + v_{k-1}

        Note
        -----
        Compared with the literature, we multiply by :math:`n`, the number of functions, so that we return an approximate gradient of the whole sum function and not an average gradient.

        Parameters
        ----------
        x : DataContainer (e.g. ImageData object)
        out: return DataContainer, if `None` a new DataContainer is returned, default `None`.
        function_num: `int`
            Between 0 and n-1, where n is the number of functions in the list
        Returns
        --------
        DataContainer (e.g. ImageData object)
            the value of the approximate gradient of the sum function at :code:`x` given a `function_number` in {0,...,len(functions)-1}
        """

        self._sarah_iter_number += 1

        self._stoch_grad = self.functions[function_num].gradient(
            x, out=self._stoch_grad)
        self._gradient_estimator.sapyb(
            1., self._stoch_grad, self.num_functions, out=self._gradient_estimator)

        self._stoch_grad = self.functions[function_num].gradient(
            self._previous_iterate, out=self._stoch_grad)
        self._gradient_estimator.sapyb(
            1., self._stoch_grad, -self.num_functions, out=self._gradient_estimator)

        #  Only the data of the function indexed by `function_num` was used this iteration, even though
        #  its gradient was evaluated at two points. 
        self._update_data_passes_indices([function_num])


        self._previous_iterate.fill(x)


        if out is None:
            return self._gradient_estimator.copy()

        out.fill(self._gradient_estimator)
        return out

    def _update_full_gradient_and_return(self, x, out=None):
        r"""
        Restarts the recursion at the point :math:`x`, setting the gradient estimator to the full gradient :math:`\nabla \sum_{i=0}^{n-1}f_i(x)` and saving :math:`x` as the previous iterate. The function returns the full gradient as the gradient calculation.

        Parameters
        ----------
        x : DataContainer (e.g. ImageData object)
        out: return DataContainer, if `None` a new DataContainer is returned, default `None`.

        Returns
        --------
        DataContainer (e.g. ImageData object)
            the value of the gradient of the sum function at :code:`x`
        """

        self._sarah_iter_number += 1

        self._gradient_estimator = self.full_gradient(
            x, out=self._gradient_estimator)

        if self._previous_iterate is None:
            self._previous_iterate = x.copy()
        else:
            self._previous_iterate.fill(x)

        # In this iteration all functions in the sum were used to update the gradient
        self._update_data_passes_indices(list(range(self.num_functions)))


        if out is None:
            return self._gradient_estimator.copy()

        out.fill(self._gradient_estimator)
        return out
