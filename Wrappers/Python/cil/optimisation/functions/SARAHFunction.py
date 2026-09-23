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
import numpy as np


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
    ----
    The additional memory requirement is 3 times the image size (the running gradient estimator, the previous iterate and one lot of intermediary calculations).

    Note
    ----
    Convergence theory is available for the smooth case, i.e. for use with :class:`~cil.optimisation.algorithms.GD`. 
    - With :class:`~cil.optimisation.algorithms.ISTA`, this is the :math:`\gamma_t = 1` case of Algorithm 1 of Pham et al., 
    - With :class:`~cil.optimisation.algorithms.FISTA` we are not aware of results in the literature. 

    References
    ----------
    Nguyen, L.M., Liu, J., Scheinberg, K. and Takáč, M., 2017. SARAH: A novel method for machine learning problems using stochastic recursive gradient. Proceedings of the 34th International Conference on Machine Learning, PMLR 70:2613-2621. https://proceedings.mlr.press/v70/nguyen17b.html

    Pham, N.H., Nguyen, L.M., Phan, D.T. and Tran-Dinh, Q., 2020. ProxSARAH: An efficient algorithmic framework for stochastic composite nonconvex optimization. Journal of Machine Learning Research, 21(110):1-48. https://jmlr.org/papers/v21/19-248.html

    Parameters
    ----------
    functions : `list`  of functions
        A list of functions: :code:`[f_{0}, f_{1}, ..., f_{n-1}]`. Each function is assumed to be smooth with an implemented :func:`~Function.gradient` method. All functions must have the same domain. The number of functions must be strictly greater than 1.
    sampler: An instance of a CIL Sampler class ( :meth:`~optimisation.utilities.sampler`) or of another class which has a `next` function implemented to output integers in {0, 1, ..., n-1}.
        This sampler is called each time gradient is called and  sets the internal `function_num` passed to the `approximate_gradient` function.  Default is `Sampler.random_with_replacement(len(functions))`.
    update_frequency : positive int or None, optional
        The interval for recomputing the full gradient and restarting the recursion.  The default is 2*len(functions). 

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


class LSARAHFunction(SARAHFunction):

    r"""
    The LoopLess SARAH (L2S) function calculates the approximate gradient of :math:`\sum_{i=0}^{n-1}f_i`. This is similar to :class:`~cil.optimisation.functions.SARAHFunction`, except the full gradient is calculated at random intervals rather than at a fixed number of iterations. At each iteration, with probability `snapshot_update_probability` a full gradient is calculated and the recursion is restarted, and otherwise an index :math:`i_k` is sampled and the estimator is updated *recursively* from the one returned on the previous iteration:

    .. math ::
        v_k = n*\nabla f_{i_k}(x_k) - n*\nabla f_{i_k}(x_{k-1}) + v_{k-1},

    where :math:`x_{k-1}` is the point `gradient` was called with on the previous iteration and :math:`v_{k-1}` is the value it returned.

    Note
    -----
    Compared with the literature, we multiply by :math:`n`, the number of functions, so that we return an approximate gradient of the whole sum function and not an average gradient.

    Note
    -----
    This implements Algorithm 2 of the reference, which covers the convex and non-convex cases. The strongly convex variant (Algorithm 3 of the reference) additionally steps the iterate back, setting :math:`x_k = x_{k-1}` before taking the snapshot, which cannot be done from within a function because it modifies the state of the algorithm.

    Note
    -----
    There is no convergence theory for the proximal case. The reference covers the smooth problem only, so combining this function with :class:`~cil.optimisation.algorithms.ISTA` or :class:`~cil.optimisation.algorithms.FISTA` is unsupported: Pham et al. analyse proximal SARAH with a fixed inner loop length, and this reference analyses the loopless method without a proximal term, but neither covers the two together. 
    Note
    ----
    The additional memory requirement is 3 times the image size, the same as for :class:`~cil.optimisation.functions.SARAHFunction`.

    Reference
    ---------
    Li, B., Ma, M. and Giannakis, G.B., 2020. On the convergence of SARAH and beyond. Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics, PMLR 108:223-233. https://proceedings.mlr.press/v108/li20a.html

    Parameters
    ----------
    functions : `list`  of functions
        A list of functions: :code:`[f_{0}, f_{1}, ..., f_{n-1}]`. Each function is assumed to be smooth with an implemented :func:`~Function.gradient` method. All functions must have the same domain. The number of functions must be strictly greater than 1.
    sampler: An instance of a CIL Sampler class ( :meth:`~optimisation.utilities.sampler`) or of another class which has a `next` function implemented to output integers in {0, 1, ..., n-1}.
        This sampler is called each time gradient is called and  sets the internal `function_num` passed to the `approximate_gradient` function.  Default is `Sampler.random_with_replacement(len(functions))`.
    snapshot_update_probability: positive float, default: 1/n
        The probability of calculating a full gradient and restarting the recursion at each iteration, written :math:`1/m` in the reference. The default is :math:`1./n` so, in expectation, a full gradient is calculated every :math:`n` iterations, matching both the default of :class:`~cil.optimisation.functions.LSVRGFunction` and the choice :math:`m = \Theta(n)` used for the complexity results of the reference.
    seed: int
        Seed for the random snapshot decisions (generated using numpy.random).

    """

    def __init__(self, functions, sampler=None, snapshot_update_probability=None, seed=None):

        super(LSARAHFunction, self).__init__(functions, sampler=sampler)

        #  The inherited `update_frequency` is unused: the snapshots are decided by `snapshot_update_probability` instead.
        self.snapshot_update_probability = snapshot_update_probability
        #  Default snapshot_update_probability for Loopless SARAH
        if self.snapshot_update_probability is None:
            self.snapshot_update_probability = 1./self.num_functions

        #  The random generator used to decide if the gradient calculation is a full gradient or an approximate gradient
        self.generator = np.random.default_rng(seed=seed)

    def gradient(self, x, out=None):
        r""" Selects a random function using the `sampler` and then calls the approximate gradient at :code:`x` or calculates a full gradient depending on the update probability.

        Parameters
        ----------
        x : DataContainer (e.g. ImageData object)
        out: return DataContainer, if `None` a new DataContainer is returned, default `None`.

        Returns
        --------
        DataContainer (e.g. ImageData object)
            the value of the approximate gradient of the sum function at :code:`x`
        """

        #  The first call must be a full gradient, to initialise the recursion.
        if self._sarah_iter_number == 0 or self.generator.uniform() < self.snapshot_update_probability:

            return self._update_full_gradient_and_return(x, out=out)

        else:

            self.function_num = self.sampler.next()
            if not isinstance(self.function_num, numbers.Number):
                raise ValueError("Batch gradient is not yet implemented")
            if self.function_num >= self.num_functions or self.function_num < 0:
                raise IndexError(
                    f"The sampler has produced the index {self.function_num} which does not match the expected range of available functions to sample from. Please ensure your sampler only selects from [0,1,...,len(functions)-1] ")
            return self.approximate_gradient(x, self.function_num, out=out)
