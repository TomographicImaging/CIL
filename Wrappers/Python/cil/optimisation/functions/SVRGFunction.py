#   Copyright 2024 United Kingdom Research and Innovation
#   Copyright 2024 The University of Manchester
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
#  Authors:
#  - CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt
#  - Daniel Deidda (National Physical Laboratory, UK)
#  - Claire Delplancke (Electricite de France, Research and Development)
#  - Ashley Gillman (Australian e-Health Res. Ctr., CSIRO, Brisbane, Queensland, Australia)
#  - Zeljko Kereta (Department of Computer Science, University College London, UK)
#  - Evgueni Ovtchinnikov (STFC - UKRI)
#  - Georg Schramm (Department of Imaging and Pathology, Division of Nuclear Medicine, KU Leuven, Leuven, Belgium)


from .ApproximateGradientSumFunction import ApproximateGradientSumFunction
import numpy as np
import numbers


class SVRGFunction(ApproximateGradientSumFunction):

    r"""
    The Stochastic Variance Reduced Gradient (SVRG) function calculates the approximate gradient of :math:`\sum_{i=1}^{n-1}f_i`.  For this approximation, every `snapshot_update_interval` number of iterations, a full gradient calculation is made at this "snapshot" point. Intermediate gradient calculations update this snapshot by taking a index :math:`i_k` and calculating the gradient of :math:`f_{i_k}`s at the current iterate and the snapshot, updating the approximate gradient to be:

    .. math ::
        n*\nabla f_{i_k}(x_k) - n*\nabla f_{i_k}(\tilde{x}) + \nabla \sum_{i=0}^{n-1}f_i(\tilde{x}),

    where :math:`\tilde{x}` is the latest "snapshot" point and :math:`x_k` is the value at the current iteration. 

    Note
    -----
    Compared with the literature, we multiply by :math:`n`, the number of functions, so that we return an approximate gradient of the whole sum function and not an average gradient. 

    Note
    ----
    In the case where `store_gradients=False` the memory requirements are 4 times the image size (1 stored full gradient at the "snapshot", one stored "snapshot" point and two lots of intermediary calculations). Alternatively, if  `store_gradients=True`  the memory requirement is `n+4` (`n` gradients at the snapshot for each function in the sum, one stored full gradient at the "snapshot", one stored "snapshot" point and two lots of intermediary calculations).
    
    Reference
    ---------
    Johnson, R. and Zhang, T., 2013. Accelerating stochastic gradient descent using predictive variance reduction. Advances in neural information processing systems, 26.https://proceedings.neurips.cc/paper_files/paper/2013/file/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Paper.pdf


    Parameters
    ----------
    functions : `list`  of functions
        A list of functions: :code:`[f_{0}, f_{1}, ..., f_{n-1}]`. Each function is assumed to be smooth with an implemented :func:`~Function.gradient` method. All functions must have the same domain. The number of functions must be strictly greater than 1. 
    sampler: An instance of a CIL Sampler class ( :meth:`~optimisation.utilities.sampler`) or of another class which has a `next` function implemented to output integers in {0, 1, ..., n-1}.
        This sampler is called each time gradient is called and  sets the internal `function_num` passed to the `approximate_gradient` function.  Default is `Sampler.random_with_replacement(len(functions))`. 
    snapshot_update_interval : positive int or None, optional
        The interval for updating the full gradient (taking a snapshot). The default is 2*len(functions) so a "snapshot" is taken every 2*len(functions) iterations. If the user passes `0` then no full gradient snapshots will be taken. 
    store_gradients : bool, default: `False`
        Flag indicating whether to store an update a list of gradients for each function :math:`f_i` or just to store the snapshot point :math:` \tilde{x}` and its gradient :math:`\nabla \sum_{i=0}^{n-1}f_i(\tilde{x})`.


    """

    def __init__(self, functions, sampler=None, snapshot_update_interval=None, store_gradients=False):
        super(SVRGFunction, self).__init__(functions, sampler)

        #  snapshot_update_interval for SVRG
        self.snapshot_update_interval = snapshot_update_interval
    
        if snapshot_update_interval is None:
            self.snapshot_update_interval = 2*self.num_functions
        self.store_gradients = store_gradients

        self._svrg_iter_number = 0

        self._full_gradient_at_snapshot = None
        self._list_stored_gradients = None

        self.stoch_grad_at_iterate = None
        self._stochastic_grad_difference = None
        
        self.snapshot = None

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

        #  For SVRG, every `snapshot_update_interval` a full gradient step is calculated, else an approximate gradient is taken.
        if ( (self.snapshot_update_interval != 0) and (self._svrg_iter_number % (self.snapshot_update_interval)) == 0):

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
        r""" Calculates the stochastic gradient at the point :math:`x` by using the gradient of the selected function, indexed by :math:`i_k`, the `function_number` in {0,...,len(functions)-1}, and the full gradient at the snapshot :math:`\tilde{x}`
        
        .. math ::
            n*\nabla f_{i_k}(x_k) - n*\nabla f_{i_k}(\tilde{x}) + \nabla \sum_{i=0}^{n-1}f_i(\tilde{x})

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

        self._svrg_iter_number += 1

        self.stoch_grad_at_iterate = self.functions[function_num].gradient(x, out=self.stoch_grad_at_iterate)

        if self.store_gradients is True:
            self._stochastic_grad_difference = self.stoch_grad_at_iterate.sapyb(
                1., self._list_stored_gradients[function_num], -1., out=self._stochastic_grad_difference)
        else:
            self._stochastic_grad_difference = self.stoch_grad_at_iterate.sapyb(
                1., self.functions[function_num].gradient(self.snapshot), -1., out=self._stochastic_grad_difference)

        self._update_data_passes_indices([function_num])

        out = self._stochastic_grad_difference.sapyb(
            self.num_functions, self._full_gradient_at_snapshot, 1., out=out)

        return out

    def _update_full_gradient_and_return(self, x, out=None):
        """
        Takes a "snapshot" at the point :math:`x`, saving both the point :math:` \tilde{x}=x` and its gradient :math:`\sum_{i=0}^{n-1}f_i{\tilde{x}}`. The function returns :math:`\sum_{i=0}^{n-1}f_i{\tilde{x}}` as the gradient calculation. If :code:`store_gradients==True`, the gradient of all the :math:`f_i`s is computed and stored at the "snapshot"..

        Parameters
        ----------
        Takes a "snapshot" at the point :math:`x`. The function returns :math:`\sum_{i=0}^{n-1}f_i{\tilde{x}}` as the gradient calculation. If :code:`store_gradients==True`, the gradient of all the :math:`f_i`s is stored, otherwise only  the sum of the gradients and the snapshot point :math:` \tilde{x}=x` are stored.
        out: return DataContainer, if `None` a new DataContainer is returned, default `None`.

        Returns
        --------
        DataContainer (e.g. ImageData object)
            the value of the approximate gradient of the sum function at :code:`x` given a `function_number` in {0,...,len(functions)-1}
        """

        self._svrg_iter_number += 1

        if self.store_gradients is True:
            if self._list_stored_gradients is None:
                # Save the gradient of each individual f_i and the gradient of the full sum at the point x.
                self._list_stored_gradients = [
                    fi.gradient(x) for fi in self.functions]
                self._full_gradient_at_snapshot = sum(
                    self._list_stored_gradients, start=0*x)
            else:
                for i, fi in enumerate(self.functions):
                    fi.gradient(x, out=self._list_stored_gradients[i])

                self._full_gradient_at_snapshot.fill(
                    sum(self._list_stored_gradients, start=0*x))
                self._full_gradient_at_snapshot *= 0

                for i, el in enumerate(self._list_stored_gradients):
                    self._full_gradient_at_snapshot += el

        else:
            # Save the snapshot point and the gradient of the full sum at the point x.
            self._full_gradient_at_snapshot = self.full_gradient(
                x, out=self._full_gradient_at_snapshot)

        if self.snapshot is None:
            self.snapshot = x.copy()

        self.snapshot.fill(x)

        # In this iteration all functions in the sum were used to update the gradient
        self._update_data_passes_indices(list(range(self.num_functions)))

        # Return the gradient of the full sum at the snapshot.
        if out is None:
            out = self._full_gradient_at_snapshot
        else:
            out.fill(self._full_gradient_at_snapshot)

        return out


class LSVRGFunction(SVRGFunction):
    """""
    A class representing a function for Loopless Stochastic Variance Reduced Gradient (SVRG) approximation. This is similar to SVRG, except the full gradient at a "snapshot"  is calculated at random intervals rather than at fixed numbers of iterations. 


    Reference
    ----------

    Kovalev, D., Horváth, S. &; Richtárik, P.. (2020). Don’t Jump Through Hoops and Remove Those Loops:  SVRG and Katyusha are Better Without the Outer Loop. Proceedings of the 31st International Conference  on Algorithmic Learning Theory, in Proceedings of Machine Learning Research 117:451-467 Available from https://proceedings.mlr.press/v117/kovalev20a.html.



    Parameters
    ----------
     functions : `list`  of functions
        A list of functions: :code:`[f_{0}, f_{1}, ..., f_{n-1}]`. Each function is assumed to be smooth with an implemented :func:`~Function.gradient` method. All functions must have the same domain. The number of functions `n` must be strictly greater than 1. 
    sampler: An instance of a CIL Sampler class ( :meth:`~optimisation.utilities.sampler`) or of another class which has a `next` function implemented to output integers in {0,...,n-1}.
        This sampler is called each time gradient is called and  sets the internal `function_num` passed to the `approximate_gradient` function.  Default is `Sampler.random_with_replacement(len(functions))`. 
    snapshot_update_probability: positive float, default: 1/n
        The probability of updating the full gradient (taking a snapshot) at each iteration. The default is :math:`1./n` so, in expectation, a snapshot will be taken every :math:`n` iterations. 
    store_gradients : bool, default: `False`
        Flag indicating whether to store an update a list of gradients for each function :math:`f_i` or just to store the snapshot point :math:` \tilde{x}` and it's gradient :math:`\nabla \sum_{i=0}^{n-1}f_i(\tilde{x})`.


    Note
    ----
    In the case where `store_gradients=False` the memory requirements are 4 times the image size (1 stored full gradient at the "snapshot", one stored "snapshot" point and two lots of intermediary calculations). Alternatively, if  `store_gradients=True`  the memory requirement is `n+4` (`n` gradients at the snapshot for each function in the sum, one stored full gradient at the "snapshot", one stored "snapshot" point and two lots of intermediary calculations).

    """

    def __init__(self, functions, sampler=None, snapshot_update_probability=None, store_gradients=False, seed=None):

        super(LSVRGFunction, self).__init__(
            functions, sampler=sampler, store_gradients=store_gradients)

        # Update frequency based on probability.
        self.snapshot_update_probability = snapshot_update_probability
        #  Default snapshot_update_probability for Loopless SVRG
        if self.snapshot_update_probability is None:
            self.snapshot_update_probability = 1./self.num_functions

        #  The random generator used to decide if the gradient calculation is a full gradient or an approximate gradient
        self.generator = np.random.default_rng(seed=seed)

    def gradient(self, x, out=None):
        """ Selects a random function using the `sampler` and then calls the approximate gradient at :code:`x` or calculates a full gradient depending on the update probability.

        Parameters
        ----------
        x : DataContainer (e.g. ImageData objects)
        out: return DataContainer, if `None` a new DataContainer is returned, default `None`.

        Returns
        --------
        DataContainer (e.g. ImageData object)
            the value of the approximate gradient of the sum function at :code:`x`
        """

        if self._svrg_iter_number == 0 or self.generator.uniform() < self.snapshot_update_probability:

            return self._update_full_gradient_and_return(x, out=out)

        else:

            self.function_num = self.sampler.next()
            if not isinstance(self.function_num, numbers.Number):
                raise ValueError("Batch gradient is not yet implemented")
            if self.function_num >= self.num_functions or self.function_num < 0:
                raise IndexError(
                    f"The sampler has produced the index {self.function_num} which does not match the expected range of available functions to sample from. Please ensure your sampler only selects from [0,1,...,len(functions)-1] ")
            return self.approximate_gradient(x, self.function_num, out=out)
