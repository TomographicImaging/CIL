#  Copyright 2024 United Kingdom Research and Innovation
#  Copyright 2024 The University of Manchester
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
# - Daniel Deidda (National Physical Laboratory, UK)
# - Claire Delplancke (Electricite de France, Research and Development)
# - Ashley Gillman (Australian e-Health Res. Ctr., CSIRO, Brisbane, Queensland, Australia)
# - Zeljko Kereta (Department of Computer Science, University College London, UK)
# - Evgueni Ovtchinnikov (STFC - UKRI)
# - Georg Schramm (Department of Imaging and Pathology, Division of Nuclear Medicine, KU Leuven, Leuven, Belgium)


from .ApproximateGradientSumFunction import ApproximateGradientSumFunction
import numpy as np
import numbers


class SVRGFunction(ApproximateGradientSumFunction):

    r"""
    A class representing a function for Stochastic Variance Reduced Gradient (SVRG) approximation. For this approximation, every `update_frequency` number of iterations, a full gradient calculation is made at this "snapshot" point. Intermediate gradient calculations update this snapshot by calculating the gradient of one of the :math:`f_i`s at the current iterate and at the snapshot giving iterations:
    
        .. math ::
            x_{k+1} = x_k - \gamma [n*\nabla f_i(x_k) - n*\nabla f_i(\tilde{x}) + \nabla \sum_{i=0}^{n-1}f_i(\tilde{x})],    where :math:`\tilde{x}` is the latest "snapshot" point . Note that compared with the literature, we multiply by :math:`n`, the number of functions, so that we return an approximate gradient of the whole sum function and not an average gradient. 
    
    Reference: Johnson, R. and Zhang, T., 2013. Accelerating stochastic gradient descent using predictive variance reduction. Advances in neural information processing systems, 26.
    

    Parameters
    ----------
     functions : `list`  of functions
        A list of functions: :code:`[f_{0}, f_{1}, ..., f_{n-1}]`. Each function is assumed to be smooth with an implemented :func:`~Function.gradient` method. All functions must have the same domain. The number of functions must be strictly greater than 1. 
    sampler: An instance of a CIL Sampler class ( :meth:`~optimisation.utilities.sampler`) or of another class which has a `next` function implemented to output integers in {0,...,n-1}.
        This sampler is called each time gradient is called and  sets the internal `function_num` passed to the `approximate_gradient` function.  Default is `Sampler.random_with_replacement(len(functions))`. 
    update_frequency : int or None, optional
        The frequency of updating the full gradient (taking a snapshot). The default is 2*len(functions) so a "snapshot" is taken every 2*len(functions) iterations. 
    store_gradients : bool, default: `False`
        Flag indicating whether to store an update a list of gradients for each function :math:`f_i` or just to store the snapshot point :math:` \tilde{x}` and its gradient :math:`\nabla \sum_{i=0}^{n-1}f_i(\tilde{x})`.

    
    """

    def __init__(self, functions, sampler=None, update_frequency=None, store_gradients=False):
        super(SVRGFunction, self).__init__(functions, sampler)

        # update_frequency for SVRG
        self.update_frequency = update_frequency
    
        if self.update_frequency is None:
            self.update_frequency = 2*self.num_functions
        self.store_gradients = store_gradients

        self._svrg_iter_number = 0
        
        self._full_gradient_at_snapshot = None
        if self.store_gradients:
            self._list_stored_gradients = None
        
        self.snapshot = None 

    def gradient(self, x, out=None):
        """ Selects a random function using the `sampler` and then calls the approximate gradient at :code:`x` or calculates a full gradient depending on the update frequency

        Parameters
        ----------
        x : DataContainer (e.g. ImageData object)
        out: return DataContainer, if `None` a new DataContainer is returned, default `None`.

        Returns
        --------
        DataContainer (e.g. ImageData object)
            the value of the approximate gradient of the sum function at :code:`x` 
        """


    
        # For SVRG, every `update_frequency` a full gradient step is calculated, else an approximate gradient is taken. 
        if (np.isinf(self.update_frequency) == False and (self._svrg_iter_number % (self.update_frequency)) == 0):

            return self._update_full_gradient_and_return(x, out=out)

        else:

            self.function_num = self.sampler.next()

            
            
            if self.function_num >= self.num_functions or self.function_num<0 :
                raise IndexError('The sampler has outputted an index larger than the number of functions to sample from. Please ensure your sampler samples from {0,1,...,len(functions)-1} only.')
            return self.approximate_gradient(x, self.function_num, out=out)

        

    def approximate_gradient(self, x, function_num, out=None):
        """ Calculates the stochastic gradient at the point :math:`x` by using the gradient of the selected function, indexed by `function_number` in {0,...,len(functions)-1}, and the full gradient at the snapshot :math:`\tilde{x}`
            .. math ::
                n*\nabla f_i(x_k) - n*\nabla f_i(\tilde{x}) + \nabla \sum_{i=0}^{n-1}f_i(\tilde{x})
        
        Note that compared with the literature, we multiply by :math:`n`, the number of functions, so that we return an approximate gradient of the whole sum function and not an average gradient.
        
        Parameters
        ----------
        x : DataContainer ( e.g. ImageData)
        out: return DataContainer, if `None` a new DataContainer is returned, default `None`.
        function_num: `int` 
            Between 0 and the number of functions in the list  
        Returns
        --------
        DataContainer (e.g. ImageData)
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
        DataContainer (e.g. ImageData)
            the value of the approximate gradient of the sum function at :code:`x` given a `function_number` in {0,...,len(functions)-1}
        """

        self._svrg_iter_number += 1 

        if self.store_gradients is True:
            if self._list_stored_gradients is None: 
                #Save the gradient of each individual f_i and the gradient of the full sum at the point x. 
                self._list_stored_gradients = [
                    fi.gradient(x) for fi in self.functions]
                self._full_gradient_at_snapshot = sum(self._list_stored_gradients, start=0*x)
            else:
                for i, fi in enumerate(self.functions):
                    self._list_stored_gradients[i].fill( fi.gradient(x))
                self._full_gradient_at_snapshot.fill( sum(self._list_stored_gradients, start=0*x))
           
        else:
            #Save the snapshot point and the gradient of the full sum at the point x. 
            self._full_gradient_at_snapshot = self.full_gradient(x, out=self._full_gradient_at_snapshot) 
            
        if self.snapshot is None: 
            self.snapshot = x.copy()

        self.snapshot.fill(x) 

        #In this iteration all functions in the sum were used to update the gradient 
        self._update_data_passes_indices(list(range(self.num_functions))) 

        #Return the gradient of the full sum at the snapshot. 
        if out is None:
            out = self._full_gradient_at_snapshot
        else:
            out.fill( self._full_gradient_at_snapshot)

        return out


class LSVRGFunction(SVRGFunction):
    """""
    A class representing a function for Loopless Stochastic Variance Reduced Gradient (SVRG) approximation. This is similar to SVRG, except the full gradient at a "snapshot"  is calculated at random  intervals rather than at fixed numbers of iterations. 
    
    
    Reference: D. Kovalev et al., “Don’t jump through hoops and remove those loops: SVRG and Katyusha are better without the outer loop,” in Algo Learn Theo, PMLR, 2020.

    Parameters
    ----------
     functions : `list`  of functions
        A list of functions: :code:`[f_{0}, f_{1}, ..., f_{n-1}]`. Each function is assumed to be smooth with an implemented :func:`~Function.gradient` method. All functions must have the same domain. The number of functions must be strictly greater than 1. 
    sampler: An instance of a CIL Sampler class ( :meth:`~optimisation.utilities.sampler`) or of another class which has a `next` function implemented to output integers in {0,...,n-1}.
        This sampler is called each time gradient is called and  sets the internal `function_num` passed to the `approximate_gradient` function.  Default is `Sampler.random_with_replacement(len(functions))`. 
    update_prob: positive float, default: 1/n
        The probability of updating the full gradient (taking a snapshot) at each iteration. The default is :math:`1./n` so, in expectation, a snapshot will be taken every :math:`n` iterations. 
    store_gradients : bool, default: `False`
        Flag indicating whether to store an update a list of gradients for each function :math:`f_i` or just to store the snapshot point :math:` \tilde{x}` and it's gradient :math:`\nabla \sum_{i=0}^{n-1}f_i(\tilde{x})`.

        
   

    """

    def __init__(self, functions, sampler=None, update_prob=None, store_gradients=False, seed=None):

        super(LSVRGFunction, self).__init__(
            functions, sampler=sampler, store_gradients=store_gradients)

        # update frequency based on probability
        self.update_prob = update_prob
        # default update_prob for Loopless SVRG
        if self.update_prob is None:
            self.update_prob = 1./self.num_functions

        # the random generator used to decide if the gradient calculation is a full gradient or an approximate gradient 
        self.generator = np.random.default_rng(seed=seed)

    def gradient(self, x, out=None):
        """ Selects a random function using the `sampler` and then calls the approximate gradient at :code:`x` or calculates a full gradient depending on the update probability 

        Parameters
        ----------
        x : DataContainer ( e.g. ImageData)
        out: return DataContainer, if `None` a new DataContainer is returned, default `None`.

        Returns
        --------
        DataContainer ( e.g. ImageData)
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

        
