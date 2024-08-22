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


class SAGFunction(ApproximateGradientSumFunction):

    r"""
    The stochastic average gradient (SAG) function takes a index :math:`i_k` and calculates the approximate gradient of :math:`\sum_{i=1}^{n-1}f_i` at iteration :math:`x_k` as
    
    .. math ::
                \sum_{i=1}^{n-1} g_i^k \qquad \text{where} \qquad g_i^k= \begin{cases}
                                                                            \nabla f_i(x_k), \text{ if } i=i_k\\
                                                                            g_i^{k-1},\text{ otherwise }
                                                                            \end{cases}

        
            
    
    The idea is that by incorporating a memory of previous gradient values the SAG method can achieve a faster convergence rate than black-box stochastic gradient methods. 
    
    Note
    -----
    Compared with the literature, we do not divide by :math:`n`, the number of functions, so that we return an approximate gradient of the whole sum function and not an average gradient.

    Reference
    ----------
    Schmidt, M., Le Roux, N. and Bach, F., 2017. Minimizing finite sums with the stochastic average gradient. Mathematical Programming, 162, pp.83-112. https://doi.org/10.1007/s10107-016-1030-6. 

    Parameters:
    -----------
    functions : `list`  of functions
        A list of functions: :math:`[f_{0}, f_{1}, ..., f_{n-1}]`. Each function is assumed to be smooth with an implemented :func:`~Function.gradient` method. All functions must have the same domain. The number of functions (equivalently the length of the list `n`) must be strictly greater than 1. 
    sampler: An instance of a CIL Sampler class ( :meth:`~optimisation.utilities.sampler`) or of another class which has a `next` function implemented to output integers in :math:`{0,...,n-1}`.
        This sampler is called each time `gradient` is called and sets the internal `function_num` passed to the `approximate_gradient` function.  Default is `Sampler.random_with_replacement(len(functions))`. 
    
    Note
    ------
    
    The user has the option of calling the class method `warm_start_approximate_gradients` after initialising this class. This will compute and store the gradient for each function at an initial point, equivalently setting :math:`g_i^0=\nabla f_i(x_0)` for initial point :math:`x_0`.  If this method is not called, the gradients are initialised with zeros. 

    Note:  
    ------  

    This function's memory requirements are `n + 3` times the image space, that is with 100 subsets the memory requirement is 103 images, which is huge.
    

    """

    def __init__(self, functions,  sampler=None):
        self._list_stored_gradients = None
        self._full_gradient_at_iterate = None
        self._warm_start_just_done = False
        self._sampled_grad = None
        
        super(SAGFunction, self).__init__(functions, sampler)


        

    def approximate_gradient(self, x, function_num,  out=None):
        """ SAG approximate gradient, calculated at the point :math:`x` and updated using the function index given by `function_num`.  

        Parameters
        ----------
        x : DataContainer (e.g. ImageData object)
            Element in the domain of the `functions`
        function_num: `int` 
            Between 0 and the number of functions in the list  

        """
        
        
        if self._list_stored_gradients is None: # Initialise the stored gradients on the first call of gradient unless using warm start.  
            self._list_stored_gradients = [
                0*x for fi in self.functions]
            self._full_gradient_at_iterate = 0*x
            self._sampled_grad = x.copy()
            self._stochastic_grad_difference = x.copy()
        
        if self.function_num >= self.num_functions or self.function_num<0 : # check the sampler and raise an error if needed
            raise IndexError(f"The sampler has produced the index {self.function_num} which does not match the expected range of available functions to sample from. Please ensure your sampler only selects from [0,1,...,len(functions)-1] ")

            
        # Calculate the gradient of the sampled function at the current iterate 
        self.functions[function_num].gradient(x, out=self._sampled_grad)

        
        # Calculate the difference between the new gradient of the sampled function and the stored one
        self._sampled_grad.sapyb(
            1., self._list_stored_gradients[function_num], -1., out=self._stochastic_grad_difference)

        # Calculate the  approximate gradient
        out = self._update_approx_gradient(out)

        # Update the stored gradients 
        self._list_stored_gradients[function_num].fill(
            self._sampled_grad)
        
        # Calculate the stored full gradient
        self._full_gradient_at_iterate.sapyb(
            1., self._stochastic_grad_difference, 1., out=self._full_gradient_at_iterate)

        return out
    
    def _update_approx_gradient(self, out):
        """Internal function used to differentiate between the SAG and SAGA calculations. This is the SAG approximation: """
        out = self._stochastic_grad_difference.sapyb(  
                1., self._full_gradient_at_iterate, 1., out=out)  

        return out 
    
    def warm_start_approximate_gradients(self, initial):
        """A function to warm start SAG or SAGA algorithms by initialising all the gradients at an initial point. Equivalently setting :math:`g_i^0=\nabla f_i(x_0)` for initial point :math:`x_0`. 
        
        Parameters
        ----------
        initial: DataContainer,
            The initial point to warmstart the calculation
            
        Note
        ----
        When using SAG or SAGA with a deterministic algorithm, you should warm start the SAG-SAGA Function with the same initial point that you initialise the algorithm
        
        """
        self._list_stored_gradients = [
            fi.gradient(initial) for fi in self.functions]
        self._full_gradient_at_iterate = np.sum(self._list_stored_gradients)
        self._update_data_passes_indices(list(range(self.num_functions)))
        self._sampled_grad = initial.copy()
        self._stochastic_grad_difference = initial.copy()

    @property
    def data_passes_indices(self): 
        """ The property :code:`data_passes_indices` is a list of lists holding the indices of the functions that are processed in each call of `gradient`. This list is updated each time `gradient` is called by appending a list of the indices of the functions used to calculate the gradient.  
        This is overwritten from the base class to first check to see if the approximate gradient was warm started and, if it was, ensure that the first element of `data_passes_indices` contains each index used to warm start and the index used in the first call to `gradient`. Thus the length of `data_passes_indices` is always equal to the number of calls to `gradient`. 
        """
        ret = self._data_passes_indices[:]  
        if len(ret[0]) == self.num_functions:  
            a = ret.pop(1)  
            ret[0] += a  
        return ret
    
class SAGAFunction(SAGFunction):

    r"""
    SAGA (SAG-Ameliore) is an accelerated version of the stochastic average gradient (SAG) function which takes a index :math:`i_k` and calculates the approximate gradient of :math:`\sum_{i=1}^{n-1}f_i` at iteration :math:`x_k` as
    
    .. math ::
                 n\left(g_{i_k}^{k}-g_{i_k}^{k-1}\right)+\sum_{i=1}^{n-1} g_i^{k-1} \qquad \text{where} \qquad g_i^k= \begin{cases}
                                                                            \nabla f_i(x_k), \text{ if } i=i_k\\
                                                                            g_i^{k-1},\text{ otherwise}
                                                                            \end{cases}
                                                                        
         
    SAGA improves on the theory behind SAG and SVRG, with better theoretical convergence rates. Compared to SAG it is an unbiased estimator. 
    
    Note
    ------
    Compared with the literature, we do not divide by :math:`n`, the number of functions, so that we return an approximate gradient of the whole sum function and not an average gradient.

    Note:  
    ------  

    This function's memory requirements are `n + 3` times the image space, that is with 100 subsets the memory requirement is 103 images, which is huge.
    
    Reference
    ----------
    Defazio, A., Bach, F. and Lacoste-Julien, S., 2014. SAGA: A fast incremental gradient method with support for non-strongly convex composite objectives. Advances in neural information processing systems, 27. https://proceedings.neurips.cc/paper_files/paper/2014/file/ede7e2b6d13a41ddf9f4bdef84fdc737-Paper.pdf
   

    Parameters:
    -----------
    functions : `list`  of functions
                A list of functions: :code:`[f_{0}, f_{1}, ..., f_{n-1}]`. Each function is assumed to be smooth function with an implemented :func:`~Function.gradient` method. Each function must have the same domain. The number of functions must be strictly greater than 1. 
    sampler: An instance of one of the :meth:`~optimisation.utilities.sampler` classes which has a `next` function implemented and a `num_indices` property.
        This sampler is called each time gradient is called and  sets the internal `function_num` passed to the `approximate_gradient` function.  The `num_indices` must match the number of functions provided. Default is `Sampler.random_with_replacement(len(functions))`. 
    
    Note
    ----
    The user has the option of calling the class method `warm_start_approximate_gradients` after initialising this class. This will compute and store the gradient for each function at an initial point, equivalently setting :math:`g_i^0=\nabla f_i(x_0)` for initial point :math:`x_0`. If this method is not called, the gradients are initialised with zeros. 

  
    """

    def __init__(self, functions,  sampler=None):
        super(SAGAFunction, self).__init__(functions, sampler)


    def _update_approx_gradient(self, out):
        """Internal function used to differentiate between the SAG and SAGA calculations. This is the SAGA approximation and differs in the constants multiplying the gradients: """

        # Due to the convention that we follow: without the 1/n factor
        out= self._stochastic_grad_difference.sapyb(  
            self.num_functions, self._full_gradient_at_iterate, 1., out)  

        return out 