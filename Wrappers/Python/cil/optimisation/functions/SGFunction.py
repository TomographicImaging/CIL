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
# - Zeljko Kerata (Department of Computer Science, University College London, UK)
# - Evgueni Ovtchinnikov (STFC - UKRI)
# - Georg Schramm (Department of Imaging and Pathology, Division of Nuclear Medicine, KU Leuven, Leuven, Belgium)

from .ApproximateGradientSumFunction import ApproximateGradientSumFunction
from .Function import SumFunction

class SGFunction(ApproximateGradientSumFunction):

    r"""
    Stochastic gradient function, a child class of :code:`ApproximateGradientSumFunction`, which defines from a list of functions, :math:`{f_0,...,f_{n-1}}` a `SumFunction`, :math:`f_0+...+f_{n-1}` where each time the `gradient` is called, the :code:`sampler` provides an index, :math:`i \in {0,...,n-1}` 
    and the :code:`gradient` method returns the approximate gradient :math:`n \nabla_x f_i(x)`. This can be used with the :code:`cil.optimisation.algorithms` algorithm :code:`GD` to give a stochastic gradient descent algorithm. 
   
    Parameters
    -----------
    functions : `list`  of functions
        A list of functions: :code:`[f_{0}, f_{1}, ..., f_{n-1}]`. Each function is assumed to be smooth with an implemented :func:`~Function.gradient` method. Although CIL does not define a domain of a :code:`Function`, all functions are supposed to have the same domain. The number of functions must be strictly greater than 1. 
    sampler: An instance of a CIL Sampler class ( :meth:`~optimisation.utilities.sampler`) or of another class which has a :code:`__next__` function implemented to output integers in {0,...,n-1}.
        This sampler is called each time gradient is called and  sets the internal :code:`function_num` passed to the :code:`approximate_gradient` function.  Default is :code:`Sampler.random_with_replacement(len(functions))`. 
    """
  
    def __init__(self, functions, sampler=None):
        super(SGFunction, self).__init__(functions, sampler)    
        

    def approximate_gradient(self, x, function_num,  out=None):
        
        r""" Returns the gradient of the function at index `function_num` at :code:`x`. 
        
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
        if self.function_num >= self.num_functions or self.function_num<0 :
            raise IndexError(
                'The sampler has outputted an index larger than the number of functions to sample from. Please ensure your sampler samples from {0,1,...,len(functions)-1} only.')

        

        # compute gradient of the function indexed by function_num
        if out is None:
            out = self.functions[function_num].gradient(x)
        else:
            self.functions[function_num].gradient(x, out = out) 

        # scale wrt number of functions 
        out*=self.num_functions 
        
        return out         





           