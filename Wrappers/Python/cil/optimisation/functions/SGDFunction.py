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

from cil.optimisation.functions import SubsetSumFunction

class SGDFunction(SubsetSumFunction):

    r""" Stochastic Gradient Descent Function (SGDFunction) 

    The SDGFunction represents the objective function :math:`\sum_{i=1}^{n}F_{i}(x)`, where
    :math:`n` denotes the number of subsets. 
    
    Parameters:
    -----------
    functions : list(functions) 
                A list of functions: :code:`[F_{1}, F_{2}, ..., F_{n}]`. Each function is assumed to be smooth function with an implemented :func:`~Function.gradient` method.
    sampling : :obj:`string`, Default = :code:`random`
               Selection process for each function in the list. It can be :code:`random` or :code:`sequential`. 
    replacement : :obj:`boolean`. Default = :code:`True`
               The same subset can be selected when :code:`replacement=True`. 
    
    Note
    ----
        
    The :meth:`~SGDFunction.gradient` computes the `gradient` of one function from the list :math:`[F_{1},\cdots,F_{n}]`,
    
    .. math:: \partial F_{i}(x) .

    The ith function is selected from the :meth:`~SubsetSumFunction.next_subset` method.
    
"""
  
    def __init__(self, functions, sampling = "random", replacement = False, suffle="random"):

        super(SGDFunction, self).__init__(functions, sampling = sampling, replacement = replacement, suffle=suffle)

    def gradient(self, x, out):
        
        """ Returns the gradient of the selected function at :code:`x`. The function is selected using the :meth:`~SubsetSumFunction.next_subset`
        """

        # Select the next subset
        self.next_subset()
        
        # Compute new gradient for current subset
        self.functions[self.subset_num].gradient(x, out=out)
        out*=self.num_subsets         
        