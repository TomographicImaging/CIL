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

from cil.optimisation.functions import ApproximateGradientSumFunction
import numbers

class SGFunction(ApproximateGradientSumFunction):

    r""" Stochastic Gradient Function (SGFunction) 
    
"""
  
    def __init__(self, functions, selection=None):

        super(SGFunction, self).__init__(functions, selection)

    def approximate_gradient(self,subset_num, x, out):
        
        """ Returns the gradient of the selected function or batch of functions at :code:`x`. 
            The function or batch of functions is selected using the :meth:`~ApproximateGradientSumFunction.next_function`.
        """        

        # single function 
        if isinstance(self.function_num, numbers.Number):
            self.functions[self.function_num].gradient(x, out=out)

        # mini-batch, i.e., collection of functions
        else:
            sum_grad = 0
            for i in self.function_num:
                sum_grad += self.functions[i].gradient(x)
            out.fill(sum_grad)
            
        out*=self.selection.num_batches

