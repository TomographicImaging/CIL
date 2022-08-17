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
    '''Class for use as objective function in gradient type algorithms to enable the use of subsets.
    The `gradient` method does not correspond to the mathematical gradient of a sum of functions, 
    but rather to a variance-reduced approximated gradient corresponding to the minibatch SGD algorithm.
    More details can be found below, in the gradient method.
    Parameters:
    -----------
    - (optional) 
        precond: function taking into input an integer (subset_num) and a DataContainer and outputting a DataContainer
        serves as diagonal preconditioner
        default None
    - (optional)
        gradient_initialisation_point: point to initialize the gradient of each subset
        and the full gradient
        default None
           
    '''
  
    def __init__(self, functions, sampling = "random", precond=None):

        super(SGDFunction, self).__init__(functions, sampling = sampling)
        self.precond = precond

    def gradient(self, x, out=None):
        """
        Returns a vanilla stochastic gradient estimate, defined below.
        For f = 1/num_subsets \sum_{i=1}^num_subsets f_i, the output is computed as follows:
            - choose a subset j with function next_subset()
            - compute the gradient of the j-th function at current iterate
            - this gives an unbiased estimator of the gradient
        """

        # Select the next subset (uniformly at random by default). This generates self.subset_num
        self.next_subset()
        
        # Compute new gradient for current subset, store in ret
        if out is None:
            ret = 0.0 * x
            self.functions[self.subset_num].gradient(x, out=ret)
        else:
            self.functions[self.subset_num].gradient(x, out=out)

        # Apply preconditioning
        if self.precond is not None:
            if out is None:
                ret.multiply(self.precond(self.subset_num,x),out=ret)
            else:
                out.multiply(self.precond(self.subset_num,x),out=out)

        if out is None:
            return ret
        