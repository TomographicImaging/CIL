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

from cil.optimisation.functions import Function
import numpy

class IndicatorBox(Function):
    
    
    r'''Indicator function for box constraint
            
      .. math:: 
         
         f(x) = \mathbb{I}_{[a, b]} = \begin{cases}  
                                            0, \text{ if } x \in [a, b] \\
                                            \infty, \text{otherwise}
                                     \end{cases}
    
    '''
    
    def __init__(self,lower=-numpy.inf,upper=numpy.inf):
        '''creator

        :param lower: lower bound
        :type lower: float, default = :code:`-numpy.inf`
        :param upper: upper bound
        :type upper: float, optional, default = :code:`numpy.inf`
        '''
        super(IndicatorBox, self).__init__()
        self.lower = lower
        self.upper = upper

    def __call__(self,x):
        
        '''Evaluates IndicatorBox at x'''
                
        if (numpy.all(x.as_array() >= self.lower) and 
            numpy.all(x.as_array() <= self.upper) ):
            val = 0
        else:
            val = numpy.inf
        return val
    
    def gradient(self,x):
        return ValueError('Not Differentiable') 
    
    def convex_conjugate(self,x):
        
        '''Convex conjugate of IndicatorBox at x'''

        return x.maximum(0).sum()
         
    def proximal(self, x, tau, out=None):
        
        r'''Proximal operator of IndicatorBox at x

            .. math:: prox_{\tau * f}(x)
        '''
        
        if out is None:
            return (x.maximum(self.lower)).minimum(self.upper)        
        else:               
            x.maximum(self.lower, out=out)
            out.minimum(self.upper, out=out) 
            
    def proximal_conjugate(self, x, tau, out=None):
        
        r'''Proximal operator of the convex conjugate of IndicatorBox at x:

          ..math:: prox_{\tau * f^{*}}(x)
        '''

        if out is None:
            
            return x - tau * self.proximal(x/tau, tau)
        
        else:
            
            self.proximal(x/tau, tau, out=out)
            out *= -1*tau
            out += x
