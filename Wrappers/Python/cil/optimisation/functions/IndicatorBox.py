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
from cil.framework import AcquisitionData, ImageData, DataContainer
import numpy as np


class IndicatorBox(Function):
    
    
    r'''Indicator function for box constraint
            
      .. math:: 
         
         f(x) = \mathbb{I}_{[a, b]} = \begin{cases}  
                                            0, \text{ if } x \in [a, b] \\
                                            \infty, \text{otherwise}
                                     \end{cases}
    
    '''
    
    def __init__(self,lower=-np.inf,upper=np.inf):
        '''creator

        Parameters:
        -----------
        lower : float, DataContainer or numpy array, default ``-np.inf``
          Lower bound
        upper : float, DataContainer or numpy array, default ``np.inf``
          upper bound
        
        If passed a DataContainer or numpy array, the bounds can be set to different values for each element.
        '''
        super(IndicatorBox, self).__init__()
        
        # We set lower and upper to either a float or a numpy array        
        self.lower = lower
        self.upper = upper
        if isinstance(lower, (np.ndarray, DataContainer, AcquisitionData, ImageData)):
            if not isinstance(lower, np.ndarray):
                self.lower = lower.as_array()
        if isinstance(upper, (np.ndarray, DataContainer, AcquisitionData, ImageData)):
            if not isinstance(upper, np.ndarray):
                self.upper = upper.as_array()

    def __call__(self,x):
        
        '''Evaluates IndicatorBox at x'''
                
        if (np.all(x.as_array() >= self.lower) and 
            np.all(x.as_array() <= self.upper) ):
            val = 0
        else:
            val = np.inf
        return val
    
    def gradient(self,x):
        '''IndicatorBox is not differentiable, so calling gradient will raise a ValueError'''
        return ValueError('Not Differentiable') 
    
    def convex_conjugate(self,x):
        '''Convex conjugate of IndicatorBox at x'''
        return x.maximum(0).sum()
         
    def proximal(self, x, tau, out=None):
        
        r'''Proximal operator of IndicatorBox at x

            .. math:: prox_{\tau * f}(x)
        '''
        
        if out is None:
            out = x.maximum(self.lower)
            out.minimum(self.upper, out=out)
            return out
        else:               
            x.maximum(self.lower, out=out)
            out.minimum(self.upper, out=out) 
            
    def proximal_conjugate(self, x, tau, out=None):
        
        r'''Proximal operator of the convex conjugate of IndicatorBox at x:

          ..math:: prox_{\tau * f^{*}}(x)
        '''

        if out is None:
            # x - tau * self.proximal(x/tau, tau):
            # use x as temporary storage variable
            x/=tau
            out = self.proximal(x, tau)
            out *= -1*tau
            # restore the values of x
            x*=tau
            tmp += x
            return tmp
        
        else:
            self.proximal(x/tau, tau, out=out)
            out *= -1*tau
            out += x
