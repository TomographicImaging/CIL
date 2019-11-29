# -*- coding: utf-8 -*-
#  CCP in Tomographic Imaging (CCPi) Core Imaging Library (CIL).

#   Copyright 2017 UKRI-STFC
#   Copyright 2017 University of Manchester

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from ccpi.optimisation.functions import Function
from ccpi.framework import BlockDataContainer
import numpy as np
import functools


# promxial conjugate, proximal no closed-form solutions
# convex conjugate closed form solution, not implemented


class smoothMixedL21Norm(Function):
    
        
    def __init__(self, epsilon):

        super(smoothMixedL21Norm, self).__init__(L=1)          
        self.epsilon = epsilon        
        
        if self.epsilon==0:
            raise ValueError('We need epsilon>0')
                            
    def __call__(self, x):
        
        r"""Returns the value of the MixedL21Norm function at x.                                            
        """
        if not isinstance(x, BlockDataContainer):
            raise ValueError('__call__ expected BlockDataContainer, got {}'.format(type(x))) 
         
        tmp_cont = x.containers                        
        tmp = x.get_item(0) * 0.
        for el in tmp_cont:
            tmp += el.power(2.)
        tmp.add(self.epsilon**2, out = tmp)    
        return tmp.sqrt().sum()


    def gradient(self, x, out=None): 
        
        tmp = x.get_item(0) * 0.
        for el in x.containers:
            tmp += el.power(2.)
        tmp+=self.epsilon**2
                  
        if out is None:
            return x.divide(tmp.sqrt())
        else:
            out.fill(x.divide(tmp.sqrt()))
                        
                            
#    def convex_conjugate(self,x):
#        
#        r"""Returns the value of the convex conjugate of the MixedL21Norm function at x.
#        
#        This is the Indicator function of :math:`\mathbb{I}_{\{\|\cdot\|_{2,\infty}\leq1\}}(x^{*})`,
#        
#        i.e., 
#        
#        .. math:: \mathbb{I}_{\{\|\cdot\|_{2, \infty}\leq1\}}(x^{*}) 
#            = \begin{cases} 
#            0, \mbox{if } \|x\|_{2, \infty}\leq1\\
#            \infty, \mbox{otherwise}
#            \end{cases}
#        
#        where, 
#        
#        .. math:: \|x\|_{2,\infty} = \max\{ \|x\|_{2} \} = \max\{ \sqrt{ (x^{1})^{2} + (x^{2})^{2} + \dots}\}
#        
#        """
#        if not isinstance(x, BlockDataContainer):
#            raise ValueError('__call__ expected BlockDataContainer, got {}'.format(type(x))) 
#                
#        tmp1 = x.get_item(0) * 0.
#        for el in x.containers:
#            tmp1 += el.power(2.)
#        tmp1.add(self.epsilon**2, out = tmp1)
#        tmp = tmp1.sqrt().as_array().max() - 1
#                    
#        if tmp<=1e-6:
#            return 0
#        else:
#            return np.inf
#        
#    def proximal(self, x, tau, out=None):   
#        
#        
#        return x - tau * self.proximal_conjugate(x/tau, 1/tau, out=None)
#        
#        # This has no closed form solution and need to be computed numerically
#                
##        raise NotImplementedError
#            
#    def proximal_conjugate(self, x, tau, out=None): 
#        
#        if out is None:                                        
#            tmp = x.get_item(0) * 0	
#            for el in x.containers:	
#                tmp += el.power(2.)	
#            tmp+=self.epsilon**2                
#            tmp.sqrt(out=tmp)	
#            tmp.maximum(1.0, out=tmp)	
#            frac = [ el.divide(tmp) for el in x.containers ]	
#            return BlockDataContainer(*frac)        
#        else:                            
#            res1 = functools.reduce(lambda a,b: a + b*b, x.containers, x.get_item(0) * 0 )
#            res1+=self.epsilon**2
#            res1.sqrt(out=res1)	
#            res1.maximum(1.0, out=res1)	
#            x.divide(res1, out=out)
#                             