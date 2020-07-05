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


from ccpi.optimisation.functions import Function
from ccpi.framework import BlockDataContainer
import numpy as np
import functools


class MixedL21Norm(Function):
    
    
    """ MixedL21Norm function: :math:`F(x) = ||x||_{2,1} = \sum |x|_{2} = \sum \sqrt{ (x^{1})^{2} + (x^{2})^{2} + \dots}`                  
    
        where x is a BlockDataContainer, i.e., :math:`x=(x^{1}, x^{2}, \dots)`
    
    """      
    
    def __init__(self, **kwargs):

        super(MixedL21Norm, self).__init__()  
                    
        
    def __call__(self, x):
        
        r"""Returns the value of the MixedL21Norm function at x. 

        :param x: :code:`BlockDataContainer`                                           
        """
        if not isinstance(x, BlockDataContainer):
            raise ValueError('__call__ expected BlockDataContainer, got {}'.format(type(x))) 
              
        return x.pnorm(p=2).sum()                                
            
                            
    def convex_conjugate(self,x):
        
        r"""Returns the value of the convex conjugate of the MixedL21Norm function at x.
        
        This is the Indicator function of :math:`\mathbb{I}_{\{\|\cdot\|_{2,\infty}\leq1\}}(x^{*})`,
        
        i.e., 
        
        .. math:: \mathbb{I}_{\{\|\cdot\|_{2, \infty}\leq1\}}(x^{*}) 
            = \begin{cases} 
            0, \mbox{if } \|x\|_{2, \infty}\leq1\\
            \infty, \mbox{otherwise}
            \end{cases}
        
        where, 
        
        .. math:: \|x\|_{2,\infty} = \max\{ \|x\|_{2} \} = \max\{ \sqrt{ (x^{1})^{2} + (x^{2})^{2} + \dots}\}
        
        """
        if not isinstance(x, BlockDataContainer):
            raise ValueError('__call__ expected BlockDataContainer, got {}'.format(type(x))) 
                                        
        tmp = (x.pnorm(2).max() - 1)
        if tmp<=1e-5:
            return 0
        else:
            return np.inf
                    
    def proximal(self, x, tau, out=None):
        
        r"""Returns the value of the proximal operator of the MixedL21Norm function at x.
        
        .. math :: \mathrm{prox}_{\tau F}(x) = \frac{x}{\|x\|_{2}}\max\{ \|x\|_{2} - \tau, 0 \}
        
        where the convention 0 · (0/0) = 0 is used.
        
        """

        if out is None:
            tmp = x.pnorm(2)
            res = (tmp - tau).maximum(0.0) * x/tmp

            # TODO avoid using numpy, add operation in the framework
            # This will be useful when we add cupy 
                                 
            for el in res.containers:

                elarray = el.as_array()
                elarray[np.isnan(elarray)]=0
                el.fill(elarray)

            return res
            
        else:
            
            tmp = x.pnorm(2)
            tmp_ig = 0.0 * tmp
            (tmp - tau).maximum(0.0, out = tmp_ig)
            tmp_ig.multiply(x, out = out)
            out.divide(tmp, out = out)
            
            for el in out.containers:
                
                elarray = el.as_array()
                elarray[np.isnan(elarray)]=0
                el.fill(elarray)  

            out.fill(out)

    def proximal_conjugate(self, x, tau, out=None): 

        
        if out is None:                                        
            tmp = x.get_item(0) * 0	
            for el in x.containers:	
                tmp += el.power(2.)	
            tmp.sqrt(out=tmp)	
            tmp.maximum(1.0, out=tmp)	
            frac = [ el.divide(tmp) for el in x.containers ]	
            return BlockDataContainer(*frac)
            
        else:
                            
            res1 = functools.reduce(lambda a,b: a + b*b, x.containers, x.get_item(0) * 0 )
            res1.sqrt(out=res1)	
            res1.maximum(1.0, out=res1)	
            x.divide(res1, out=out)            

class SmoothMixedL21Norm(Function):
    
    """ SmoothMixedL21Norm function: :math:`F(x) = ||x||_{2,1} = \sum |x|_{2} = \sum \sqrt{ (x^{1})^{2} + (x^{2})^{2} + \epsilon^2 + \dots}`                  
    
        where x is a BlockDataContainer, i.e., :math:`x=(x^{1}, x^{2}, \dots)`
        
        Conjugate, proximal and proximal conjugate methods no closed-form solution
        
    
    """    
        
    def __init__(self, epsilon):
                
        r'''
        :param epsilon: smoothing parameter making MixedL21Norm differentiable 
        '''

        super(SmoothMixedL21Norm, self).__init__(L=1)
        self.epsilon = epsilon   
                
        if self.epsilon==0:
            raise ValueError('We need epsilon>0. Otherwise, call "MixedL21Norm" ')
                            
    def __call__(self, x):
        
        r"""Returns the value of the SmoothMixedL21Norm function at x.
        """
        if not isinstance(x, BlockDataContainer):
            raise ValueError('__call__ expected BlockDataContainer, got {}'.format(type(x))) 
            
            
        return (x.pnorm(2).power(2) + self.epsilon**2).sqrt().sum()
         

    def gradient(self, x, out=None): 
        
        r"""Returns the value of the gradient of the SmoothMixedL21Norm function at x.
        
        \frac{x}{|x|}
                
                
        """     
        
        if not isinstance(x, BlockDataContainer):
            raise ValueError('__call__ expected BlockDataContainer, got {}'.format(type(x))) 
                   
        denom = (x.pnorm(2).power(2) + self.epsilon**2).sqrt()
                          
        if out is None:
            return x.divide(denom)
        else:
            x.divide(denom, out=out)        
