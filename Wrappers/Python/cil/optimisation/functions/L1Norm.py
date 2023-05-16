# -*- coding: utf-8 -*-
#  Copyright 2019 United Kingdom Research and Innovation
#  Copyright 2019 The University of Manchester
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
# CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt

from cil.optimisation.functions import Function    
from cil.framework import BlockDataContainer   
import numpy as np

 
def soft_shrinkage(x, tau, out=None):
    
    r"""Returns the value of the soft-shrinkage operator at x.
    """

    should_return = False
    if out is None:
        out = x.abs()
        should_return = True
    else:
        x.abs(out = out)
    out -= tau
    out.maximum(0, out = out)
    out *= x.sign()   

    if should_return:
        return out        
    
class L1Norm(Function):
    
    r"""L1Norm function
            
        Consider the following cases:           
            a) .. math:: F(x) = ||x||_{1}
            b) .. math:: F(x) = ||x - b||_{1}
                                
    """   
           
    def __init__(self, **kwargs):
        '''creator

        Cases considered (with/without data):            
        a) :math:`f(x) = ||x||_{1}`
        b) :math:`f(x) = ||x - b||_{1}`

        :param b: translation of the function
        :type b: :code:`DataContainer`, optional
        '''
        super(L1Norm, self).__init__()
        self.b = kwargs.get('b',None)
        
    def __call__(self, x):
        
        r"""Returns the value of the L1Norm function at x.
        
        Consider the following cases:           
            a) .. math:: F(x) = ||x||_{1}
            b) .. math:: F(x) = ||x - b||_{1}        
        
        """
        
        y = x
        if self.b is not None: 
            y = x - self.b
        return y.abs().sum()  
          
    def convex_conjugate(self,x):
        
        r"""Returns the value of the convex conjugate of the L1Norm function at x.
        Here, we need to use the convex conjugate of L1Norm, which is the Indicator of the unit 
        :math:`L^{\infty}` norm
        
        Consider the following cases:
                
                a) .. math:: F^{*}(x^{*}) = \mathbb{I}_{\{\|\cdot\|_{\infty}\leq1\}}(x^{*}) 
                b) .. math:: F^{*}(x^{*}) = \mathbb{I}_{\{\|\cdot\|_{\infty}\leq1\}}(x^{*}) + <x^{*},b>      
        
    
        .. math:: \mathbb{I}_{\{\|\cdot\|_{\infty}\leq1\}}(x^{*}) 
            = \begin{cases} 
            0, \mbox{if } \|x^{*}\|_{\infty}\leq1\\
            \infty, \mbox{otherwise}
            \end{cases}
    
        """        
        
        tmp = x.abs().max() - 1
        if tmp<=1e-5:            
            if self.b is not None:
                return self.b.dot(x)
            else:
                return 0.
        return np.inf    

                    
    def proximal(self, x, tau, out=None):
        
        r"""Returns the value of the proximal operator of the L1Norm function at x.
        
        
        Consider the following cases:
                
                a) .. math:: \mathrm{prox}_{\tau F}(x) = \mathrm{ShinkOperator}(x)
                b) .. math:: \mathrm{prox}_{\tau F}(x) = \mathrm{ShinkOperator}(x) + b   
    
        where,
        
        .. math :: \mathrm{prox}_{\tau F}(x) = \mathrm{ShinkOperator}(x) = sgn(x) * \max\{ |x| - \tau, 0 \}
                            
        """  

                    
        if out is None:                                                
            if self.b is not None:                                
                return self.b + soft_shrinkage(x - self.b, tau)
            else:
                return soft_shrinkage(x, tau)             
        else: 
            
            if self.b is not None:
                soft_shrinkage(x - self.b, tau, out = out)
                out += self.b
            else:
                soft_shrinkage(x, tau, out = out)   


class MixedL11Norm(Function):

    r"""MixedL11Norm function
                     
    .. math:: F(x) = ||x||_{1,1} = \sum |x_{1}| + |x_{2}| + \cdots + |x_{n}|

    Note
    ----
    MixedL11Norm is a separable function, therefore it can also be defined using the :class:`BlockFunction`.


    See Also
    --------
    L1Norm, MixedL21Norm


    """   

    def __init__(self, **kwargs):
        super(MixedL11Norm, self).__init__(**kwargs)

    def __call__(self, x):

        r"""Returns the value of the MixedL11Norm function at x. 

        :param x: :code:`BlockDataContainer`                                           
        """
        if not isinstance(x, BlockDataContainer):
            raise ValueError('__call__ expected BlockDataContainer, got {}'.format(type(x))) 
                    
        return x.abs().sum()
        
    def proximal(self, x, tau, out = None):

        r"""Returns the value of the proximal operator of the MixedL11Norm function at x.
                
        .. math:: \mathrm{prox}_{\tau F}(x) = \mathrm{ShinkOperator}(x)
    
        where,
        
        .. math :: \mathrm{prox}_{\tau F}(x) = \mathrm{ShinkOperator}(x) := sgn(x) * \max\{ |x| - \tau, 0 \}
                            
        """      

        if not isinstance(x, BlockDataContainer):
            raise ValueError('__call__ expected BlockDataContainer, got {}'.format(type(x)))         

        return soft_shrinkage(x, tau, out = out) 



