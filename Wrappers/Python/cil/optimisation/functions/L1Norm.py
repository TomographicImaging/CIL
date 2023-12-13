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
            
    In the weighted case, :math:`w` is an array of positive weights.
    
    a) .. math:: F(x) = ||x||_{L^1(w)}
    b) .. math:: F(x) = ||x - b||_{L^1(w)}
        
    with :math:`||x||_{L^1(w)} = || x \cdot w||_1 = \sum_{i=1}^{n} |x_i| w_i`.

    Parameters
    -----------

        weight: DataContainer, numpy ndarray, default None
            Array of weights. If :code:`None` returns the L1 Norm.
        b: DataContainer, default None
            Translation of the function.
                                
    """
    def __init__(self, b=None, weight=None):
        super(L1Norm, self).__init__(L=None)
        if weight is None:
            self.function = _L1Norm(b=b)
        else:
            self.function = _WeightedL1Norm(b=b, weight=weight)

    def __call__(self, x):
        r"""Returns the value of the L1Norm function at x.
        
        .. math:: f(x) = ||x - b||_{L^1(w)}
        """
        return self.function(x)

    def convex_conjugate(self, x):
        r"""Returns the value of the convex conjugate of the L1 Norm function at x.


    This is the indicator of the unit :math:`L^{\infty}` norm 
    
    Consider the following cases:
            
    a) .. math:: F^{*}(x^{*}) = \mathbb{I}_{\{\|\cdot\|_{\infty}\leq1\}}(x^{*}) 
    b) .. math:: F^{*}(x^{*}) = \mathbb{I}_{\{\|\cdot\|_{\infty}\leq1\}}(x^{*}) + \langle x^{*},b\rangle      
    

    .. math:: \mathbb{I}_{\{\|\cdot\|_{\infty}\leq1\}}(x^{*}) 
        = \begin{cases} 
        0, \mbox{if } \|x^{*}\|_{\infty}\leq1\\
        \infty, \mbox{otherwise}
        \end{cases}

    In the weighted case the convex conjugate is the indicator of the unit 
    :math:`L^{\infty}` norm.

    See:
    https://math.stackexchange.com/questions/1533217/convex-conjugate-of-l1-norm-function-with-weight
    
    a) .. math:: F^{*}(x^{*}) = \mathbb{I}_{\{\|\cdot\|_{L^\infty(w^{-1})}\leq 1\}}(x^{*})
    b) .. math:: F^{*}(x^{*}) = \mathbb{I}_{\{\|\cdot\|_{L^\infty(w^{-1})}\leq 1\}}(x^{*}) + \langle x^{*},b\rangle

    with :math:`\|x\|_{L^\infty(w^{-1})} = \max_{i} \frac{|x_i|}{w_i}`.

    Parameters
    -----------

    x : DataContainer
        where to evaluate the convex conjugate of the L1 Norm function.

    Returns
    --------
    the value of the convex conjugate of the WeightedL1Norm function at x: float

        """        
        return self.function.convex_conjugate(x)

    def proximal(self, x, tau, out=None):
        r"""Returns the value of the proximal operator of the L1 Norm function at x.
        
        
    Consider the following cases:
            
    a) .. math:: \mathrm{prox}_{\tau F}(x) = \mathrm{ShinkOperator}(x)
    b) .. math:: \mathrm{prox}_{\tau F}(x) = \mathrm{ShinkOperator}(x) + b   

    where,
    
    .. math :: \mathrm{prox}_{\tau F}(x) = \mathrm{ShinkOperator}(x) = sgn(x) * \max\{ |x| - \tau, 0 \}

    The weighted case follows from Example 6.23 in Chapter 6 of "First-Order Methods in Optimization"
    by Amir Beck, SIAM 2017 https://archive.siam.org/books/mo25/mo25_ch6.pdf
    
    a) .. math:: \mathrm{prox}_{\tau F}(x) = \mathrm{ShinkOperator}_{\tau*w}(x)
    b) .. math:: \mathrm{prox}_{\tau F}(x) = \mathrm{ShinkOperator}_{\tau*w}(x) + b

    
    Parameters
    -----------
    x: DataContainer
    tau: float, ndarray, DataContainer
    out: DataContainer, default None
        If not None, the result will be stored in this object.
    
    Returns
    --------
    The value of the proximal operator of the L1 norm function at x: DataContainer.
                        
        """  
        return self.function.proximal(x, tau, out=out)


class _L1Norm(Function):
    
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
        super().__init__()
        self.b = kwargs.get('b',None)
        
    def __call__(self, x):
        y = x
        if self.b is not None: 
            y = x - self.b
        return y.abs().sum()  
          
    def convex_conjugate(self,x):        
        tmp = x.abs().max() - 1
        if tmp<=1e-5:            
            if self.b is not None:
                return self.b.dot(x)
            else:
                return 0.
        return np.inf

                    
    def proximal(self, x, tau, out=None):
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

class _WeightedL1Norm(Function): 

    def __init__(self, weight, b=None):
        super().__init__()
        self.weight = weight
        self.b = b

        if np.min(weight) <= 0:
            raise ValueError("Weights should be strictly positive!")
        
    def __call__(self, x):
        y = x*self.weight

        if self.b is not None: 
            y -= self.b

        return y.abs().sum() 
          
    def convex_conjugate(self,x):
        tmp = (x.abs()/self.weight).max() - 1

        if tmp<=1e-5:            
            if self.b is not None:
                return self.b.dot(x)
            else:
                return 0.
        return np.inf

    def proximal(self, x, tau, out=None):
        tau *= self.weight
        ret = _L1Norm.proximal(self, x, tau, out=out)
        tau /= self.weight
        return ret
