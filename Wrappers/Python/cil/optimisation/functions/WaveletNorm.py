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
from cil.optimisation.functions.L1Norm import soft_shrinkage
###############################################################################
###############################################################################
####################### L1-norm of Wavelet Coefficients #######################
###############################################################################
###############################################################################    
    
class WaveletNorm(Function):
    
    r"""WaveletNorm function
            
        Consider the following case:           
            a) .. math:: F(x) = ||Wx||_{\ell^1}
            b) .. math:: F(x) = ||Wx||_{\ell^1(w)} (weighted norm)
                                
    """   
           
    def __init__(self, W, weight = None):
        '''creator

        Cases considered :            
        a) :math:`f(x) = ||Wx||_{\ell^1}`
        b) :math:`f(x) = ||Wx||_{\ell^1(w)}` (weighted norm)

        :param W: Wavelet transform
        :type W: :code:`WaveletOperator`

        [OPTIONAL PARAMETERS]
        :param weight: weight array matching the size of the wavelet coefficients
        '''
        super(WaveletNorm, self).__init__()
        self.W = W
        self.weight = weight

        if (weight is not None) and (np.min(weight) <= 0):
            raise ValueError("Weights should be strictly positive!")
        
    def __call__(self, x):
        
        r"""Returns the value of the WaveletNorm function at x.
        
        Consider the following case:           
            a) .. math:: f(x) = ||Wx||_{\ell^1}     
            b) .. math:: f(x) = ||Wx||_{\ell^1(w)} (weighted norm)
        """
        if self.weight is None:
            y = self.W.direct(x)
        else:
            y = self.W.direct(x)*self.weight

        return y.abs().sum()  
          
    def convex_conjugate(self,x):
        
        r"""Returns the value of the convex conjugate of the WaveletNorm function at x.
        Here, we need to use the convex conjugate of WaveletNorm, which is the Indicator of the unit 
        :math:`\ell^{\infty}` norm on the Wavelet domain. (Since W is a basis of L^2).

        Weighted case should be easy:
        https://math.stackexchange.com/questions/1533217/convex-conjugate-of-l1-norm-function-with-weight
        
        Consider the following cases:
                
                a) .. math:: F^{*}(x^{*}) = \mathbb{I}_{\{\|\cdot\|_{\ell^\infty}\leq 1\}}(W x^{*})    
                b) .. math:: F^{*}(x^{*}) = \mathbb{I}_{\{\|\cdot\|_{\ell^\infty(w^{-1})}\leq 1\}}(W x^{*})
        
    
        .. math:: \mathbb{I}_{\{\|\cdot\|_{\infty}\leq1\}}(x^{*}) 
            = \begin{cases} 
            0, \mbox{if } \|x^{*}\|_{\infty}\leq 1 \\
            \infty, \mbox{otherwise}
            \end{cases}
    
        """        
        if self.weight is None:
            tmp = self.W.direct(x).abs().max() - 1
        else:
            tmp = (self.W.direct(x).abs/self.weight).max() - 1

        if tmp<=1e-5:            
            return 0.
        return np.inf

                    
    def proximal(self, x, tau, out=None):
        
        r"""Returns the value of the proximal operator of the WaveletNorm function at x.
        
        Weighted case follows from Example 6.23 in Chapter 6 of "First-Order Methods in Optimization"
        by Amir Beck, SIAM 2017
        https://archive.siam.org/books/mo25/mo25_ch6.pdf

        Consider the following cases:
                
                a) .. math:: \mathrm{prox}_{\tau F}(x) = W^*\mathrm{ShinkOperator}_{\tau}(Wx)
                b) .. math:: \mathrm{prox}_{\tau F}(x) = W^*\mathrm{ShinkOperator}_{\tau*weight}(Wx)

    
        where,
        
        .. math :: \mathrm{prox}_{\tau F}(x) = \mathrm{ShinkOperator}_{\tau}(x) = sgn(x) * \max\{ |x| - \tau, 0 \}
                            
        """  
        if self.weight is not None:
            y = soft_shrinkage(self.W.direct(x), tau*self.weight)
        else:
            y = soft_shrinkage(self.W.direct(x), tau)

        if out is None:                                              
            return self.W.adjoint(y)
        else: 
            self.W.adjoint(y, out=out)