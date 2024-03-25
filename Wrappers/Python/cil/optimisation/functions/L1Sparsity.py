# -*- coding: utf-8 -*-
#  Copyright 2023 United Kingdom Research and Innovation
#  Copyright 2023 The University of Manchester
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

from cil.optimisation.functions import Function, L1Norm, soft_shrinkage

import numpy as np
import pywt


class L1Sparsity(Function):

    r"""L1Sparsity function

    Calculates the following cases, depending on if the optional parameter `weight` is passed:        TODO: do we optionally want to pass data?  
            a) .. math:: F(x) = ||Ax||_{\ell^1}
            b) .. math:: F(x) = ||Ax||_{\ell^1(w)} (weighted norm)
    
    for a given orthogonal operator :math:`A`. 
    Parameters
    ---------
     A: orthogonal Operator, default = Identity operator 
        Note that for the correct calculation of the proximal the provided operator must be orthogonal 
    weight: array, optional, default=None
        weight array matching the size of the :math:`Ax`. 
    """

    def __init__(self, A, weight=None):
        '''creator

        Cases considered :            
        a) :math:`f(x) = ||Ax||_{\ell^1}`
        b) :math:`f(x) = ||Ax||_{\ell^1(w)}` (weighted norm)        
        '''

        if not A.is_orthogonal():
            raise AttributeError(
                f"Invalid operator: `{A}`. L1Sparsity is only defined for orthogonal operators!")

        super(L1Sparsity, self).__init__()
        self.A = A

        self.l1norm = L1Norm(weight=weight)

    def __call__(self, x):
        r"""Returns the value of the L1Sparsity function at x.

        Consider the following case:           
            a) .. math:: f(x) = ||Ax||_{\ell^1}     
            b) .. math:: f(x) = ||Ax||_{\ell^1(w)} (weighted norm)
        """
        y = self.A.direct(x)
        return self.l1norm(y)

    def convex_conjugate(self, x):
        r"""Returns the value of the convex conjugate of the L1Sparsity function at x.
        Here, we need to use the convex conjugate of L1Sparsity, which is the Indicator of the unit 
        :math:`\ell^{\infty}` norm on the operator domain. (Since A is a basis of L^2).

        Weighted case should be easy:
        https://math.stackexchange.com/questions/1533217/convex-conjugate-of-l1-norm-function-with-weight

        Consider the following cases:

                a) .. math:: F^{*}(x^{*}) = \mathbb{I}_{\{\|\cdot\|_{\ell^\infty}\leq 1\}}(A x^{*})    
                b) .. math:: F^{*}(x^{*}) = \mathbb{I}_{\{\|\cdot\|_{\ell^\infty(w^{-1})}\leq 1\}}(A x^{*})


        .. math:: \mathbb{I}_{\{\|\cdot\|_{\infty}\leq1\}}(x^{*}) 
            = \begin{cases} 
            0, \mbox{if } \|x^{*}\|_{\infty}\leq 1 \\
            \infty, \mbox{otherwise}
            \end{cases}

        """
        y = self.A.direct(x)
        return self.l1norm.convex_conjugate(y)

    def proximal(self, x, tau, out=None):
        r"""Returns the value of the proximal operator of the L1Sparsity function at x.

        Weighted case follows from Example 6.23 in Chapter 6 of "First-Order Methods in Optimization"
        by Amir Beck, SIAM 2017
        https://archive.siam.org/books/mo25/mo25_ch6.pdf

        Consider the following cases:

                a) .. math:: \mathrm{prox}_{\tau F}(x) = A^*\mathrm{ShinkOperator}_{\tau}(Ax)
                b) .. math:: \mathrm{prox}_{\tau F}(x) = A^*\mathrm{ShinkOperator}_{\tau*weight}(Ax)


        where,

        .. math :: \mathrm{prox}_{\tau F}(x) = \mathrm{ShinkOperator}_{\tau}(x) = sgn(x) * \max\{ |x| - \tau, 0 \}

        """
        y = self.A.direct(x)
        self.l1norm.proximal(y, tau, out=y)
        return self.A.adjoint(y, out)
