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

from cil.optimisation.functions import Function, L1Norm
import warnings

class L1Sparsity(Function):

    r"""L1Sparsity function

    Calculates the following cases, depending on if the optional parameter `weight`  or data `b` is passed. For `weight=None`: 


    a) .. math:: F(x) = ||Qx||_{1}
    b) .. math:: F(x) = ||Qx - b||_{1}

    In the weighted case, `weight` = :math:`w` is an array of positive weights.

    a) .. math:: F(x) = ||Qx||_{L^1(w)}
    b) .. math:: F(x) = ||Qx - b||_{L^1(w)}

    with :math:`||x||_{L^1(w)} = || x \cdot w||_1 = \sum_{i=1}^{n} |x_i| w_i`.
    
    In all cases :math:`Q` is an orthogonal operator. 
    
    Parameters
    ---------
    Q: orthogonal Operator 
        Note that for the correct calculation of the proximal the provided operator must be orthogonal 
    b : Data, DataContainer, default is None 
    weight: array, optional, default=None
        positive weight array matching the size of the range of operator :math:`Q`.
    """

    def __init__(self, Q, b=None, weight=None):
        '''creator
        '''

        if not Q.is_orthogonal(): 
            warnings.warn(
                f"Invalid operator: `{Q}`. L1Sparsity is only defined for orthogonal operators!", UserWarning)

        super(L1Sparsity, self).__init__()
        self.Q = Q

        self.l1norm = L1Norm(b=b, weight=weight)

    def __call__(self, x):
        r"""Returns the value of the L1Sparsity function at x.

        Consider the following cases:

        a) .. math:: F(x) = ||Qx||_{1}
        b) .. math:: F(x) = ||Qx - b||_{1}

        In the weighted case, `weight` = :math:`w` is an array of positive weights.

        a) .. math:: F(x) = ||Qx||_{L^1(w)}
        b) .. math:: F(x) = ||Qx - b||_{L^1(w)}

        with :math:`||x||_{L^1(w)} = || x w||_1 = \sum_{i=1}^{n} |x_i| w_i`.
            
        """
        y = self.Q.direct(x)
        return self.l1norm(y)

    def convex_conjugate(self, x):
        r"""Returns the value of the convex conjugate of the L1Sparsity function at x.
        Here, we need to use the convex conjugate of L1Sparsity, which is the Indicator of the unit 
        :math:`\ell^{\infty}` norm on the operator domain. (Since Q is a basis of L^2).


        Consider the non-weighted case: 


        a) .. math:: F^{*}(x^{*}) = \mathbb{I}_{\{\|\cdot\|_{\infty}\leq1\}}(Qx^{*})
        b) .. math:: F^{*}(x^{*}) = \mathbb{I}_{\{\|\cdot\|_{\infty}\leq1\}}(Qx^{*}) + \langle Qx^{*},b\rangle


        .. math:: \mathbb{I}_{\{\|\cdot\|_{\infty}\leq1\}}(x^{*})
            = \begin{cases}
            0, \mbox{if } \|x^{*}\|_{\infty}\leq1\\
            \infty, \mbox{otherwise}
            \end{cases}

        In the weighted case the convex conjugate is the indicator of the unit
        :math:`L^{\infty}` norm.

        See:
        https://math.stackexchange.com/questions/1533217/convex-conjugate-of-l1-norm-function-with-weight

        a) .. math:: F^{*}(x^{*}) = \mathbb{I}_{\{\|\cdot\|_{L^\infty(w^{-1})}\leq 1\}}(Qx^{*})
        b) .. math:: F^{*}(x^{*}) = \mathbb{I}_{\{\|\cdot\|_{L^\infty(w^{-1})}\leq 1\}}(Qx^{*}) + \langle Qx^{*},b\rangle

        with :math:`\|x\|_{L^\infty(w^{-1})} = \max_{i} \frac{|x_i|}{w_i}`.
    
    
        """
        y = self.Q.direct(x)
        return self.l1norm.convex_conjugate(y)

    def proximal(self, x, tau, out=None):

        r"""Returns the value of the proximal operator of the L1 Norm function at x with scaling parameter `tau`.


        Consider the following cases:

        a) .. math:: \mathrm{prox}_{\tau F}(x) = Q^T \mathrm{ShinkOperator}_{\tau}(Qx)
        b) .. math:: \mathrm{prox}_{\tau F}(x) = Q^T \left( \mathrm{ShinkOperator}_\tau(Qx- b) + b \right)

        where,

        .. math :: \mathrm{prox}_{\tau | \cdot |}(x) = \mathrm{ShinkOperator}(x) = sgn(x) * \max\{ |x| - \tau, 0 \}

        The weighted case follows from Example 6.23 in Chapter 6 of "First-Order Methods in Optimization"
        by Amir Beck, SIAM 2017 https://archive.siam.org/books/mo25/mo25_ch6.pdf

        a) .. math:: \mathrm{prox}_{\tau F}(x) = Q^T \mathrm{ShinkOperator}_{\tau*w}(Qx)
        b) .. math:: \mathrm{prox}_{\tau F}(x) = Q^T \left( \mathrm{ShinkOperator}_{\tau*w}(Qx-b) + b \right)


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
        y = self.Q.direct(x)
        self.l1norm.proximal(y, tau, out=y)
        return self.Q.adjoint(y, out)
