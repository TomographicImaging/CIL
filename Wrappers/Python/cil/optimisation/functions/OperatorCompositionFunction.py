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


class OperatorCompositionFunction(Function):
    
    """ OperatorCompositionFunction

    :math:`F = H \circ A, (H \circ A)(\cdot) = H(A(\cdot))`

    Parameters
    ----------

    function : Function
               A function to be composed with an operator
    operator : Operator
               A general operator


    Note
    ----
    For a general operator, we have no explicit formulas for the :code:`convex_conjugate`,
    the :code:`proximal` and :code:`proximal_conjugate` methods.

    .. todo:: If the operator is unitary, i.e., :math:`A\circ A^{T} = \mathbb{I}`, there is an explicit formula for these methods.

    See Chapter 6 in :cite:`Beck2017` and Tables 1,2 in :cite:`Combettes2011`.

    Examples
    --------
    >>> F = MixedL21Norm()
    >>> G = GradientOperator(ig)
    >>> H = OperatorCompositionFunction(F, G) # The L21 norm of the gradient, i.e., the TotalVariation 
    
    """
    
    def __init__(self, function, operator):

        super(OperatorCompositionFunction, self).__init__()    
        
        self.function = function     
        self.operator = operator

    @property
    def L(self):
        if self._L is None:
            try:
                self._L = self.function.L  * (self.operator.norm() ** 2)
            except ValueError as ve:
                self._L = None
        return self._L
    
    def __call__(self, x):
        
        """ Returns the value of the OperatorCompositionFunction at :code:`x`.

        :math:`(H \circ A)(x) = H(A(x))`

        """
    
        return self.function(self.operator.direct(x))  
    
    def gradient(self, x, out=None):

        r"""Returns the value of the gradient of the OperatorCompositionFunction at :code:`x`.
        
        :math:`(H(Ax))' = A^{T}H'(Ax)`
            
        """
        
        tmp = self.operator.range_geometry().allocate()
        self.operator.direct(x, out=tmp)
        self.function.gradient(tmp, out=tmp)
        if out is None:
            return self.operator.adjoint(tmp)
        else: 
            self.operator.adjoint(tmp, out=out)
