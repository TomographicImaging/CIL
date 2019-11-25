# -*- coding: utf-8 -*-
#========================================================================
# Copyright 2019 Science Technology Facilities Council
# Copyright 2019 University of Manchester
#
# This work is part of the Core Imaging Library developed by Science Technology
# Facilities Council and University of Manchester
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#=========================================================================

from ccpi.optimisation.functions import Function
from ccpi.optimisation.operators import Operator, ScaledOperator

import warnings

class FunctionOperatorComposition(Function):
    
    """ Composition of a function with an operator as : :math:`(F o A)(x) = F(Ax)`
    
            : parameter function : function F
            : parameter operator : operator A
            
            
        For general operator, we have no explicit formulas for convex_conjugate,
        proximal and proximal_conjugate            
    
    """
    
    def __init__(self, function, operator):
        
        super(FunctionOperatorComposition, self).__init__()
        
        self.function = function     
        self.operator = operator
        
        if not isinstance(self.function, Function):
            raise ValueError('{} is not function '.format(type(self.function)))
          
        # check also ScaledOperator because it's not a child atm of Operator            
        if not isinstance(self.operator, (Operator, ScaledOperator)):
            raise ValueError('{} is not function '.format(type(self.operator)))            
        
        try:
            self.L = function.L * operator.norm()**2 
        except ValueError as er:
            self.L = None
            warnings.warn("Lipschitz constant was not calculated")
        
    def __call__(self, x):
        
        """ Returns :math:`F(Ax)`
        """
    
        return self.function(self.operator.direct(x))  
    
    def gradient(self, x, out=None):
        
        """ Return the gradient of F(Ax), 
        
        ..math ::  (F(Ax))' = A^{T}F'(Ax)
            
        """
        
        tmp = self.operator.range_geometry().allocate()
        self.operator.direct(x, out=tmp)
        self.function.gradient(tmp, out=tmp)
        if out is None:
            return self.operator.adjoint(tmp)
        else: 
            self.operator.adjoint(tmp, out=out)

