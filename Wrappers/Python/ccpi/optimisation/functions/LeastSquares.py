# -*- coding: utf-8 -*-
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ccpi.optimisation.operators import LinearOperator, DiagonalOperator
from ccpi.optimisation.functions import Function
from ccpi.framework import DataContainer
import warnings


class LeastSquares(Function):
    
    
    r""" (Weighted) Least Squares function
    
    .. math:: F(x) = c\|Ax-b\|_2^2 
    
    or
    
    .. math:: F(x) = c\|Ax-b\|_{2,W}^{2}
    
    Parameters:
        
        A : Operator
        
        c : Scaling Constant
        
        b : Data
        
        weight: 1.0 (Default) or DataContainer
        
    Members:        
            
        L : Lipshitz Constant of the gradient of :math:`F` which is :math:`2 c ||A||_2^2 = 2 c s1(A)^2`, or
        
        L : Lipshitz Constant of the gradient of :math:`F` which is :math:`2 c ||weight|| ||A||_2^2 = 2s1(A)^2`,
    
    where s1(A) is the largest singular value of A.
       
    
    """
    
    def __init__(self, A, b, c=1.0, weight = 1.0):
        super(LeastSquares, self).__init__()
    
        self.A = A  # Should be an operator, default identity
        self.b = b  # Default zero DataSet?
        self.c = c  # Default 1.
        
        # weight
        self.weight = weight      
        self.weight_norm = 1.0
        
        # allocate default spaces
        self.tmp_space = A.range_geometry().allocate()
        self.tmp_space1 = self.tmp_space.copy()
                                
        if isinstance(self.weight, DataContainer):
            if (self.weight<0).any():
                raise ValueError('Weight contains negative values') 
            D = DiagonalOperator(self.weight)
            self.weight_norm = D.norm()                    
                
        # Compute the Lipschitz parameter from the operator if possible
        # Leave it initialised to None otherwise
        try:
            self.L = 2.0 * self.c * (self.weight_norm * self.A.norm()**2)
        except AttributeError as ae:
            if self.A.is_linear():
                Anorm = LinearOperator.PowerMethod(self.A, 10)[0]
                self.L = 2.0 * self.c * ( self.weight_norm * Anorm*Anorm)
            else:
                warnings.warn('{} could not calculate Lipschitz Constant. {}'.format(
                self.__class__.__name__, ae))
            
        except NotImplementedError as noe:
            warnings.warn('{} could not calculate Lipschitz Constant. {}'.format(
                self.__class__.__name__, noe))          
                
        
    def __call__(self, x):
        
        r""" Returns the value of :math:`F(x) = c\|Ax-b\|_2^2` or c\|Ax-b\|_{2,weight}^2
                        
        """
        
        self.A.direct(x, out = self.tmp_space)
        self.tmp_space.subtract(self.b, out = self.tmp_space) 
                
        if isinstance(self.weight, DataContainer):    
            self.weight.multiply(self.tmp_space, out=self.tmp_space1)
            return self.c * self.tmp_space.dot(self.tmp_space1) 
        else:
            return self.c * self.tmp_space.dot(self.tmp_space)
        
    
    def gradient(self, x, out=None):
        
        r""" Returns the value of the gradient of :math:`F(x) = c*\|A*x-b\|_2^2`
        
             .. math:: F'(x) = 2cA^T(Ax-b)
             
             .. math:: F'(x) = 2cA^T(weight(Ax-b))

        """
        
        if out is not None:
            self.A.direct(x, out = out)
            out.subtract(self.b , out=out)
            if isinstance(self.weight, DataContainer):
                out *= self.weight           
            self.A.adjoint(out, out = out)
            out.multiply(self.c * 2.0, out=out)
        else:
            return (2.0*self.c)*self.A.adjoint(self.weight * (self.A.direct(x) - self.b))        