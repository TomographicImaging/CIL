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
from numbers import Number


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
    
    def __init__(self, A, b, c=1.0, weight = None):
        super(LeastSquares, self).__init__()
    
        self.A = A  # Should be an operator, default identity
        self.b = b  # Default zero DataSet?
        self.c = c  # Default 1.
        
        # weight
        self.weight = weight      
        self._weight_norm = None

        if weight is not None:
            if (self.weight<0).any():
                raise ValueError('Weight contains negative values') 
            
        
    def __call__(self, x):
        
        r""" Returns the value of :math:`F(x) = c\|Ax-b\|_2^2` or c\|Ax-b\|_{2,weight}^2
                        
        """
        # c * (A.direct(x)-b).dot((A.direct(x) - b))
        y = self.A.direct(x)
        y.subtract(self.b, out = y) 
        
        if self.weight is None:    
            return self.c * y.dot(y)
        else:
            wy = self.weight.multiply(y)
            return self.c * y.dot(wy) 

    def gradient(self, x, out=None):
        
        r""" Returns the value of the gradient of :math:`F(x) = c*\|A*x-b\|_2^2`
        
             .. math:: F'(x) = 2cA^T(Ax-b)
             
             .. math:: F'(x) = 2cA^T(weight(Ax-b))

        """
        
        if out is not None:
            #return 2.0*self.c*self.A.adjoint( self.A.direct(x) - self.b )
            tmp = self.A.direct(x)
            tmp.subtract(self.b , out=tmp)
            if self.weight is not None:
                tmp.multiply(self.weight, out=tmp)
            self.A.adjoint(tmp, out = out)
            out.multiply(self.c * 2.0, out=out)
        else:
            if self.weight is None:
                return (2.0*self.c)*self.A.adjoint(self.A.direct(x) - self.b)
            else:
                return (2.0*self.c)*self.A.adjoint(self.weight * (self.A.direct(x) - self.b))
        
    @property
    def L(self):
        if self._L is None:
            self.calculate_Lipschitz()
        return self._L
    @L.setter
    def L(self, value):
        warnings.warn("You should set the Lipschitz constant with calculate_Lipschitz().")
        if isinstance(value, (Number,)) and value >= 0:
            self._L = value
        else:
            raise TypeError('The Lipschitz constant is a real positive number')

    def calculate_Lipschitz(self):
        # Compute the Lipschitz parameter from the operator if possible
        # Leave it initialised to None otherwise
        try:
            self._L = 2.0*self.c*(self.A.norm()**2)
        except AttributeError as ae:
            if self.A.is_linear():
                Anorm = LinearOperator.PowerMethod(self.A, 10)[0]
                self._L = 2.0 * self.c * (Anorm*Anorm)
            else:
                warnings.warn('{} could not calculate Lipschitz Constant. {}'.format(
                self.__class__.__name__, ae))
        except NotImplementedError as noe:
            warnings.warn('{} could not calculate Lipschitz Constant. {}'.format(
                self.__class__.__name__, noe))
        if self.weight is not None:
                self._L *= self.weight_norm
    @property
    def weight_norm(self):
        if self.weight is not None:
            if self._weight_norm is None:
                D = DiagonalOperator(self.weight)
                self._weight_norm = D.norm()
        else:
            self._weight_norm = 1.0
        return self._weight_norm


class StochasticNorm2Sq(LeastSquares):
    def __init__(self, A, b, c=1.0, number_of_subsets=1):
        super(StochasticNorm2Sq, self).__init__(A, b, c)
       
    def notify_new_subset(self, subset_id, number_of_subsets):
        self.A.notify_new_subset(subset_id, number_of_subsets)
