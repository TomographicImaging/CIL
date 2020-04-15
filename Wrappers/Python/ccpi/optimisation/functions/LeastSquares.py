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

from ccpi.optimisation.operators import LinearOperator
from ccpi.optimisation.functions import Function
from ccpi.framework import DataContainer
import warnings


class LeastSquares(Function):
    
    r"""Least Squares function
    
    .. math:: F(x) = c\|Ax-b\|_2^2
    
    Parameters:
        
        A : Operator
        
        c : Scaling Constant
        
        b : Data
        
    L : Lipshitz Constant of the gradient of :math:`F` which is :math:`2c||A||_2^2 = 2s1(A)^2`,
    
    where s1(A) is the largest singular value of A.
        
    
    """
    
    def __init__(self, A, b, c=1.0):
        super(LeastSquares, self).__init__()
    
        self.A = A  # Should be an operator, default identity
        self.b = b  # Default zero DataSet?
        self.c = c  # Default 1.
        self.range_tmp = A.range_geometry().allocate()

        # Compute the Lipschitz parameter from the operator if possible
        # Leave it initialised to None otherwise
        try:
            self.L = 2.0*self.c*(self.A.norm()**2)
        except AttributeError as ae:
            if self.A.is_linear():
                Anorm = LinearOperator.PowerMethod(self.A, 10)[0]
                self.L = 2.0 * self.c * (Anorm*Anorm)
            else:
                warnings.warn('{} could not calculate Lipschitz Constant. {}'.format(
                self.__class__.__name__, ae))
            
        except NotImplementedError as noe:
            warnings.warn('{} could not calculate Lipschitz Constant. {}'.format(
                self.__class__.__name__, noe))
                
        
    def __call__(self, x):
        
        r""" Returns the value of :math:`F(x) = c\|Ax-b\|_2^2`
        """

        y = self.A.direct(x)
        y.subtract(self.b, out=y)
        try:
            return y.squared_norm() * self.c
        except AttributeError as ae:
            # added for compatibility with SIRF
            warnings.warn('squared_norm method not found! Proceeding with norm.')
            yn = y.norm()
            if self.c == 1:
                return yn * yn
            return (yn * yn) * self.c
    
    def gradient(self, x, out=None):
        
        r""" Returns the value of the gradient of :math:`F(x) = c*\|A*x-b\|_2^2`
        
             .. math:: F'(x) = 2cA^T(Ax-b)

        """
        
        if out is not None:
            self.A.direct(x, out=self.range_tmp)
            self.range_tmp.subtract(self.b , out=self.range_tmp)
            self.A.adjoint(self.range_tmp, out=out)
            out.multiply (self.c * 2.0, out=out)
        else:
            return (2.0*self.c)*self.A.adjoint(self.A.direct(x) - self.b)
        
        
class WeightedLeastSquares(Function):
                
    # The class is implemented so, LeastSquares and WeightedLS can be in one class
    # and have default a weight parameter = 1.0, So no need to have two classes
    # Do you agree
                    
    def __init__(self, A, b, c = 1.0, weight = 1.0, **kwargs): 
        
        super(WeightedLeastSquares, self).__init__()             
                               
        self.A = A
        self.b = b
        self.c = c
                    
        self.weight = weight      
        self.weight_norm = 1.0
        
        self.tmp_space = A.range_geometry().allocate() 
                                
        if isinstance(self.weight, DataContainer):
            if (self.weight<0).any():
                raise ValueError('Weigth contains negative values') 
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
        
        self.A.direct(x, out = self.tmp_space)
        self.tmp_space.subtract(self.b, out = self.tmp_space)
        
        return self.c * self.tmp_space.dot(self.weight * self.tmp_space)
    
    def gradient(self, x):
        
        self.A.direct(x, out = self.tmp_space)
        self.tmp_space.subtract(self.b, out = self.tmp_space)
                    
        return 2 * self.c * self.weight * self.tmp_space.power(2)        
        
             
if __name__ == "__main__" :
    
    from ccpi.optimisation.operators import  Identity, DiagonalOperator
    from ccpi.framework import ImageGeometry
    import numpy

                        
    ig = ImageGeometry(3,3)
    A = Identity(ig)
    b = ig.allocate('random')
    x = ig.allocate('random')
    c = 0.3
    
    weight = ig.allocate('random') 
    
    D = DiagonalOperator(weight)
    norm_weight = D.norm()
    
    f1 = WeightedLeastSquares(A, b, c, weight) 
    f2 = WeightedLeastSquares(A, b, c) 
    
    # check Lipshitz    
    numpy.testing.assert_almost_equal(f2.L, 2 * c * A.norm()**2)   
    numpy.testing.assert_almost_equal(f1.L, 2 * c * norm_weight * A.norm()**2)   
        
    # check call with weight    
           
    res1 = c * (A.direct(x)-b).dot(weight * (A.direct(x) - b))
    res2 = f1(x)    
    numpy.testing.assert_almost_equal(res1, res2)
    
    # check call without weight   
           
    res1 = c * (A.direct(x)-b).dot((A.direct(x) - b))
    res2 = f2(x)    
    numpy.testing.assert_almost_equal(res1, res2) 
    
    # check gradient with weight     
    
    res1 = f1.gradient(x)
    res2 = 2 * c * weight * (A.direct(x)-b).power(2)
    numpy.testing.assert_array_almost_equal(res1.as_array(), res2.as_array())
    
    # check gradient without weight     
    
    res1 = f2.gradient(x)
    res2 = 2 * c * (A.direct(x)-b).power(2)
    numpy.testing.assert_array_almost_equal(res1.as_array(), res2.as_array()) 
    

    
    
    
    
    
    
    
    
