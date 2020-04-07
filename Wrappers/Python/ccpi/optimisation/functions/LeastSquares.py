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
    
    def __init__(self, A, b, c=1.0, estimate_Lipschitz = True):
        super(LeastSquares, self).__init__()
    
        self.A = A  # Should be an operator, default identity
        self.b = b  # Default zero DataSet?
        self.c = c  # Default 1.
        self.range_tmp = A.range_geometry().allocate()
        
        if estimate_Lipschitz:
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

        self.A.direct(x, out=self.range_tmp)
        self.range_tmp.subtract(self.b, out=self.range_tmp)
        try:
            return self.range_tmp.squared_norm() * self.c
        except AttributeError as ae:
            # added for compatibility with SIRF
            warnings.warn('squared_norm method not found! Proceeding with norm.')
            yn =  self.range_tmp.norm()
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
            
        
if __name__ == '__main__':
    
    
    
    print("Check LeastSquares")
    from ccpi.framework import ImageGeometry
    from ccpi.optimisation.operators import Identity
    from ccpi.optimisation.functions import L2NormSquared, FunctionOperatorComposition
    import numpy
    
    ig = ImageGeometry(4,5)
    Aop = 2*Identity(ig)
    data = ig.allocate('random')
    
    alpha = 0.4
    f = LeastSquares(Aop, data, c = alpha)
    x = ig.allocate('random')
    
    res1 = f(x)
    res2 = alpha * ((Aop.direct(x) - data)**2).sum()
    numpy.testing.assert_almost_equal(res1, res2, decimal=5) 
    print("Checking call .... OK ")
    
    res1 = f.gradient(x)
    res2 = 2*alpha * Aop.adjoint(Aop.direct(x) - data)
    numpy.testing.assert_almost_equal(res1.as_array(), res2.as_array(), decimal=4) 
    print("Checking gradient .... OK ")
                
    print("Check LeastSquares with FunctionOperatorComposition")        
    ig = ImageGeometry(4,5)
    Aop = 2 * Identity(ig)
    data = ig.allocate('random')
    
    x = ig.allocate('random')
    alpha = 5
    tmp = alpha * L2NormSquared(b=data)
    f1 = FunctionOperatorComposition(tmp, Aop)
    f2 = LeastSquares(Aop, data, c = alpha)
    res1 = f1(x)
    res2 = f2(x)
    
    numpy.testing.assert_almost_equal(res1, res2)   
    print("Checking call .... OK ")
    
    res1 = f1.gradient(x)
    res2 = f2.gradient(x)
    numpy.testing.assert_array_almost_equal(res1.as_array(), res2.as_array())
    print("Checking gradient .... OK ")


    
    

    


    
    
    
    
    
    
    
    

    
    
    
    
        

    
    
    
