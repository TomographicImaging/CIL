# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library developed by
#   Visual Analytics and Imaging System Group of the Science Technology
#   Facilities Council, STFC

#   Copyright 2018 Jakob Jorgensen, Daniil Kazantsev and Edoardo Pasca

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from ccpi.reconstruction.ops import Identity


class BaseFunction(object):
    def __init__(self):
        self.op = Identity()
    def fun(self,x):      return 0
    def grad(self,x):     return 0
    def prox(self,x,tau): return x    

# Define a class for 2-norm
class Norm2sq(BaseFunction):
    '''
    f(x) = c*||A*x-b||_2^2
    
    which has 
    
    grad[f](x) = 2*c*A^T*(A*x-b)
    
    and Lipschitz constant
    
    L = 2*c*||A||_2^2 = 2*s1(A)^2
    
    where s1(A) is the largest singular value of A.
    
    '''
    
    def __init__(self,A,b,c=1.0):
        self.A = A  # Should be an operator, default identity
        self.b = b  # Default zero DataSet?
        self.c = c  # Default 1.
        
        # Compute the Lipschitz parameter from the operator.
        # Initialise to None instead and only call when needed.
        self.L = 2.0*self.c*(self.A.get_max_sing_val()**2)
        super(Norm2sq, self).__init__()
    
    def grad(self,x):
        #return 2*self.c*self.A.adjoint( self.A.direct(x) - self.b )
        return 2.0*self.c*self.A.adjoint( self.A.direct(x) - self.b )
    
    def fun(self,x):
        #return self.c* np.sum(np.square((self.A.direct(x) - self.b).ravel()))
        return self.c*( ( (self.A.direct(x)-self.b)**2).sum() )

## Define a class to represent a least squares data fidelity
#class LeastSquares(BaseFunction):
#    
#    b     = None
#    A     = None
#    L     = None
#    
#    def __init__(self, A, b):
#        self.A    = A
#        #b.shape = (b.shape[0],1)
#        self.b    = b
#        
#        # Compute the Lipschitz parameter from the operator
#        # Initialise to None instead and only call when needed.
#        self.L = self.A.get_max_sing_val()**2
#    
#    def grad(self, x):
#        #return np.dot(self.A.transpose(),  (np.dot(self.A,x) - self.b)  )
#        return self.A.adjoint( self.A.direct(x) - self.b )
#    
#    def fun(self, x):
#        # p = np.dot(self.A, x)
#        return 0.5*( ( (self.A.direct(x)-self.b)**2).sum() )

# Define a class to represent the zero-function, to test pure least squares 
# minimization using FISTA
class ZeroFun(BaseFunction):
    
    def __init__(self,gamma=0,L=1):
        self.gamma = gamma
        self.L = L
        super(ZeroFun, self).__init__()
    
    def fun(self,x):
        return 0
    
    def prox(self,x,tau):
        return x

# A more interesting example, least squares plus 1-norm minimization.
# Define class to represent 1-norm including prox function
class Norm1(BaseFunction):
    
    def __init__(self,gamma):
        # Do nothing
        self.gamma = gamma
        self.L = 1
        super(Norm1, self).__init__()
    
    def fun(self,x):
        return self.gamma*(x.abs().sum())
    
    def prox(self,x,tau):
        return (x.abs() - tau*self.gamma).maximum(0) * x.sign()
    
