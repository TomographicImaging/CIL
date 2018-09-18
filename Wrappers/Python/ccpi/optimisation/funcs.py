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

from ccpi.optimisation.ops import Identity, FiniteDiff2D
import numpy


class Function(object):
    def __init__(self):
        self.op = Identity()
    def __call__(self,x):       return 0
    def grad(self, x):          return 0
    def prox(self, x, tau):     return x
    def gradient(self, x):      return self.grad(x)
    def proximal(self, x, tau): return self.prox(x, tau)

class Norm2(Function):
    
    def __init__(self, 
                 gamma=1.0, 
                 direction=None):
        super(Norm2, self).__init__()
        self.gamma     = gamma;
        self.direction = direction; 
    
    def __call__(self, x):
        
        xx = numpy.sqrt(numpy.sum(numpy.square(x.as_array()), self.direction,
                                  keepdims=True))
        p  = numpy.sum(self.gamma*xx)        
        
        return p
    
    def prox(self, x, tau):

        xx = numpy.sqrt(numpy.sum( numpy.square(x.as_array()), self.direction, 
                                  keepdims=True ))
        xx = numpy.maximum(0, 1 - tau*self.gamma / xx)
        p  = x.as_array() * xx
        
        return type(x)(p,geometry=x.geometry)

class TV2D(Norm2):
    
    def __init__(self, gamma):
        super(TV2D,self).__init__(gamma, 0)
        self.op = FiniteDiff2D()
        self.L = self.op.get_max_sing_val()
        

# Define a class for squared 2-norm
class Norm2sq(Function):
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
    
    def __call__(self,x):
        #return self.c* np.sum(np.square((self.A.direct(x) - self.b).ravel()))
        return self.c*( ( (self.A.direct(x)-self.b)**2).sum() )


class ZeroFun(Function):
    
    def __init__(self,gamma=0,L=1):
        self.gamma = gamma
        self.L = L
        super(ZeroFun, self).__init__()
    
    def __call__(self,x):
        return 0
    
    def prox(self,x,tau):
        return x

# A more interesting example, least squares plus 1-norm minimization.
# Define class to represent 1-norm including prox function
class Norm1(Function):
    
    def __init__(self,gamma):
        # Do nothing
        self.gamma = gamma
        self.L = 1
        super(Norm1, self).__init__()
    
    def __call__(self,x):
        return self.gamma*(x.abs().sum())
    
    def prox(self,x,tau):
        return (x.abs() - tau*self.gamma).maximum(0) * x.sign()

# Box constraints indicator function. Calling returns 0 if argument is within 
# the box. The prox operator is projection onto the box. Only implements one 
# scalar lower and one upper as constraint on all elements. Should generalise 
# to vectors to allow different constraints one elements.
class IndicatorBox(Function):
    
    def __init__(self,lower=-numpy.inf,upper=numpy.inf):
        # Do nothing
        self.lower = lower
        self.upper = upper
        super(IndicatorBox, self).__init__()
    
    def __call__(self,x):
        
        if (numpy.all(x.array>=self.lower) and 
            numpy.all(x.array <= self.upper) ):
            val = 0
        else:
            val = numpy.inf
        return val
    
    def prox(self,x,tau=None):
        return  (x.maximum(self.lower)).minimum(self.upper)
    
