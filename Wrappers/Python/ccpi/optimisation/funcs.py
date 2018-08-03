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
from ccpi.framework import DataContainer


def isSizeCorrect(data1 ,data2):
    if issubclass(type(data1), DataContainer) and \
       issubclass(type(data2), DataContainer):
        # check dimensionality
        if data1.check_dimensions(data2):
            return True
    elif issubclass(type(data1) , numpy.ndarray) and \
         issubclass(type(data2) , numpy.ndarray):
        return data1.shape == data2.shape
    else:
        raise ValueError("{0}: getting two incompatible types: {1} {2}"\
                         .format('Function', type(data1), type(data2)))
    return False
        
class Function(object):
    def __init__(self):
        self.op = Identity()
    def __call__(self,x, out=None):       return 0
    def grad(self, x):                    return 0
    def prox(self, x, tau):               return x
    def gradient(self, x, out=None):      return self.grad(x)
    def proximal(self, x, tau, out=None): return self.prox(x, tau)

class Norm2(Function):
    
    def __init__(self, 
                 gamma=1.0, 
                 direction=None):
        super(Norm2, self).__init__()
        self.gamma     = gamma;
        self.direction = direction; 
    
    def __call__(self, x, out=None):
        
        if out is None:
            xx = numpy.sqrt(numpy.sum(numpy.square(x.as_array()), self.direction,
                                  keepdims=True))
        else:
            if isSizeCorrect(out, x):
                # check dimensionality
                if issubclass(type(out), DataContainer):
                    arr = out.as_array()
                    numpy.square(x.as_array(), out=arr)
                    xx = numpy.sqrt(numpy.sum(arr, self.direction, keepdims=True))
                        
                elif issubclass(type(out) , numpy.ndarray):
                    numpy.square(x.as_array(), out=out)
                    xx = numpy.sqrt(numpy.sum(out, self.direction, keepdims=True))
            else:
                raise ValueError ('Wrong size: x{0} out{1}'.format(x.shape,out.shape) )
        
        p  = numpy.sum(self.gamma*xx)        
        
        return p
    
    def prox(self, x, tau):

        xx = numpy.sqrt(numpy.sum( numpy.square(x.as_array()), self.direction, 
                                  keepdims=True ))
        xx = numpy.maximum(0, 1 - tau*self.gamma / xx)
        p  = x.as_array() * xx
        
        return type(x)(p,geometry=x.geometry)
    def proximal(self, x, tau, out=None):
        if out is None:
            return self.prox(x,tau)
        else:
            if isSizeCorrect(out, x):
                # check dimensionality
                if issubclass(type(out), DataContainer):
                    xx = numpy.sqrt(numpy.sum( numpy.square(x.as_array()), self.direction, 
                                  keepdims=True ))
                    xx = numpy.maximum(0, 1 - tau*self.gamma / xx)
                    p  = x.as_array() * xx
                    
                    arr = out.as_array()
                        
                elif issubclass(type(out) , numpy.ndarray):
                    numpy.square(x.as_array(), out=out)
                    xx = numpy.sqrt(numpy.sum(out, self.direction, keepdims=True))
            else:
                raise ValueError ('Wrong size: x{0} out{1}'.format(x.shape,out.shape) )
        
            

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
    
    def gradient(self, x, out=None):
        if out is None:
            return self.grad(x)
        else:
            #return self.c*( ( (self.A.direct(x)-self.b)**2).sum() )
            y = self.A.direct(x)
            y -= self.b
            y *= y
            return y.sum() * self.c
            


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
        self.gamma = gamma
        self.L = 1
        super(Norm1, self).__init__()
    
    def __call__(self,x):
        return self.gamma*(x.abs().sum())
    
    def prox(self,x,tau):
        return (x.abs() - tau*self.gamma).maximum(0) * x.sign()
    
    def proximal(self, x, tau, out=None):
        if out is None:
            return self.prox(x, tau)
        else:
            #(x.abs() - tau*self.gamma).maximum(0) * x.sign()
            y = x.abs()
            # here there is a new allocation of memory for the product
            y -= (tau*self.gamma)
            y.maximum(0, out=y)
            x.sign(out=x)
            y *= x
            return y
    
