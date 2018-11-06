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
        pass
    def __call__(self,x, out=None):       raise NotImplementedError 
    def grad(self, x):                    raise NotImplementedError
    def prox(self, x, tau):               raise NotImplementedError
    def gradient(self, x, out=None):      raise NotImplementedError
    def proximal(self, x, tau, out=None): raise NotImplementedError

class Norm2(Function):
    
    def __init__(self, 
                 gamma=1.0, 
                 direction=None):
        super(Norm2, self).__init__()
        self.gamma     = gamma;
        self.direction = direction; 
    
    def __call__(self, x):
        
        #if out is None:
        #    xx = numpy.sqrt(numpy.sum(numpy.square(x.as_array()), self.direction,
        #                          keepdims=True))
        if issubclass(type(x), DataContainer):
            #arr = out.as_array()
            #numpy.square(x.as_array(), out=arr)
            #xx = numpy.sqrt(numpy.sum(arr, self.direction, keepdims=True))
            
            xx = numpy.sqrt(
                    x.power(2)
                     .sum(axis=self.direction, keepdims=False)
                )    
        elif issubclass(type(x) , numpy.ndarray):
            
            xx = numpy.sqrt(numpy.sum(numpy.square(x), self.direction, keepdims=False))
        
        p  = numpy.sum(self.gamma*xx)        
        
        return p
    
    def prox(self, x, tau):

        #xx = numpy.sqrt(numpy.sum( numpy.square(x.as_array()), self.direction, 
        #                          keepdims=True ))
        xx = self.__call__(x)
        xx = numpy.maximum(0, 1 - tau*self.gamma / xx)

        p  = x * xx
        
        if issubclass(type(x), DataContainer):
            return type(x)(p,geometry=x.geometry)
        elif issubclass(type(x) , numpy.ndarray):
            return p
    def proximal(self, x, tau, out=None):
        if out is None:
            return self.prox(x,tau)
        else:
            if isSizeCorrect(out, x):
                # check dimensionality
                if issubclass(type(out), DataContainer):
                    #numpy.square(x.as_array(), out = out.as_array())
                    
                    #xx = numpy.sqrt(numpy.sum( out.as_array() , self.direction, 
                    #              keepdims=True ))
                    #xx = numpy.maximum(0, 1 - tau*self.gamma / xx)
                    #x.multiply(xx, out= out.as_array())
                    x.power(2, out=out)
                    xx = numpy.sqrt(
                      out.sum(axis=self.direction, keepdims=False)
                    ) 
                    xx = numpy.maximum(xx, tau*self.gamma)
                    xx = 1 - tau*self.gamma/xx
                    x.multiply( xx , out = out)
                    return out    
                elif issubclass(type(out) , numpy.ndarray):
                    numpy.square(x.as_array(), out=out)
                    xx = numpy.sqrt(numpy.sum(out, self.direction, keepdims=True))
                    
                    xx = numpy.maximum(0, 1 - tau*self.gamma / xx)
                    x.multiply(xx, out= out)
                    return out
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
    
    def __init__(self,A,b,c=1.0,memopt=False):
        self.A = A  # Should be an operator, default identity
        self.b = b  # Default zero DataSet?
        self.c = c  # Default 1.
        self.memopt = memopt
        if memopt:
            #self.direct_placehold = A.adjoint(b)
            self.direct_placehold = A.allocate_direct()
            self.adjoint_placehold = A.allocate_adjoint()
            
        
        # Compute the Lipschitz parameter from the operator.
        # Initialise to None instead and only call when needed.
        self.L = 2.0*self.c*(self.A.get_max_sing_val()**2)
        super(Norm2sq, self).__init__()
    
    def grad(self,x):
        #return 2*self.c*self.A.adjoint( self.A.direct(x) - self.b )
        return (2.0*self.c)*self.A.adjoint( self.A.direct(x) - self.b )
    
    def __call__(self,x):
        #return self.c* np.sum(np.square((self.A.direct(x) - self.b).ravel()))
        #if out is None:
        #    return self.c*( ( (self.A.direct(x)-self.b)**2).sum() )
        #else:
        y = self.A.direct(x)
        y.__isub__(self.b)
        y.__imul__(y)
        return y.sum() * self.c
    
    def gradient(self, x, out = None):
        if self.memopt:
            #return 2.0*self.c*self.A.adjoint( self.A.direct(x) - self.b )
            
            self.A.direct(x, out=self.adjoint_placehold)
            self.adjoint_placehold.__isub__( self.b )
            self.A.adjoint(self.adjoint_placehold, out=self.direct_placehold)
            self.direct_placehold.__imul__(2.0 * self.c)
            # can this be avoided?
            out.fill(self.direct_placehold)
        else:
            return self.grad(x)
            


class ZeroFun(Function):
    
    def __init__(self,gamma=0,L=1):
        self.gamma = gamma
        self.L = L
        super(ZeroFun, self).__init__()
    
    def __call__(self,x):
        return 0
    
    def prox(self,x,tau):
        return x.copy()
    
    def proximal(self, x, tau, out=None):
        if out is None:
            return self.prox(x, tau)
        else:
            if isSizeCorrect(out, x):
                # check dimensionality  
                if issubclass(type(out), DataContainer):    
                    out.fill(x) 
                            
                elif issubclass(type(out) , numpy.ndarray): 
                    out[:] = x  
            else:   
                raise ValueError ('Wrong size: x{0} out{1}'
                                    .format(x.shape,out.shape) )

# A more interesting example, least squares plus 1-norm minimization.
# Define class to represent 1-norm including prox function
class Norm1(Function):
    
    def __init__(self,gamma):
        self.gamma = gamma
        self.L = 1
        self.sign_x = None
        super(Norm1, self).__init__()
    
    def __call__(self,x,out=None):
        if out is None:
            return self.gamma*(x.abs().sum())
        else:
            if not x.shape == out.shape:
                raise ValueError('Norm1 Incompatible size:',
                                 x.shape, out.shape)
            x.abs(out=out)
            return out.sum() * self.gamma
    
    def prox(self,x,tau):
        return (x.abs() - tau*self.gamma).maximum(0) * x.sign()
    
    def proximal(self, x, tau, out=None):
        if out is None:
            return self.prox(x, tau)
        else:
            if isSizeCorrect(x,out):
                # check dimensionality
                if issubclass(type(out), DataContainer):
                    v = (x.abs() - tau*self.gamma).maximum(0)
                    x.sign(out=out)
                    out *= v
                    #out.fill(self.prox(x,tau))    
                elif issubclass(type(out) , numpy.ndarray):
                    v = (x.abs() - tau*self.gamma).maximum(0)
                    out[:] = x.sign()
                    out *= v
                    #out[:] = self.prox(x,tau)
            else:
                raise ValueError ('Wrong size: x{0} out{1}'.format(x.shape,out.shape) )

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
    
    def proximal(self, x, tau, out=None):
        if out is None:
            return self.prox(x, tau)
        else:
            if not x.shape == out.shape:
                raise ValueError('Norm1 Incompatible size:',
                                 x.shape, out.shape)
            #(x.abs() - tau*self.gamma).maximum(0) * x.sign()
            x.abs(out = out)
            out.__isub__(tau*self.gamma)
            out.maximum(0, out=out)
            if self.sign_x is None or not x.shape == self.sign_x.shape:
                self.sign_x = x.sign()
            else:
                x.sign(out=self.sign_x)
                
            out.__imul__( self.sign_x )
