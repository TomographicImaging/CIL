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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import warnings

from numbers import Number
from ccpi.optimisation.operators import ZeroOperator, Identity

class Function(object):
    
    ''' Abstract class representing a function
    
            :param L: Lipschitz constant of the gradient of the function, when it is differentiable.
            :param domain: The domain of the function.
            
    '''
    def __init__(self, domain = None, L = None):
        
        self.L = L
        self.domain = domain

    def __call__(self,x):
        
        '''Returns the value of the function at x : .. math:: $f(x)$ '''
        raise NotImplementedError

    def gradient(self, x, out=None):
        
        ''' Returns the value of the gradient of the function at x : .. math:: $f'(x)$ '''
        raise NotImplementedError

    def proximal(self, x, tau, out=None):
        r''' Returns the value of the proximal operator of \tau * f at x:  \mathrm{prox}_{\tau f}(x)'''
        raise NotImplementedError

    def convex_conjugate(self, x):
        '''This evaluates the convex conjugate of the function at x'''
        raise NotImplementedError

    def proximal_conjugate(self, x, tau, out = None):
        
        '''This returns the proximal operator for the convex conjugate of the function at x, tau
        
            Due to Moreau Identity, we have an analytic formula (that depends on the proximal) 
            for this and there is no need to compute using the convex_conjugate
        
        '''
        if out is None:
            return x - tau * self.proximal(x/tau, 1/tau)
        else:            
            self.proximal(x/tau, 1/tau, out = out)
            out*=-tau
            out.add(x, out = out) 

    def grad(self, x):
        '''Alias of gradient(x,None)'''
        warnings.warn('''This method will disappear in following 
        versions of the CIL. Use gradient instead''', DeprecationWarning)
        return self.gradient(x, out=None)

    def prox(self, x, tau):
        '''Alias of proximal(x, tau, None)'''
        warnings.warn('''This method will disappear in following 
        versions of the CIL. Use proximal instead''', DeprecationWarning)
        return self.proximal(x, tau, out=None)
    
    def domain(self):
        ''' This returns the domain of the function. '''
        return self.domain
    
    # Algebra for Function Class
    
        # Add functions
        # Subtract functions
        # Add/Substract with Scalar
        # Multiply with Scalar
    
    def __add__(self, other):
        
        ''' This returns a sum of functions'''
        
        if isinstance(other, Function):
            return SumFunction(self, other)
        elif isinstance(other, (SumFunctionScalar, ConstantFunction, Number)):
            return SumFunctionScalar(self, other)
        else:
            raise ValueError('Not implemented')   
            
    def __radd__(self, other):
        
        return self + other 
                          
    def __sub__(self, other):
        ''' This returns a subtract of functions '''
        return self + (-1) * other    

    def __rmul__(self, scalar):
        '''Defines the multiplication by a scalar on the left returns a ScaledFunction'''                
        return ScaledFunction(self, scalar)
    
    def __mul__(self, scalar):
        '''Defines the multiplication by a scalar on the left returns a ScaledFunction'''                
        return scalar * ScaledFunction(self, 1)   
    
    def centered_at(self, center):
        '''This returns the proximal operator for the convex conjugate of the function at x, tau'''
        return TranslateFunction(self, center)  
    
class SumFunction(Function):
    
    '''SumFunction

    A class to represent the sum of two Functions
   
       
    '''    
    def __init__(self, function1, function2 ):
                
        super(SumFunction, self).__init__()        

        #if function1.domain != function2.domain:            
        #    raise ValueError('{} is not the same as {}'.format(function1.domain, function2.domain)) 
            
        #self.domain = function1.domain
                                
        if function1.L is not None and function2.L is not None:
            self.L = function1.L + function2.L
            
        self.function1 = function1
        self.function2 = function2               
            
    def __call__(self,x):
        '''Evaluates the function at x '''
        return self.function1(x) + self.function2(x)
    
    def gradient(self, x, out=None):
        
        '''Returns the gradient of the sum of functions at x, if both of them are differentiable'''
        
#        try: 
        if out is None:            
            return self.function1.gradient(x) +  self.function2.gradient(x)  
        else:
            out_tmp = out.copy()
            out_tmp *=0
            self.function1.gradient(x, out=out)
            self.function2.gradient(x, out=out_tmp)
            out_tmp.add(out, out=out)
#            out.add(self.function2.gradient(x, out=out), out=out)
#        except NotImplementedError:
#            print("Either {} or {} is not differentiable".format(type(self.function1).__name__), type(self.function1).__name__)) 
                            
            
        
class ScaledFunction(Function):
    
    '''ScaledFunction

    A class to represent the scalar multiplication of an Function with a scalar.
    It holds a function and a scalar. Basically it returns the multiplication
    of the product of the function __call__, convex_conjugate and gradient with the scalar.
    For the rest it behaves like the function it holds.

    Args:
       function (Function): a Function or BlockOperator
       scalar (Number): a scalar multiplier
    Example:
       The scaled operator behaves like the following:
       
    '''
    def __init__(self, function, scalar):
        
        super(ScaledFunction, self).__init__() 
                                                     
        if not isinstance (scalar, Number):
            raise TypeError('expected scalar: got {}'.format(type(scalar)))
        
        if function.L is not None:        
            self.L = abs(scalar) * function.L  
            
        self.scalar = scalar
        self.function = function       
              
    def __call__(self,x, out=None):
        '''Evaluates the function at x '''
        return self.scalar * self.function(x)

    def convex_conjugate(self, x):
        '''returns the convex_conjugate of the scaled function '''
        return self.scalar * self.function.convex_conjugate(x/self.scalar)
    
    def gradient(self, x, out=None):
        '''Returns the gradient of the function at x, if the function is differentiable'''
        
#        try:
        if out is None:            
            return self.scalar * self.function.gradient(x)
        else:
            self.function.gradient(x, out=out)
            out *= self.scalar  
#        except NotImplementedError:
#            print("{} is not differentiable".format(type(self.function).__name__))                         

    def proximal(self, x, tau, out=None):
        '''This returns the proximal operator for the function at x, tau
        '''
#        if out is None:
        return self.function.proximal(x, tau*self.scalar, out=out)     
#        else:
#            self.function.proximal(x, tau*self.scalar, out = out)

#    def proximal_conjugate(self, x, tau, out = None):
#        '''This returns the proximal operator for the function at x, tau
#        '''
#        if out is None:
#            return self.scalar * self.function.proximal_conjugate(x/self.scalar, tau/self.scalar)
#        else:
#            self.function.proximal_conjugate(x/self.scalar, tau/self.scalar, out=out)
#            out *= self.scalar

    def function(self):
        return self.function

class SumFunctionScalar(SumFunction):
    
    '''SumFunctionScalar

    A class to represent the sum a Function and a scalar
          
    
    This is child of SumFunction where the second function is a ConstantFunction
    
    Although SumFunction has no general expressions for i) convex_conjugate
                                                        ii) proximal
                                                        iii) proximal_conjugate
            
    if the second argument is a ConstantFunction then we can derive the above analytically
    
    '''    
    
    def __init__(self, function, constant):
        
        super(SumFunctionScalar, self).__init__(function, ConstantFunction(constant))        
        self.constant = constant
        self.function = function
        
    def convex_conjugate(self,x):
        
        return self.function.convex_conjugate(x) - self.constant
    
    def proximal(self, x, tau, out=None):
        
        return self.function.proximal(x, tau, out=out)        
    
#    def proximal_conjugate(self, x, tau, out = None):
#        
#        self.function.proximal_conjugate(x, tau, out = out) 
        
    def function(self):       
       return self.function    
    
                        
class ConstantFunction(Function):
    
            
    r'''ConstantFunction: .. math:: f(x) = constant, constant\in\mathbb{R}         
        
    '''
    
    def __init__(self, constant = 0):
        
        super(ConstantFunction, self).__init__(L=0)
        
        if not isinstance (constant, Number):
            raise TypeError('expected scalar: got {}'.format(type(constant)))
                
        self.constant = constant
              
    def __call__(self,x):
        
        '''Evaluates ConstantFunction at x'''
        return self.constant
        
    def gradient(self, x, out=None):
        
        '''Evaluates gradient of ConstantFunction at x: Returns a number'''        
        
        return ZeroOperator(x.geometry).direct(x, out=out)
    
    def convex_conjugate(self, x):
                        
        ''' This is the Indicator of singleton {constant}. It is either -constnat if x^* = 0
        or infinity.
        However, x^{*} = 0 only in the limit of iterations, so in fact this can be infinity.
        We do not want to have inf values in the convex conjugate, so we have to peanalise this value 
        along the iterations
        
        '''       
        
        return (x-self.constant).maximum(0).sum()
#        if x.norm()<1e-6:            
#            return -self.constant
#        else:
#            return np.inf
                
    def proximal(self, x, tau, out=None):
        
        ''' This returns the proximal of a constnat function which is the same as x'''
        
        return Identity(x.geometry).direct(x, out=out)

                       
    def proximal_conjugate(self, x, tau, out = None):
        
        return ZeroOperator(x.geometry).direct(x, out=out)


class ZeroFunction(ConstantFunction):
    
    r'''ZeroFunction: .. math:: f(x) = 0,         
        
    '''
    
    def __init__(self):
        super(ZeroFunction, self).__init__(constant = 0.) 
        
class TranslateFunction(Function):
    
    r'''TranslateFunction: Let Function f(x), here we compute f(x - center)
                
    '''
    
    def __init__(self, function, center):
        
        super(TranslateFunction, self).__init__(L = function.L) 
                        
        self.function = function
        self.center = center
        
        '''
            slope should be DataContainer
        
        '''
        
    def __call__(self, x):
        
        return self.function(x - self.center)
    
    def gradient(self, x, out = None):
        
        if out is None:
            return self.function.gradient(Identity(x.geometry).direct(x) - self.center)
        else:            
            Identity(x.geometry).direct(x, out = out)
            out.subtract(self.center, out = out)
            self.function.gradient(out, out = out)           
    
    def proximal(self, x, tau, out=None):
        
        if out is None:
            return self.function.proximal(x - self.center, tau) + self.center
        else:                    
            x.subtract(self.center, out = out)
            self.function.proximal(out, tau, out = out)
            out.add(self.center, out = out)
                    
    def convex_conjugate(self, x):
        
        return self.function.convex_conjugate(x) + self.center.dot(x)
    
#    def proximal_conjugate(self, x, tau, out=None):
#        
#        if out is None:
#            return x - tau * self.function.proximal(x/tau, 1/tau)
#        else:            
#            self.function.proximal(x/tau, 1/tau, out = out)
#            out*=-tau
#            out.add(x)       
        
    def function(self):       
       return self.function             
    

if __name__ == '__main__':
    

    from ccpi.optimisation.functions import L1Norm, ScaledFunction, \
                                            LeastSquares, L2NormSquared, \
                                            KullbackLeibler, FunctionOperatorComposition, ZeroFunction, ConstantFunction, TranslateFunction
    from ccpi.optimisation.operators import Identity                                        
    from ccpi.framework import ImageGeometry, BlockGeometry
    
    import unittest
    import numpy
    from numbers import Number

    ig = ImageGeometry(4,4)
    tmp = ig.allocate('random_int')
    b = ig.allocate('random_int')
    scalar = 0.4
               
    f = L2NormSquared().centered_at(b) * scalar
    
    res1 = f(tmp)
    
    res2 = scalar * (tmp-b).squared_norm()
    
#    f = L2NormSquared()
    print(type(f).__name__)
    
    print(f.function)
    
    
    f = L2NormSquared().centered_at(b) 
    print(f(tmp))
    
    Id = Identity(ig)
    f = FunctionOperatorComposition(L2NormSquared().centered_at(b), Id)
    print(f(tmp))
    
    
    
    
         
