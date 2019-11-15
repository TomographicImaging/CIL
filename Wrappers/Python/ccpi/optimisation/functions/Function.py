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

import numpy as np

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
        
        ''' This is the IndicatorSingleton function that returns 0 if x^* = 0 and inf otherwise 

            We can implment IndicatorSingleton but then we need it to add it also here, 
            and increase the size of this file
        
        '''        
        
        if x.norm()==0:
            return 0.
        else:
            return np.inf
        
        # not working with BlockDataContainer
#        tmp = x.as_array()
#                   
#        if not np.any(tmp):
#            return 0.
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
        super(ZeroFunction, self).__init__(constant = 0) 

class TranslateFunction(Function):
    
    r'''TranslateFunction: Let Function f(x), here we compute f(x - center)
                
    '''
    
    def __init__(self, function, center):
        
        super(TranslateFunction, self).__init__() 
                        
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
    
if __name__ == '__main__':
    
    from ccpi.framework import ImageGeometry
    import numpy
    from ccpi.optimisation.functions import L2NormSquared, L1Norm, KullbackLeibler
                
    # Test TranslationFunction

    ig = ImageGeometry(4,4)
    tmp = ig.allocate('random_int')
    b = ig.allocate('random_int')
    scalar = 0.4
    list_functions = [L2NormSquared(), L1Norm()]
    decimal = 5
    
    for i in list_functions:
        
        print('Test Translation for Function {} '.format(type(i).__name__))
        
        if isinstance(i, L2NormSquared):
            
            f = L2NormSquared(b = b)    
            g = TranslateFunction(L2NormSquared(), b)
            
        elif isinstance(i, L1Norm):
            
            f = L1Norm(b = b)    
            g = TranslateFunction(L1Norm(), b)
            
        elif isinstance(i, ScaledFunction):

            if isinstance(i.function, L2NormSquared):
                f = scalar * L2NormSquared(b = b)    
                g = scalar * TranslateFunction(L2NormSquared(), b)
                
            if isinstance(i.function, L1Norm):
                f = scalar * L1Norm(b = b)    
                g = scalar * TranslateFunction(L1Norm(), b)                
                        
        # check call
        res1 = f(tmp)
        res2 = g(tmp)    
        numpy.testing.assert_equal(res1, res2)
        
        # check gradient
            
        if not isinstance(i, L1Norm):
        
            res1 = f.gradient(tmp)
            res2 = g.gradient(tmp) 
            numpy.testing.assert_equal(res1.as_array(), res2.as_array()) 
        
            # check gradient out
            res3 = ig.allocate()
            res4 = ig.allocate()
            f.gradient(tmp, out = res3)
            g.gradient(tmp, out = res4)
            numpy.testing.assert_equal(res3.as_array(), res4.as_array())
        
        # check convex conjugate
        res1 = f.convex_conjugate(tmp)
        res2 = g.convex_conjugate(tmp)
        numpy.testing.assert_equal(res1, res2) 
        
        # check proximal    
        tau = 0.5
        res1 = f.proximal(tmp, tau)
        res2 = g.proximal(tmp, tau)
        numpy.testing.assert_equal(res1.as_array(), res2.as_array()) 
        
        # check proximal out           
        res3 = ig.allocate()
        res4 = ig.allocate()
        f.proximal(tmp, tau, out = res3)
        g.proximal(tmp, tau, out = res4)
        numpy.testing.assert_array_almost_equal(res3.as_array(), res4.as_array(),decimal = decimal)     
        
        # check proximal conjugate  
        tau = 0.4
        res1 = f.proximal_conjugate(tmp, tau)
        res2 = g.proximal_conjugate(tmp, tau)
        numpy.testing.assert_array_almost_equal(res1.as_array(), res2.as_array(),decimal = decimal)  
                            
        # check proximal out           
        res3 = ig.allocate()
        res4 = ig.allocate()
        f.proximal_conjugate(tmp, tau, out = res3)
        g.proximal_conjugate(tmp, tau, out = res4)
        numpy.testing.assert_array_almost_equal(res3.as_array(), res4.as_array(),decimal = decimal)          
        
        
        f = L2NormSquared() + 1
        print(f(tmp))
        
        
    
#    tau = 0.5    
#    f = L2NormSquared(b=b) 
#    g = TranslateFunction(f, b)
#    res1 = f.proximal_conjugate(tmp, tau)    
#    res2 = tmp - tau * f.proximal(tmp/tau, 1/tau)
#    res3 = g.proximal_conjugate(tmp, tau)
    
#    print(res1.as_array())
#    print(res3.as_array())
#    numpy.testing.assert_equal(res1.as_array(), res2.as_array()) 
#    numpy.testing.assert_equal(res1.as_array(), res3.as_array()) 
    
    
    
    
         
