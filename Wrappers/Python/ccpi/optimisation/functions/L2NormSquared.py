# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library developed by
#   Visual Analytics and Imaging System Group of the Science Technology
#   Facilities Council, STFC

#   Copyright 2018-2019 Evangelos Papoutsellis and Edoardo Pasca

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy
from ccpi.optimisation.functions import Function
from ccpi.optimisation.functions.ScaledFunction import ScaledFunction
from ccpi.framework import DataContainer, ImageData, ImageGeometry  

############################   L2NORM FUNCTION   #############################
class L2NormSquared(Function):
    
    def __init__(self, **kwargs):
        
        ''' L2NormSquared class
            f : ImageGeometry --> R
            
            Cases: f(x) = ||x||^{2}_{2}
                   f(x) = || x - b ||^{2}_{2}     
        
        '''
        
        #TODO need x, b to live in the same geometry if b is not None
                        
        super(L2NormSquared, self).__init__()
        self.b = kwargs.get('b',None)  

    def __call__(self, x):
        ''' Evaluates L2NormSq at point x'''
        
        y = x
        if self.b is not None: 
#            x.subtract(self.b, out = x)
            y = x - self.b
#        else:
#            y
#        if out is None:
#            return x.squared_norm()
#        else:
        try:
            return y.squared_norm()
        except AttributeError as ae:
            # added for compatibility with SIRF 
            return (y.norm()**2)
        
            
        
    def gradient(self, x, out=None):
        ''' Evaluates gradient of L2NormSq at point x'''
        if out is not None:
            out.fill(x)
            if self.b is not None:
                out -= self.b
            out *= 2
        else:
            y = x
        if self.b is not None:
#            x.subtract(self.b, out=x)
            y = x - self.b
        return 2*y
        
                                                       
    def convex_conjugate(self, x, out=None):
        ''' Evaluate convex conjugate of L2NormSq'''
            
        tmp = 0
        if self.b is not None:
#            tmp = (self.b * x).sum()
            tmp = (x * self.b).sum()
            
        if out is None:
            # FIXME: this is a number
            return (1./4.) * x.squared_norm() + tmp
        else:
            # FIXME: this is a DataContainer
            out.fill((1./4.) * x.squared_norm() + tmp)
                    

    def proximal(self, x, tau, out = None):

        ''' The proximal operator ( prox_\{tau * f\}(x) ) evaluates i.e., 
                argmin_x { 0.5||x - u||^{2} + tau f(x) }
        '''        
        
        if out is None:
            if self.b is not None:
                return (x - self.b)/(1+2*tau) + self.b
            else:
                return x/(1+2*tau)
        else:
            out.fill(x)
            if self.b is not None:
                out -= self.b
            out /= (1+2*tau)
            if self.b is not None:
                out += self.b
                #out.fill((x - self.b)/(1+2*tau) + self.b)
            #else:
            #    out.fill(x/(1+2*tau))                

    
    def proximal_conjugate(self, x, tau, out=None):
        
        if out is None:
            if self.b is not None:
                # change the order cannot add ImageData + NestedBlock
                return (x - tau*self.b)/(1 + tau/2) 
            else:
                return x/(1 + tau/2 )
        else:
            if self.b is not None:
                out.fill((x - tau*self.b)/(1 + tau/2))
            else:
                out.fill(x/(1 + tau/2 ))
                                        
    def __rmul__(self, scalar):
        return ScaledFunction(self, scalar)        


if __name__ == '__main__':
    
    
    # TESTS for L2 and scalar * L2
    
    M, N, K = 2,3,5
    ig = ImageGeometry(voxel_num_x=M, voxel_num_y = N, voxel_num_z = K)
    u = ig.allocate('random_int')
    b = ig.allocate('random_int') 
    
    # check grad/call no data
    f = L2NormSquared()
    a1 = f.gradient(u)
    a2 = 2 * u
    numpy.testing.assert_array_almost_equal(a1.as_array(), a2.as_array(), decimal=4)
    numpy.testing.assert_equal(f(u), u.squared_norm())

    # check grad/call with data
    f1 = L2NormSquared(b=b)
    b1 = f1.gradient(u)
    b2 = 2 * (u-b)
        
    numpy.testing.assert_array_almost_equal(b1.as_array(), b2.as_array(), decimal=4)
    numpy.testing.assert_equal(f1(u), (u-b).squared_norm())
    
    #check convex conjuagate no data
    c1 = f.convex_conjugate(u)
    c2 = 1/4 * u.squared_norm()
    numpy.testing.assert_equal(c1, c2)
    
    #check convex conjuagate with data
    d1 = f1.convex_conjugate(u)
    d2 = (1/4) * u.squared_norm() + (u*b).sum()
    numpy.testing.assert_equal(d1, d2)  
    
    # check proximal no data
    tau = 5
    e1 = f.proximal(u, tau)
    e2 = u/(1+2*tau)
    numpy.testing.assert_array_almost_equal(e1.as_array(), e2.as_array(), decimal=4)
    
    # check proximal with data
    tau = 5
    h1 = f1.proximal(u, tau)
    h2 = (u-b)/(1+2*tau) + b
    numpy.testing.assert_array_almost_equal(h1.as_array(), h2.as_array(), decimal=4)    
    
    # check proximal conjugate no data
    tau = 0.2
    k1 = f.proximal_conjugate(u, tau)
    k2 = u/(1 + tau/2 )
    numpy.testing.assert_array_almost_equal(k1.as_array(), k2.as_array(), decimal=4) 
    
    # check proximal conjugate with data
    l1 = f1.proximal_conjugate(u, tau)
    l2 = (u - tau * b)/(1 + tau/2 )
    numpy.testing.assert_array_almost_equal(l1.as_array(), l2.as_array(), decimal=4)     
    
        
    # check scaled function properties
    
    # scalar 
    scalar = 100
    f_scaled_no_data = scalar * L2NormSquared()
    f_scaled_data = scalar * L2NormSquared(b=b)
    
    # call
    numpy.testing.assert_equal(f_scaled_no_data(u), scalar*f(u))
    numpy.testing.assert_equal(f_scaled_data(u), scalar*f1(u))
    
    # grad
    numpy.testing.assert_array_almost_equal(f_scaled_no_data.gradient(u).as_array(), scalar*f.gradient(u).as_array(), decimal=4)
    numpy.testing.assert_array_almost_equal(f_scaled_data.gradient(u).as_array(), scalar*f1.gradient(u).as_array(), decimal=4)
    
    # conj
    numpy.testing.assert_almost_equal(f_scaled_no_data.convex_conjugate(u), \
                               f.convex_conjugate(u/scalar) * scalar, decimal=4)
    
    numpy.testing.assert_almost_equal(f_scaled_data.convex_conjugate(u), \
                               scalar * f1.convex_conjugate(u/scalar), decimal=4)
    
    # proximal
    numpy.testing.assert_array_almost_equal(f_scaled_no_data.proximal(u, tau).as_array(), \
                                            f.proximal(u, tau*scalar).as_array())
    
    
    numpy.testing.assert_array_almost_equal(f_scaled_data.proximal(u, tau).as_array(), \
                                            f1.proximal(u, tau*scalar).as_array())
                               
    
    # proximal conjugate
    numpy.testing.assert_array_almost_equal(f_scaled_no_data.proximal_conjugate(u, tau).as_array(), \
                                            (u/(1 + tau/(2*scalar) )).as_array(), decimal=4)
    
    numpy.testing.assert_array_almost_equal(f_scaled_data.proximal_conjugate(u, tau).as_array(), \
                                            ((u - tau * b)/(1 + tau/(2*scalar) )).as_array(), decimal=4)    
    
    
    
