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

from ccpi.optimisation.functions import Function

class L2NormSquared(Function):
    
    r""" L2NormSquared function: :math:`F(x) = \| x\|^{2}_{2} = \underset{i}{\sum}x_{i}^{2}`
          
    Following cases are considered:
                
        a) :math:`F(x) = \|x\|^{2}_{2}`
        b) :math:`F(x) = \|x - b\|^{2}_{2}`
        
    .. note::  For case b) case we can use :code:`F = L2NormSquared().centered_at(b)`,
               see *TranslateFunction*.
        
    :Example:
        
        >>> F = L2NormSquared()
        >>> F = L2NormSquared(b=b) 
        >>> F = L2NormSquared().centered_at(b)
                                                          
    """    
    
    def __init__(self, **kwargs):
        '''creator

        Cases considered (with/without data):            
                a) .. math:: f(x) = \|x\|^{2}_{2} 
                b) .. math:: f(x) = \|\|x - b\|\|^{2}_{2}

        :param b:  translation of the function
        :type b: :code:`DataContainer`, optional
        '''                        
        super(L2NormSquared, self).__init__(L = 2)
        self.b = kwargs.get('b',None) 
        
        #if self.b is not None:
        #    self.domain = self.b.geometry  
                            
    def __call__(self, x):

        r"""Returns the value of the L2NormSquared function at x.
        
        Following cases are considered:
            
            a) :math:`F(x) = \|x\|^{2}_{2}`
            b) :math:`F(x) = \|x - b\|^{2}_{2}`
    
        :param: :math:`x`
        :returns: :math:`\underset{i}{\sum}x_{i}^{2}`
                
        """          
            
        y = x
        if self.b is not None: 
            y = x - self.b
        try:
            return y.squared_norm()
        except AttributeError as ae:
            # added for compatibility with SIRF 
            return (y.norm()**2)
                
    def gradient(self, x, out=None):        
        
        r"""Returns the value of the gradient of the L2NormSquared function at x.
        
        Following cases are considered:
                
            a) :math:`F'(x) = 2x`
            b) :math:`F'(x) = 2(x-b)`
                
        """
                
        if out is not None:
            
            out.fill(x)
            if self.b is not None:
                out -= self.b
            out *= 2
            
        else:
            
            y = x
            if self.b is not None:
                y = x - self.b
            return 2*y
        
                                                       
    def convex_conjugate(self, x):
        
        r"""Returns the value of the convex conjugate of the L2NormSquared function at x.
        
        Consider the following cases:
                
                a) .. math:: F^{*}(x^{*}) = \frac{1}{4}\|x^{*}\|^{2}_{2} 
                b) .. math:: F^{*}(x^{*}) = \frac{1}{4}\|x^{*}\|^{2}_{2} + <x^{*}, b>
                
        """                
        tmp = 0
        
        if self.b is not None:
            tmp = x.dot(self.b) 
            
        return (1./4.) * x.squared_norm() + tmp


    def proximal(self, x, tau, out = None):
        
        r"""Returns the value of the proximal operator of the L2NormSquared function at x.
        
        
        Consider the following cases:
                
                a) .. math:: \mathrm{prox}_{\tau F}(x) = \frac{x}{1+2\tau}
                b) .. math:: \mathrm{prox}_{\tau F}(x) = \frac{x-b}{1+2\tau} + b      
                        
        """            

        if out is None:
            
            if self.b is None:
                return x/(1+2*tau)
            else:
                tmp = x.subtract(self.b)
                tmp /= (1+2*tau)
                tmp += self.b
                return tmp

        else:
            if self.b is not None:
                x.subtract(self.b, out=out)
                out /= (1+2*tau)
                out += self.b
            else:
                x.divide((1+2*tau), out=out)

    
#    def proximal_conjugate(self, x, tau, out=None):
#        
#        r'''Proximal operator of the convex conjugate of L2NormSquared at x:
#           
#           .. math::  prox_{\tau * f^{*}}(x)'''
#        
#        if out is None:
#            if self.b is not None:
#                return (x - tau*self.b)/(1 + tau/2) 
#            else:
#                return x/(1 + tau/2)
#        else:
#            if self.b is not None:
#                x.subtract(tau*self.b, out=out)
#                out.divide(1+tau/2, out=out)
#            else:
#                x.divide(1 + tau/2, out=out)                                        

if __name__ == '__main__':
    
    from ccpi.framework import ImageGeometry
    import numpy
    # TESTS for L2 and scalar * L2
    
    M, N, K = 2,3,1
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
    
    igggg = ImageGeometry(4,4)
    f1 = L2NormSquared(b=b)
    b1 = f1.gradient(u)
    b2 = 2 * (u-b)
        
    numpy.testing.assert_array_almost_equal(b1.as_array(), b2.as_array(), decimal=4)
    numpy.testing.assert_equal(f1(u), ((u-b)).squared_norm())
    
    #check convex conjuagate no data
    c1 = f.convex_conjugate(u)
    c2 = 1/4 * u.squared_norm()
    numpy.testing.assert_equal(c1, c2)
    
    #check convex conjuagate with data
    d1 = f1.convex_conjugate(u)
    d2 = (1/4) * u.squared_norm() + u.dot(b)
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
    
    
    
    print( " ####### check without out ######### " )
          
          
    u_out_no_out = ig.allocate('random_int')         
    res_no_out = f_scaled_data.proximal_conjugate(u_out_no_out, 0.5)          
    print(res_no_out.as_array())
    
    print( " ####### check with out ######### " ) 
          
    res_out = ig.allocate()        
    f_scaled_data.proximal_conjugate(u_out_no_out, 0.5, out = res_out)
    
    print(res_out.as_array())   

    numpy.testing.assert_array_almost_equal(res_no_out.as_array(), \
                                            res_out.as_array(), decimal=4)  
    
    
    
    ig1 = ImageGeometry(2,3)
    
    tau = 0.1
    
    u = ig1.allocate('random_int')
    b = ig1.allocate('random_int')
    
    scalar = 0.5
    f_scaled = scalar * L2NormSquared(b=b)
    f_noscaled = L2NormSquared(b=b)
    
    
    res1 = f_scaled.proximal(u, tau)
    res2 = f_noscaled.proximal(u, tau*scalar)
    
#    res2 = (u + tau*b)/(1+tau)
    
    numpy.testing.assert_array_almost_equal(res1.as_array(), \
                                            res2.as_array(), decimal=4)
                                            
