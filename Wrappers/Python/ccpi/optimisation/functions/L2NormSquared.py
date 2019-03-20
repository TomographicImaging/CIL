# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 13:10:56 2019

@author: evangelos
"""

import numpy
from ccpi.optimisation.functions import Function
from ccpi.framework import DataContainer, ImageData, ImageGeometry 
from ccpi.optimisation.functions import ScaledFunction

#%%
    
class SimpleL2NormSq(Function):
    
    def __init__(self, alpha=1):
        self.alpha = alpha
        super(SimpleL2NormSq, self).__init__()         
        # Lispchitz constant of gradient
        self.L = 2
        
    def __call__(self, x):
        return self.alpha * x.power(2).sum()
    
    def gradient(self,x, out=None):
        if out is None:
            return 2 * x
        else:
            out.fill(2*x)
    
    def convex_conjugate(self,x):
        return (1/4) * x.squared_norm()
        
    def proximal(self, x, tau, out=None):
        if out is None:
            return x.divide(1+2*tau)
        else:
            x.divide(1+2*tau, out=out)
    
    def proximal_conjugate(self, x, tau, out=None):
        if out is None:
            return x.divide(1 + tau/2)    
        else:
            x.divide(1+tau/2, out=out)



############################   L2NORM FUNCTIONS   #############################
class L2NormSq(Function):
    
    def __init__(self, **kwargs):
        super(L2NormSq, self).__init__()
        self.b = kwargs.get('b',None)              

    def __call__(self, x, out=None):
        
        ''' Evaluates L2NormSq at point x'''
        
        if self.b is not None: 
#            x.subtract(self.b, out = x)
            x = x - self.b
        
        if out is None:
            return x.squared_norm()
        else:
            out = x.squared_norm()
            return out
            
        
    def gradient(self, x, out=None):
        
        ''' Evaluates gradient of L2NormSq at point x'''
        
        if self.b is not None:
#            x.subtract(self.b, out=x)
            x = x - self.b
        if out is None:
            return 2*x
        else:
            return out.fill(2*x) 
                                                       
    def convex_conjugate(self, x, out=None):
        
        ''' Evaluate convex conjugate of L2NormSq '''
            
        tmp = 0
        if self.b is not None:
            tmp = (self.b * x).sum()
            
        if out is None:
            return (1/4) * x.squared_norm() + tmp
        else:
            out.fill((1/4) * x.squared_norm() + tmp)
                    

    def proximal(self, x, tau, out = None):

        ''' The proximal operator ( prox_\{tau * f\}(x) ) evaluates i.e., 
                argmin_x { 0.5||x - u||^{2} + tau f(x) }
        '''        
        # TODO Can we do it recursively?
        
        if out is None:
            if self.b is not None:
                return (x - self.b)/(1+2*tau) + self.b
            else:
                return x/(1+2*tau)
        else:
            if self.b is not None:
                out.fill((x - self.b)/(1+2*tau) + self.b)
            else:
                out.fill(x/(1+2*tau))                

    
    def proximal_conjugate(self, x, tau, out=None):
        
        
        return x - tau * self.proximal(x/tau, 1/tau)

    
#        if self.b is None:
#            return SimpleL2NormSq.proximal_conjugate(self, x, tau)
#        else:
#            return SimpleL2NormSq.proximal_conjugate(self, x - tau * self.b, tau)
        
    def __rmul__(self, scalar):
        return ScaledFunction(self, scalar)        


if __name__ == '__main__':
    
    
    M, N = 200,300
    ig = ImageGeometry(voxel_num_x=M, voxel_num_y = N)
    u = ig.allocate('random')
    b = ig.allocate('random') 
    
    # check grad/call no data
    f = L2NormSq()
    a1 = f.gradient(u)
    a2 = 2 * u
    numpy.testing.assert_array_almost_equal(a1.as_array(), a2.as_array(), decimal=4)
    numpy.testing.assert_equal(f(u), u.squared_norm())

    # check grad/call with data
    f1 = L2NormSq(b=b)
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
    
    
    
    
#
    
    
    