# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 13:10:56 2019

@author: evangelos
"""

import numpy as np
from ccpi.optimisation.funcs import Function
from ccpi.framework import DataContainer, ImageData, ImageGeometry

import sys
sys.path.insert(0,'/Users/evangelos/Desktop/Projects/CCPi/edo_CompOpBranch/CCPi-Framework/Wrappers/Python/ccpi/optimisation/operators/')
from CompositeOperator import CompositeDataContainer


#%%  
<<<<<<< HEAD

=======
    
>>>>>>> 50e992f2a293b7f49fc995400f02a116439c00d2
class L1Norm(Function):

    def __init__(self, A, alpha, **kwargs):
        self.A = A    
        self.alpha = alpha
        
        self.b = kwargs.get('b',None)
        self.memopt = kwargs.get('memopt',False)
        self.sym_grad = kwargs.get('sym_grad',False)

        super(L1Norm, self).__init__() 
    
    def __call__(self,x):
        y = self.A.direct(x)
        tmp = np.sqrt(np.sum([y.as_array()[i]**2 for i in range(len(y.shape))]))
        eucl_norm = ImageData(tmp)        
        eucl_norm.__isub__(self.b)
        return y.abs().sum() * self.alpha 
                   
    def proximal(self, x, tau, out=None):      
        
        res = x.divide(x.abs()) * (x.abs() - tau).maximum(0)
        if self.b is not None:
            return res + self.b
        else:
            return res
            
    def gradient():
        pass
                
    def proximal_conj(self, x, tau, out = None): 
        
        if self.b==None:  
              
            if self.sym_grad:
                tmp2 = np.sqrt(x.as_array()[0]**2 +  x.as_array()[1]**2 +  2*x.as_array()[2]**2)/self.alpha
                res = x.divide(ImageData(tmp2).maximum(1.0))                                
            else:
                res = x.divide((ImageData(x.power(2).sum(axis=0)).sqrt()/self.alpha).maximum(1.0))  
                                                   
        else:            
            res =  (x - tau*self.b)/ ((x - tau*self.b)).abs().maximum(1.0)

        if self.memopt:    
            out.fill(type(x)(res, geometry = x.geometry))  
        else:
            return type(x)(res, geometry = x.geometry)  
                                                                
        
class L1NormOld(Function):

    def __init__(self,A,b=0,alpha=1.0,memopt = False):
        self.A = A
        self.b = b
        self.alpha = alpha
        self.memopt = memopt
        super(L1NormOld, self).__init__() 
        
    def __call__(self,x):
        y = self.A.direct(x)
        y.__isub__(self.b)
        return y.abs().sum() * self.alpha        
        
    def proximal(self, x, tau, out = None):
        
        res = x.divide((ImageData(x.power(2).sum(axis=0)).sqrt()/self.alpha).maximum(1.0))
                
        if self.memopt:    
            out.fill(type(x)(res, geometry = x.geometry))  
        else:
            return type(x)(res, geometry = x.geometry)
            
    # it is the same as proximal
    def proximal_conj(self, x, tau, out = None):  
                
        res = x.divide((ImageData(x.power(2).sum(axis=0)).sqrt()/self.alpha).maximum(1.0))
        
        if self.memopt:    
            out.fill(type(x)(res, geometry = x.geometry))  
        else:
            return type(x)(res, geometry = x.geometry)   
        
        
class L2NormSq(Function):
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
#        self.L = 2.0*self.c*(self.A.get_max_sing_val()**2)
        super(L2NormSq, self).__init__()
    
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
    
    def convex_conj(self, x):        
        return (self.b * self.A.direct(x)).sum() + 0.5 * x.power(2).sum()
            
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
                    
    def proximal(self, x, tau, out = None):
        
        if self.memopt:
           self.b.multiply(tau, out=self.direct_placehold) 
           self.direct_placehold += x
           self.direct_placehold *= (1/(1+tau))
           out.fill(self.direct_placehold)
        else:
            res = ( x + tau * self.b )/(1 + tau)
            return type(x)(res.as_array(),geometry=x.geometry)
        
    def proximal_conj(self, x, tau, out = None):
        
        res = ( x - tau * self.b )/(1 + tau)
        if self.memopt:            
            out.fill(res)
        else:
            return type(x)(res.as_array(),geometry=x.geometry)   
        
        
class ZeroFun(Function):
    
    def __init__(self):
        super(ZeroFun, self).__init__()
    
    def __call__(self,x):
        return 0
    
    def proximal(self,x,tau):
        return x.copy()
        
    def proximal_conjugate(self, x, tau, out=None):
        if out is None:
            return self.prox(x, tau)
        else:
            out.fill(x)
    
        
class CompositeFunction(Function):
    
    def __init__(self, *args):
        self.functions = args
        self.length = len(self.functions)
        
    def proximal_conj(self, x, tau, out = None):
        
        out = [None]*self.length
        for i in range(self.length):
            out[i] = self.functions[i].proximal_conj(x.get_item(i), tau)
        return CompositeDataContainer(*out)    
            
        
        
        
        
        

        
