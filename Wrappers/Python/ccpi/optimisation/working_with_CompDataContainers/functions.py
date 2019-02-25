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
from operators import CompositeDataContainer
from numbers import Number

#%%  

class mixed_L12Norm(Function):
    
    def __init__(self, A, alpha, **kwargs):
        self.A = A    
        self.alpha = alpha
        
        self.b = kwargs.get('b',None)
        self.memopt = kwargs.get('memopt',False)
        self.sym_grad = kwargs.get('sym_grad',False)

        super(mixed_L12Norm, self).__init__() 
        
    def __call__(self,x):
        
        y = self.A.direct(x)     
        eucl_norm = ImageData(y.power(2).sum(axis=0)).sqrt()       
        eucl_norm.__isub__(self.b)
        return eucl_norm.sum() * self.alpha 
    
    def proximal_conjugate(self, x, tau, out = None): 
        
        if self.b==None:  
              
            if self.sym_grad:
                tmp2 = np.sqrt(x.as_array()[0]**2 +  x.as_array()[1]**2 +  2*x.as_array()[2]**2)/self.alpha
                res = x.divide(ImageData(tmp2).maximum(1.0))                                
            else:
                res = x.divide((ImageData(x.power(2).sum(axis=0)).sqrt()/self.alpha).maximum(1.0))  
                                                   
        else:            
            res =  self.b + (x - tau*self.b)/ ((x - tau*self.b)).abs().maximum(1.0)

        if self.memopt:    
            out.fill(type(x)(res, geometry = x.geometry))  
        else:
            return type(x)(res, geometry = x.geometry)    
    
    
class L1Norm(Function):
    
    '''     
        f(x) = alpha * ||Ax-b||_{1} = sum |Ax-b|, |.| is absolute value        
    '''

    def __init__(self, A, alpha=1, **kwargs):
        
        self.A = A    
        self.alpha = alpha
        
        self.b = kwargs.get('b',None)
        self.memopt = kwargs.get('memopt',False)

        super(L1Norm, self).__init__() 
    
    def __call__(self,x):
        
        '''
          Returns f(x) = alpha * ||Ax-b||_{1}
        '''
        
        y = self.A.direct(x)      
        y.__isub__(self.b)
        y.__imul__(y)
        
        return y.abs().sum() * self.alpha 
    
    def gradient(self):        
        pass
                   
    def proximal(self, x, tau, out=None): 
        
        '''
         z = Ax
         proximal_tau(u) = argmin_{z}{ 0.5 * ||z - u||_{2}^{2} + tau * f(z) }
        
        '''
        
        SoftThresholdOperator = x.divide(x.abs()) * (x.abs() - tau).maximum(0)        
        return self.b + SoftThresholdOperator
        
                            
    def proximal_conjugate(self, x, tau, out = None): 
        
        ''' 
         z = Ax
         proximal_tau(u) = argmin_{z}{ 0.5 * ||z - u||_{2}^{2} + tau * f^{*}(z)}
        
        '''
        return (x + tau*self.b).divide(((x + tau*self.b)).abs().maximum(1.0))
        
#        if self.b==None:  
#            
#            res = x.divide((ImageData(x.power(2).sum(axis=0)).sqrt()/self.alpha).maximum(1.0))  
#                                                   
#        else:            
#            res =  (x - tau*self.b)/ ((x - tau*self.b)).abs().maximum(1.0)
#            
#        res =     
#
#        if self.memopt:    
#            out.fill(type(x)(res, geometry = x.geometry))  
#        else:
#            return type(x)(res, geometry = x.geometry)  
                                                                
        
class L2NormSq(Function):

    
    def __init__(self, A, alpha=1, **kwargs):
        
        self.A = A    
        self.alpha = alpha
        
        self.b = kwargs.get('b',None)
        self.memopt = kwargs.get('memopt',False)

        super(L2NormSq, self).__init__() 
        
    
    def grad(self,x):
        #return 2*self.c*self.A.adjoint( self.A.direct(x) - self.b )
        return (2.0*self.alpha)*self.A.adjoint( self.A.direct(x) - self.b )
    
    def __call__(self,x):
        #return self.c* np.sum(np.square((self.A.direct(x) - self.b).ravel()))
        #if out is None:
        #    return self.c*( ( (self.A.direct(x)-self.b)**2).sum() )
        #else:
        y = self.A.direct(x)
        y.__isub__(self.b)
        y.__imul__(y)
        return y.sum() * self.alpha
    
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
        
    def proximal_conjugate(self, x, tau, out = None):
        
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
                
    def __call__(self,x):

        t = 0
        for i in range(self.length):
            t +=self.functions[i](x.get_item(0))
        return t            
                            
    def proximal_conjugate(self, x, tau, out = None):
        
        if isinstance(tau, Number):
            tau = CompositeDataContainer(tau)
        out = [None]*self.length
        for i in range(self.length):
            out[i] = self.functions[i].proximal_conjugate(x.get_item(i), tau.get_item(0))
        return CompositeDataContainer(*out)    
            
        
        
        
        
        

        
