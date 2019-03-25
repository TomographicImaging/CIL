## -*- coding: utf-8 -*-
#
##!/usr/bin/env python3
## -*- coding: utf-8 -*-
#"""
#Created on Thu Feb  7 13:10:56 2019
#
#@author: evangelos
#"""
#
#import numpy as np
##from ccpi.optimisation.funcs import Function
#from ccpi.optimisation.functions import Function
#from ccpi.framework import DataContainer, ImageData, ImageGeometry 
#from operators import CompositeDataContainer, Identity, CompositeOperator
#from numbers import Number
#
#
#############################   L2NORM FUNCTIONS   #############################
#class SimpleL2NormSq(Function):
#    
#    def __init__(self, alpha=1):
#        
#        super(SimpleL2NormSq, self).__init__()         
#        self.alpha = alpha
#        
#    def __call__(self, x):
#        return self.alpha * x.power(2).sum()
#    
#    def gradient(self,x):
#        return 2 * self.alpha * x
#    
#    def convex_conjugate(self,x):
#        return (1/4*self.alpha) * x.power(2).sum()
#    
#    def proximal(self, x, tau):
#        return x.divide(1+2*tau*self.alpha)
#    
#    def proximal_conjugate(self, x, tau):
#        return x.divide(1 + tau/2*self.alpha )
#    
#        
#class L2NormSq(SimpleL2NormSq):
#    
#    def __init__(self, A, b = None, alpha=1, **kwargs):
#        
#        super(L2NormSq, self).__init__(alpha=alpha)         
#        self.alpha = alpha        
#        self.A = A
#        self.b = b
#                
#    def __call__(self, x):
#        
#        if self.b is None:
#            return SimpleL2NormSq.__call__(self, self.A.direct(x))
#        else:
#            return SimpleL2NormSq.__call__(self, self.A.direct(x) - self.b)
#        
#    def convex_conjugate(self, x):
#        
#        ''' The convex conjugate corresponds to the simple functional i.e., 
#        f(x) = alpha * ||x - b||_{2}^{2}
#        '''
#        
#        if self.b is None:
#            return SimpleL2NormSq.convex_conjugate(self, x)
#        else:
#            return SimpleL2NormSq.convex_conjugate(self, x) + (self.b * x).sum()
#                            
#    def gradient(self, x):
#        
#        if self.b is None:
#            return 2*self.alpha * self.A.adjoint(self.A.direct(x)) 
#        else:
#            return 2*self.alpha * self.A.adjoint(self.A.direct(x) - self.b) 
#        
#    def proximal(self, x, tau):
#        
#        ''' The proximal operator corresponds to the simple functional i.e., 
#        f(x) = alpha * ||x - b||_{2}^{2}
#        
#        argmin_x { 0.5||x - u||^{2} + tau f(x) }
#        '''
#        
#        if self.b is None:
#            return SimpleL2NormSq.proximal(self, x, tau)
#        else:
#            return self.b + SimpleL2NormSq.proximal(self, x - self.b , tau)
#
#        
#    def proximal_conjugate(self, x, tau):
#        
#        ''' The proximal operator corresponds to the simple convex conjugate 
#        functional i.e., f^{*}(x^{)        
#        argmin_x { 0.5||x - u||^{2} + tau f(x) }
#        '''
#        if self.b is None:
#            return SimpleL2NormSq.proximal_conjugate(self, x, tau)
#        else:
#            return SimpleL2NormSq.proximal_conjugate(self, x - tau * self.b, tau)
#
#
#############################   L1NORM FUNCTIONS   #############################
#class SimpleL1Norm(Function):
#    
#    def __init__(self, alpha=1):
#        
#        super(SimpleL1Norm, self).__init__()         
#        self.alpha = alpha
#        
#    def __call__(self, x):
#        return self.alpha * x.abs().sum()
#    
#    def gradient(self,x):
#        return ValueError('Not Differentiable')
#            
#    def convex_conjugate(self,x):
#        return 0
#    
#    def proximal(self, x, tau):
#        ''' Soft Threshold'''
#        return x.sign() * (x.abs() - tau * self.alpha).maximum(1.0)
#        
#    def proximal_conjugate(self, x, tau):
#        return x.divide((x.abs()/self.alpha).maximum(1.0))
#    
#class L1Norm(SimpleL1Norm):
#    
#    def __init__(self, A, b = None, alpha=1, **kwargs):
#        
#        super(L1Norm, self).__init__()         
#        self.alpha = alpha        
#        self.A = A
#        self.b = b
#        
#    def __call__(self, x):
#        
#        if self.b is None:
#            return SimpleL1Norm.__call__(self, self.A.direct(x))
#        else:
#            return SimpleL1Norm.__call__(self, self.A.direct(x) - self.b)
#    
#    def gradient(self, x):
#        return ValueError('Not Differentiable')
#            
#    def convex_conjugate(self,x):
#        if self.b is None:
#            return SimpleL1Norm.convex_conjugate(self, x)
#        else:
#            return SimpleL1Norm.convex_conjugate(self, x) + (self.b * x).sum()
#    
#    def proximal(self, x, tau):
#        
#        if self.b is None:
#            return SimpleL1Norm.proximal(self, x, tau)
#        else:
#            return self.b + SimpleL1Norm.proximal(self, x + self.b , tau)
#        
#    def proximal_conjugate(self, x, tau):
#        
#        if self.b is None:
#            return SimpleL1Norm.proximal_conjugate(self, x, tau)
#        else:
#            return SimpleL1Norm.proximal_conjugate(self, x - tau*self.b, tau)
#                        
#
#############################   mixed_L1,2NORM FUNCTIONS   #############################
#class mixed_L12Norm(Function):
#    
#    def __init__(self, A, b=None, alpha=1, **kwargs):
#
#        super(mixed_L12Norm, self).__init__() 
#        self.alpha = alpha        
#        self.A = A
#        self.b = b
#        
#        self.sym_grad = kwargs.get('sym_grad',False)
#
#        
#            
#    def gradient(self,x):
#        return ValueError('Not Differentiable')
#        
#        
#    def __call__(self,x):
#        
#        y = self.A.direct(x)     
#        eucl_norm = ImageData(y.power(2).sum(axis=0)).sqrt()       
#        eucl_norm.__isub__(self.b)
#        return eucl_norm.sum() * self.alpha 
#    
#    def convex_conjugate(self,x):
#        return 0
#    
#    def proximal_conjugate(self, x, tau): 
#        
#        if self.b is None:  
#              
#            if self.sym_grad:
#                tmp2 = np.sqrt(x.as_array()[0]**2 +  x.as_array()[1]**2 +  2*x.as_array()[2]**2)/self.alpha
#                res = x.divide(ImageData(tmp2).maximum(1.0))                                
#            else:
#                res = x.divide((ImageData(x.power(2).sum(axis=0)).sqrt()/self.alpha).maximum(1.0))  
#                                                   
#        else:            
#            res =  (x - tau*self.b)/ ((x - tau*self.b)).abs().maximum(1.0)
#
#        return res   
#    
#
##%%
#                  
#class ZeroFun(Function):
#    
#    def __init__(self):
#        super(ZeroFun, self).__init__()
#              
#    def __call__(self,x):
#        return 0
#    
#    def convex_conjugate(self, x):
#        ''' This is the support function sup <x, x^{*}>  which in fact is the 
#        indicator function for the set = {0}
#        So 0 if x=0, or inf if x neq 0
#        '''
#        return x.maximum(0).sum()
#    
#    def proximal(self,x,tau):
#        return x.copy()
#        
#    def proximal_conjugate(self, x, tau):
#        return 0
#            
#        
#class CompositeFunction(Function):
#    
#    def __init__(self, *args):
#        self.functions = args
#        self.length = len(self.functions)
#        
#    def get_item(self, ind):        
#        return self.functions[ind]        
#                
#    def __call__(self,x):
#        
#        t = 0
#        for i in range(self.length):
#            for j in range(x.shape[0]):
#                t +=self.functions[i](x.get_item(j))
#        return t       
#
#    def convex_conjugate(self, x):
#        
#        z = 0
#        t = 0
#        for i in range(x.shape[0]):
#            t += self.functions[z].convex_conjugate(x.get_item(i))
#            z += 1        
#
#        return t 
#    
#    def proximal_conjugate(self, x, tau, out = None):
#        
#        if isinstance(tau, Number):
#            tau = CompositeDataContainer(tau)
#        out = [None]*self.length
#        for i in range(self.length):
#            out[i] = self.functions[i].proximal(x.get_item(i), tau.get_item(0))
#        return CompositeDataContainer(*out) 
#
#                            
#    def proximal_conjugate(self, x, tau, out = None):
#        
#        if isinstance(tau, Number):
#            tau = CompositeDataContainer(tau)
#        out = [None]*self.length
#        for i in range(self.length):
#            out[i] = self.functions[i].proximal_conjugate(x.get_item(i), tau.get_item(0))
#        return CompositeDataContainer(*out)    
#    
#            
# 
#    
#if __name__ == '__main__':    
#    
#    N = 3
#    ig = (N,N)
#    ag = ig       
#    op1 = Gradient(ig)
#    op2 = Identity(ig, ag)
#
#    # Form Composite Operator
#    operator = CompositeOperator((2,1), op1, op2 ) 
#    
#    # Create functions
#    alpha = 1
#    noisy_data = ImageData(np.random.randint(10, size=ag))
#    f = CompositeFunction(L1Norm(op1,alpha), \
#                      L2NormSq(op2, noisy_data, c = 0.5, memopt = False) )    
#    
#    u = ImageData(np.random.randint(10, size=ig))
#    uComp = CompositeDataContainer(u)
#
#    print(f(uComp)) # This is f(Kx) = f1(K1*u) + f2(K2*u) 
#
#    f1 = L1Norm(op1,alpha) 
#    f2 = L2NormSq(op2, noisy_data, c = 0.5, memopt = False)
#    
#    print(f1(u) + f2(u)) 
#        
#
#        
