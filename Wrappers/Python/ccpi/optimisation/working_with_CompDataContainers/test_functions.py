#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 19:24:37 2019

@author: evangelos
"""

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
from operators import CompositeDataContainer, Identity, CompositeOperator
from numbers import Number
from GradientOperator import Gradient

    
class SimpleL2NormSq(Function):
    
    def __init__(self, alpha=1):
        
        super(SimpleL2NormSq, self).__init__()         
        self.alpha = alpha
        
    def __call__(self, x):
        return self.alpha * x.power(2).sum()
    
    def gradient(self,x):
        return 2 * self.alpha * x
    
    def convex_conjugate(self,x):
        return (1/(4*self.alpha)) * x.power(2).sum()
    
    def proximal(self, x, tau):
        return x.divide(1+2*tau*self.alpha)
    
    def proximal_conjugate(self, x, tau):
        return x.divide(1 + tau/(2*self.alpha) )    


############################   L2NORM FUNCTIONS   #############################
class L2NormSq(SimpleL2NormSq):
    
    def __init__(self, alpha, **kwargs):
                 
        super(L2NormSq, self).__init__(alpha)
        self.alpha = alpha
        self.b = kwargs.get('b',None)
        self.L = 1        
        
    def __call__(self, x):
        
        if self.b is None:
            return SimpleL2NormSq.__call__(self, x)
        else:
            return SimpleL2NormSq.__call__(self, x - self.b) 
        
    def gradient(self, x):
        
        if self.b is None:
            return 2*self.alpha * x 
        else:
            return 2*self.alpha * (x - self.b) 
                    
    def composition_with(self, operator):        
        
        if self.b is None:
            return FunctionComposition(L2NormSq(self.alpha), operator)
        else:
            return FunctionComposition(L2NormSq(self.alpha, b=self.b), operator)
                                             
    def convex_conjugate(self, x):
        
        ''' The convex conjugate corresponds to the simple functional i.e., 
        f(x) = alpha * ||x - b||_{2}^{2}
        '''
        
        if self.b is None:
            return SimpleL2NormSq.convex_conjugate(self, x)
        else:
            return SimpleL2NormSq.convex_conjugate(self, x) + (self.b * x).sum()
    
    def proximal(self, x, tau):
        
        ''' The proximal operator corresponds to the simple functional i.e., 
        f(x) = alpha * ||x - b||_{2}^{2}
        
        argmin_x { 0.5||x - u||^{2} + tau f(x) }
        '''
        
        if self.b is None:
            return SimpleL2NormSq.proximal(self, x, tau)
        else:
            return self.b + SimpleL2NormSq.proximal(self, x - self.b , tau)
            
    
    def proximal_conjugate(self, x, tau):
        
        ''' The proximal operator corresponds to the simple convex conjugate 
        functional i.e., f^{*}(x^{)        
        argmin_x { 0.5||x - u||^{2} + tau f(x) }
        '''
        if self.b is None:
            return SimpleL2NormSq.proximal_conjugate(self, x, tau)
        else:
            return SimpleL2NormSq.proximal_conjugate(self, x - tau * self.b, tau)
     
        
############################   L1NORM FUNCTIONS   #############################
class SimpleL1Norm(Function):
    
    def __init__(self, alpha=1):
        
        super(SimpleL1Norm, self).__init__()         
        self.alpha = alpha
        
    def __call__(self, x):
        return self.alpha * x.abs().sum()
    
    def gradient(self,x):
        return ValueError('Not Differentiable')
            
    def convex_conjugate(self,x):
        return 0
    
    def proximal(self, x, tau):
        ''' Soft Threshold'''
        return x.sign() * (x.abs() - tau * self.alpha).maximum(1.0)
        
    def proximal_conjugate(self, x, tau):
        return x.divide((x.abs()/self.alpha).maximum(1.0))
    
class L1Norm(SimpleL1Norm):
    
    def __init__(self, alpha=1, **kwargs):
        
        super(L1Norm, self).__init__()         
        self.alpha = alpha 
        
        self.A = kwargs.get('A',None)
        self.b = kwargs.get('b',None)
        
    def __call__(self, x):
        
        if self.b is None:
            return SimpleL1Norm.__call__(self, self.A.direct(x))
        else:
            return SimpleL1Norm.__call__(self, self.A.direct(x) - self.b)
        
    def eval_norm(self, x):
        
        return SimpleL1Norm.__call__(self, x)        
    
    def gradient(self, x):
        return ValueError('Not Differentiable')
            
    def convex_conjugate(self,x):
        if self.b is None:
            return SimpleL1Norm.convex_conjugate(self, x)
        else:
            return SimpleL1Norm.convex_conjugate(self, x) + (self.b * x).sum()
    
    def proximal(self, x, tau):
        
        if self.b is None:
            return SimpleL1Norm.proximal(self, x, tau)
        else:
            return self.b + SimpleL1Norm.proximal(self, x + self.b , tau)
        
    def proximal_conjugate(self, x, tau):
        
        if self.b is None:
            return SimpleL1Norm.proximal_conjugate(self, x, tau)
        else:
            return SimpleL1Norm.proximal_conjugate(self, x - tau*self.b, tau)
                        

############################   mixed_L1,2NORM FUNCTIONS   #############################
class mixed_L12Norm(Function):
    
    def __init__(self, alpha, **kwargs):

        super(mixed_L12Norm, self).__init__() 
        
        self.alpha = alpha 
        self.b = kwargs.get('b',None)                
        self.sym_grad = kwargs.get('sym_grad',False)
        
    def __call__(self,x):
        
        if self.b is None:
            tmp1 = x
        else:
            tmp1 = x - self.b            
#        
        if self.sym_grad:
            tmp = np.sqrt(tmp1.as_array()[0]**2 +  tmp1.as_array()[1]**2 +  2*tmp1.as_array()[2]**2)
        else:
            tmp = ImageData(tmp1.power(2).sum(axis=0)).sqrt()
            
        return self.alpha*tmp.sum()          
                            
    def gradient(self,x):
        return ValueError('Not Differentiable')
                            
    def convex_conjugate(self,x):
        return 0
    
    def proximal(self, x, tau):
        pass
    
    def proximal_conjugate(self, x, tau): 
        
        if self.sym_grad:
                tmp2 = np.sqrt(x.as_array()[0]**2 +  x.as_array()[1]**2 +  2*x.as_array()[2]**2)/self.alpha
                res = x.divide(ImageData(tmp2).maximum(1.0))                                
        else:
                res = x.divide((ImageData(x.power(2).sum(axis=0)).sqrt()/self.alpha).maximum(1.0))  
                                                   
        return res 
    
    def composition_with(self, operator):
        
        if self.b is None:
            return FunctionComposition(mixed_L12Norm(self.alpha), operator)
        else:
            return FunctionComposition(mixed_L12Norm(self.alpha, b=self.b), operator)    
    

#%%
                  
class ZeroFun(Function):
    
    def __init__(self):
        super(ZeroFun, self).__init__()
              
    def __call__(self,x):
        return 0
    
    def convex_conjugate(self, x):
        ''' This is the support function sup <x, x^{*}>  which in fact is the 
        indicator function for the set = {0}
        So 0 if x=0, or inf if x neq 0                
        '''
        
        if x.shape[0]==1:
            return x.maximum(0).sum()
        else:
            return x.get_item(0).maximum(0).sum() + x.get_item(1).maximum(0).sum()
    
    def proximal(self,x,tau):
        return x.copy()
        
    def proximal_conjugate(self, x, tau):
        return 0
            
        
class CompositeFunction(Function):
    
    def __init__(self, *functions, blockMatrix):

        self.blockMatrix = blockMatrix        
        self.functions = functions
        self.length = len(self.functions)
        
    def get_item(self, ind):        
        return self.functions[ind]        
                
    def __call__(self,x):
                                
        t = 0                
        for i in range(x.shape[0]):
            t += self.functions[i](x.get_item(i))               
        return t            
    

    def convex_conjugate(self, x):
        
        z = 0
        t = 0
        for i in range(x.shape[0]):
            t += self.functions[z].convex_conjugate(x.get_item(i))
            z += 1        

        return t 
    
    def proximal(self, x, tau, out = None):
        
        if isinstance(tau, Number):
            tau = CompositeDataContainer(tau)
        out = [None]*self.length
        for i in range(self.length):
            out[i] = self.functions[i].proximal(x.get_item(i), tau.get_item(0))
        return CompositeDataContainer(*out) 

                            
    def proximal_conjugate(self, x, tau, out = None):
        
        if isinstance(tau, Number):
            tau = CompositeDataContainer(tau)
        if isinstance(x, ImageData):
            x = CompositeDataContainer(x)
        out = [None]*self.length
        for i in range(self.length):
            out[i] = self.functions[i].proximal_conjugate(x.get_item(i), tau.get_item(0))
        return CompositeDataContainer(*out)    
    
#    
#class  FunctionComposition(Function):
#    
#    def __init__(self, function, operator):
#        
#        self.function = function
#        self.alpha = self.function.alpha
#        self.b  = self.function.b         
#        self.operator = operator
#        
#    
#        super(FunctionComposition, self).__init__()
#    
#    '''    overide call and gradient '''
#    def __call__(self, x):        
#        return self.function(x)
#    
#    def gradient(self,x):        
#        return self.operator.adjoint(self.function.gradient(self.operator.direct(x)))
#    
#    ''' Same as in the parent class'''
#    def proximal(self,x, tau):
#        return self.function.proximal(x, tau)
#    
#    def proximal_conjugate(self,x, tau):
#        return self.function.proximal_conjugate(x, tau)
#
#    def convex_conjugate(self,x):
#        return self.function.convex_conjugate(x)    
    
                    
class FunctionComposition_new(Function):
    
     def __init__(self, operator, *functions):
        
        self.functions = functions      
        self.operator = operator
        self.length = len(self.functions)
        
#        if self.length==1:
#            self.L = self.functions[0].alpha*(self.operator.norm()**2)    
                        
        # length == to operator.shape[0]#    
        super(FunctionComposition_new, self).__init__()
                    
     def __call__(self, x):
         
        if isinstance(x, ImageData):
            x =  CompositeDataContainer(x)         
        
        t = 0                
        for i in range(x.shape[0]):
            t += self.functions[i](x.get_item(i))               
        return t 
    
     def convex_conjugate(self, x):
         
        if isinstance(x, ImageData):
            x =  CompositeDataContainer(x)         
        
        t = 0                
        for i in range(x.shape[0]):
            t += self.functions[i].convex_conjugate(x.get_item(i))               
        return t     
                        
     def proximal_conjugate(self, x, tau, out = None):
     
        if isinstance(x, ImageData):
            x =  CompositeDataContainer(x)
            
        out = [None]*self.length
        for i in range(self.length):
            out[i] = self.functions[i].proximal_conjugate(x.get_item(i), tau)
        
        if self.length==1:
            return ImageData(*out)   
        else:
            return CompositeDataContainer(*out)   

     def proximal(self, x, tau, out = None):
     
        if isinstance(x, ImageData):
            x =  CompositeDataContainer(x)
            
        out = [None]*self.length
        for i in range(self.length):
            out[i] = self.functions[i].proximal(x.get_item(i), tau)
        
        if self.length==1:
            return ImageData(*out)   
        else:
            return CompositeDataContainer(*out)            
    
    
if __name__ == '__main__':    
    
    N = 3
    ig = (N,N)
    ag = ig       
    op1 = Gradient(ig)
    op2 = Identity(ig, ag)

    # Form Composite Operator
    operator = CompositeOperator((2,1), op1, op2 ) 
    
    # Create functions
    noisy_data = ImageData(np.random.randint(10, size=ag))
    
    d = ImageData(np.random.randint(10, size=ag))
    
    f = mixed_L12Norm(alpha = 1).composition_with(op1)
    g = L2NormSq(alpha=0.5, b=noisy_data)
    
    # Compare call of f
    a1 = ImageData(op1.direct(d).power(2).sum(axis=0)).sqrt().sum()
    print(a1, f(d))
    
    # Compare call of g
    a2 = g.alpha*(d - noisy_data).power(2).sum()
    print(a2, g(d)) 
    
    # Compare convex conjugate of g
    a3 = 0.5 * d.power(2).sum() + (d*noisy_data).sum()
    print( a3, g.convex_conjugate(d))
    
    
    
    
    
#    
#    f1 = L2NormSq(alpha=1, b=noisy_data)
#    print(f1(noisy_data))
#    
#    f2 =  L2NormSq(alpha=5, b=noisy_data).composition_with(op2)
#    print(f2(noisy_data))
#    
#    print(f1.gradient(noisy_data).as_array())
#    print(f2.gradient(noisy_data).as_array())
##    
#    print(f1.proximal(noisy_data,1).as_array())
#    print(f2.proximal(noisy_data,1).as_array())
#    
#    
#    f3 = mixed_L12Norm(alpha = 1).composition_with(op1)
#    print(f3(noisy_data))
#            
#    print(ImageData(op1.direct(noisy_data).power(2).sum(axis=0)).sqrt().sum())
#    
#    print( 5*(op2.direct(d) - noisy_data).power(2).sum(), f2(d))
#    
#    from functions import mixed_L12Norm as mixed_L12Norm_old
#    
#    print(mixed_L12Norm_old(op1,None,alpha)(noisy_data))
    
    
    #        

        
