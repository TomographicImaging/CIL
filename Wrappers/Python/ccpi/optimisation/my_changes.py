#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 10:06:44 2019

@author: evangelos
"""

from ccpi.optimisation.ops import Operator
from ccpi.optimisation.funcs import Norm2, Function
from ccpi.framework import DataContainer
import numpy as np
import time
import matplotlib.pyplot as plt


class FiniteDiff(Operator):
    
    def __init__(self):
        self.s1 = np.sqrt(8.0)
        super(FiniteDiff, self).__init__()
        
    def direct(self, x, out=None):
        
        shape = [len(x.shape), ] + list(x.shape)
        gradient = np.zeros(shape, dtype=x.as_array().dtype)
        slice_all = [0, slice(None, -1),]
        for d in range(len(x.shape)):
            gradient[slice_all] = np.diff(x.as_array(), axis=d)
            slice_all[0] = d + 1
            slice_all.insert(1, slice(None))  
        if self.memopt:    
            out.fill(type(x)(gradient, geometry = x.geometry))  
        else:
            return type(x)(gradient, geometry = x.geometry)
             
                       
    def adjoint(self, x, out=None):
        
        res = np.zeros(x.shape[1:])
        for d in range(x.shape[0]):
            this_grad = np.rollaxis(x.as_array()[d], d)
            this_res = np.rollaxis(res, d)
            this_res[:-1] += this_grad[:-1]
            this_res[1:-1] -= this_grad[:-2]
            this_res[-1] -= this_grad[-2]
            
        if self.memopt:    
            out.fill(type(x)(-res, geometry = x.geometry))  
        else:
            return type(x)(-res, geometry = x.geometry)            
                              
    def get_max_sing_val(self):
        return self.s1  
        
class TV(Norm2):
    
    def __init__(self, gamma):
        super(TV,self).__init__(gamma, 0)
        self.op = FiniteDiff()
        self.L = self.op.get_max_sing_val()   
        self.memopt = False
        
    def __call__(self, x):        
        return self.gamma * np.sum(np.sqrt(np.sum(np.square(self.op.direct(x)), self.direction,
                                  keepdims=True))) 
        
    def prox(self, x):        
        return self.gamma * np.sum(np.sqrt(np.sum(np.square(self.op.direct(x)), self.direction,
                                  keepdims=True))) 

    def proximal(self, x, out = None):
        
        tmp = np.sqrt(np.sum(x**2, axis =0))/self.gamma
        res = x.as_array()/np.maximum(1.0, tmp)
        
        if self.memopt:            
            out.fill(type(x)(res,geometry = x.geometry))       
        else:
            return type(x)(res,geometry = x.geometry)
        

# Define a class for squared 2-norm
class Norm2sq_new(Function):
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
        super(Norm2sq_new, self).__init__()
    
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
                    
    def proximal(self, x, tau, out = None):
        
        if self.memopt:
           data.multiply(tau, out=self.direct_placehold) 
           self.direct_placehold += x
           self.direct_placehold *= (1/(1+tau))
           out.fill(self.direct_placehold)
        else:
            res = ( x + tau * self.b )/(1 + tau)
            return type(x)(res.as_array(),geometry=x.geometry)
        
class L1Norm(Function):

    def __init__(self,A,b,c=1.0):
        self.A = A
        self.b = b
        self.c = c
        super(L1Norm, self).__init__()      
            
    def proximal(self, x, tau, out = None):                     
        return self.b + (x - self.b).sign() * ((x - self.b).abs() - tau).maximum(0)
    
    def __call__(self,x):
        y = self.A.direct(x)
        y.__isub__(self.b)
        return y.sum() * self.c
                    
    
def PDHG(data, regulariser, fidelity, operator, tau = None, sigma = None, opt = None ):

                              
    if regulariser is None: regulariser = ZeroFun()
    if fidelity is None: fidelity = ZeroFun()
                
    # algorithmic parameters
    if opt is None: 
        opt = {'tol': 1e-7, 'iter': 1000, 'memopt':False}
    
    max_iter = opt['iter'] if 'iter' in opt.keys() else 1000
    tol = opt['tol'] if 'tol' in opt.keys() else 1e-4
    memopt = opt['memopt'] if 'memopt' in opt.keys() else False  
    show_iter = opt['show_iter'] if 'show_iter' in opt.keys() else False  
            
    obj_value = []
    
    # initialization
    if memopt:
        
        x_init = DataContainer(np.ones(data.shape))
        y_init = DataContainer(np.ones([len(data.shape), ] + list(data.shape) )) 
##        
        x = x_init.copy()        
        x_old = x_init.copy()
        x_tmp = x_init.copy() 
        xbar = x_init.copy()        
        
        y = y_init.copy()
        y_old = y_init.copy()        
        y_tmp = y_init.copy()
                    
    else:
        
        x_old = DataContainer(np.zeros(data.shape))
        xbar = DataContainer(np.zeros(data.shape))
        y_old = DataContainer(np.zeros([len(data.shape), ] + list(data.shape) )) 
            
    # Start time
    t = time.time()
    
    # Compute error
    error_cmp = Norm2()
    
    # theta value
    theta = 1
                    
    # Show results
    print('Iter {:<5} || {:<5} PrimalObj {:<5} || {:<4} l2_error'.format(' ',' ',' ',' '))
    
    
    for it in range(max_iter):
        
        if memopt:
            
            operator.direct(xbar, out = y_tmp)
            y_tmp *= sigma
            y_tmp += y_old
                     
            regulariser.proximal(y_tmp, out = y)
            
            operator.adjoint(y, out = x_tmp)
            x_tmp *=tau
            x_tmp -= x_old
                        
            fidelity.proximal(x_tmp, tau, out = x)
            
            x.subtract(x_old, out = xbar)
            xbar *= theta
            xbar += x
            
            x_old.fill(x)
            y_old.fill(y)
                    
        else:
            
           y_tmp = y_old + sigma * operator.direct(xbar)                      
           y = regulariser.proximal(y_tmp, regulariser.gamma)
           x_tmp = x_old - tau * operator.adjoint(y)
           x = fidelity.proximal(x_tmp, tau)
           xbar = x + theta * (x - x_old) 
           x_old = x
           y_old = y
                   
        
#        error = error_cmp(x-x_old)     
      

                    
        # Compute objective ( primal function ) 
#        obj_value.append(regulariser(x_old) + fidelity(x_old))
         
#        if error < tol:
#           break
        
#        if it % show_iter==0:
#            print('{} {:<5} || {:<5} {:.4f} {:<5} || {:<5} {:.4g}'.format(it,' ',' ',obj_value[it],' ',' ',error))                 
            
        plt.imshow(x.as_array())
        plt.show()        


    # End time        
#    t_end = time.time()        
        
    return x, t_end - t, obj_value, error    


   






