#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 10:06:44 2019

@author: evangelos
"""

from ccpi.optimisation.ops import Operator, PowerMethodNonsquare
from ccpi.optimisation.funcs import Norm2, Function
from ccpi.framework import DataContainer, ImageData, ImageGeometry, AcquisitionGeometry, \
    AcquisitionData
import numpy as np
import time
import matplotlib.pyplot as plt


def cmp_L2norm(x, x_old):    
    return np.sqrt((x-x_old).power(2).sum())

def cmp_PDgap(x, y, f, g, operator):   
    PrimalObj = sum([f[j](x) + g(x) for j in range(len(operator))])          
    DualObj =  - sum([f[j].convex_conj(operator[j].adjoint(y[j])) + g.convex_conj(operator[j].adjoint(-1*y[j])) for j in range(len(operator))])    
    PD_Gap = np.abs(PrimalObj - DualObj)
    return PD_Gap, PrimalObj, DualObj

def stop_rule(callback, *args):
    return callback(*args)


#%%        
    
     
    

        
    
class KL_diverg(Function):

    def __init__(self,A,b,c=1.0):
        self.A = A
        self.b = b
        self.c = c
        super(KL_diverg, self).__init__() 
                          
    def proximal(self, x, tau, out = None):                     
        return (0.5 * ( (x-tau) + ( (x-tau).power(2) + 4 * tau * self.b ) )).maximum(1e-12) 
    
    def proximal_conj(self, x, tau, out = None):        
        return 0.5*( (1+x) - ((1+x).power(2) - 4*(x-tau*self.b)).sqrt())
                     
        

                
class ZeroFun(Function):
    
    def __init__(self,gamma=0,L=1):
        self.gamma = gamma
        self.L = L
        super(ZeroFun, self).__init__()
    
    def __call__(self,x):
        return 0
    
    def prox(self,x,tau):
        return x.copy()
    
    def convex_conj(self, x):
        return 0
        
    def proximal(self, x, tau, out=None):
        if out is None:
            return self.prox(x, tau)
        else:
            out.fill(x)
#            if isSizeCorrect(out, x):
#                # check dimensionality  
#                if issubclass(type(out), DataContainer):    
#                    out.fill(x) 
#                            
#                elif issubclass(type(out) , numpy.ndarray): 
#                    out[:] = x  
#            else:   
#                raise ValueError ('Wrong size: x{0} out{1}'
#                                    .format(x.shape,out.shape) )

                    
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
#        self.L = 2.0*self.c*(self.A.get_max_sing_val()**2)
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
        
        
def get_primal_dual_stepsize(sigma, tau):
    # Compute sigma, tau 
    # inside we need power method to compute ||K||
    pass
        
                
#def PDHG(data, f, g, operator, ig, ag, tau = None, sigma = None, opt = None):
#        
#    # algorithmic parameters
#    if opt is None: 
#        opt = {'tol': 1e-6, 'niter': 10, 'show_iter': 100, \
#               'memopt': False} 
#        
#    if sigma is None and tau is None:
#        raise ValueError('Need sigma*tau||K||^2<1') 
#                
#    niter = opt['niter'] if 'niter' in opt.keys() else 1000
#    tol = opt['tol'] if 'tol' in opt.keys() else 1e-4
#    memopt = opt['memopt'] if 'memopt' in opt.keys() else False  
#    show_iter = opt['show_iter'] if 'show_iter' in opt.keys() else False 
#    stop_crit = opt['stop_crit'] if 'stop_crit' in opt.keys() else False 
#
#    # Initialisation
#    x_old = ImageData(geometry=ig)
#    xbar = ImageData(geometry=ig)
#
#    y_old1 = ImageData(np.zeros([len(x_old.shape), ] + list(x_old.shape) ))
#    y_old2 = ImageData(np.zeros(data.shape), geometry = ag) 
#    
#    y_old = [y_old1,y_old2]
#    y_tmp = [None]*len(operator)
#    y = [None]*len(operator)
#        
#    # relaxation parameter
#    theta = 1
#    
#    t = time.time()
#    
#    for i in range(niter):
#        
#        # Gradient ascent in the dual variable y for Grad/Aop operator
#        for i_op in range(len(operator)):
#            y_tmp[i_op] = y_old[i_op] + sigma *  operator[i_op].direct(xbar)
#            y[i_op] = f[i_op].proximal_conj(y_tmp[i_op], sigma)   
#           
#        # Gradient descent in the primal variable x for Grad/Aop operator            
#        x_tmp = x_old - tau * sum([operator[iii].adjoint(y[iii]) for iii in range(len(operator))])                                    
#        x = g.proximal(x_tmp, tau)
#            
##        print(cmp_L2norm(x, x_old))
##        if cmp_L2norm(x, x_old)<1e-3:
##            break
#        # Update
#        xbar = x + theta * (x - x_old) 
#        x_old = x
#        y_old = y   
#
#                           
#    t_end = time.time()        
#        
#    return x, t_end - t, i


def PDHG_memopt(data, f, g, operator, ig, ag, tau = None, sigma = None, opt = None):
        
    # algorithmic parameters
    if opt is None: 
        opt = {'tol': 1e-6, 'niter': 200, 'show_iter': 50, 'memopt': True} 
        
    if sigma is None and tau is None:
        raise ValueError('Need sigma*tau||K||^2<1') 
                
    niter = opt['niter'] if 'niter' in opt.keys() else 1000
    tol = opt['tol'] if 'tol' in opt.keys() else 1e-4
    memopt = opt['memopt'] if 'memopt' in opt.keys() else False  
    show_iter = opt['show_iter'] if 'show_iter' in opt.keys() else False 

    # Initialisation
    x_old = ImageData(geometry=ig)    
    xbar = ImageData(geometry=ig)
    x_tmp1 = x_old.clone()
    x_tmp2 = x_old.clone()
    x_tmp = x_old.clone()
    x = x_old.clone()
    
    y_old1 = ImageData(np.zeros([len(x_old.shape), ] + list(x_old.shape) ))
    y_old2 = ImageData(np.zeros(data.shape), geometry = ag) 
    
    
    y_old = [y_old1,y_old2]
    y_tmp = y_old.copy()    
    y = y_old.copy()
        
    # relaxation parameter
    theta = 1
    

    t = time.time()
    
    for i in range(niter):
        
        # Gradient ascent in the dual variable y for Grad/Aop operator
        for i_op in range(len(operator)):
            
            operator[i_op].direct(xbar, y_tmp[i_op])
            y_old[i_op] += sigma * y_tmp[i_op]
            f[i_op].proximal_conj(y_tmp[i_op], sigma, y[i_op])
        
                        
        # Gradient descent in the primal variable x for Grad/Aop operator 
        operator[0].adjoint(y[0], x_tmp1)
        operator[1].adjoint(y[1], x_tmp2)
        x_tmp.subtract(x_old - tau*(x_tmp1 + x_tmp2), x_tmp)
        g.proximal(x_tmp, tau, x)
        

#        for iii in range(len(operator)):
#            x_tmp.add(operator[iii].adjoint(y[iii], x_tmp))
            
#        x_old -= tau * x_tmp
        
        
            
            
#        sum([operator[iii].adjoint(y[iii], x_tmp) for iii in range(len(operator))])          
#        x_tmp = x_old - tau * sum([operator[iii].adjoint(y[iii]) for iii in range(len(operator))])                                    
#        x = g.proximal(x_tmp, tau)            
            
            
            
            
            
            
            
#            y_tmp[i_op].multiply(sigma, out = y_tmp[i_op])
#            y_tmp[i_op].add(y_old[i_op], out = y_tmp[i_op])
#            
#            f[i_op].proximal_conj(y_tmp[i_op], sigma, out = y[i_op])
#            
##        plt.imshow(x_tmp.as_array())
##        plt.show()            
#            
################################################################################                                    
##            y_tmp[i_op] = y_old[i_op] + sigma *  operator[i_op].direct(xbar)
##            y[i_op] = f[i_op].proximal_conj(y_tmp[i_op], sigma)   
################################################################################
#            
#        # Gradient descent in the primal variable x for Grad/Aop operator  
#                
#        x_tmp = x_old - tau * sum([operator[i].adjoint(y[i]) for i in range(len(operator))])                                    
#        x = g.proximal(x_tmp, tau) 
#        
##        x_old.add(-tau * sum([operator[iii].adjoint(y[iii]) for iii in range(len(operator))]), out = x_tmp)
##        g.proximal(x_tmp, tau, out = x)
        
        # Update
        
#        x.subtract(x_old, out = xbar)
#        xbar *= theta
#        xbar.add(x, xbar)
        
#        x_old.fill(x)
#        for kk in range(len(operator)):
#            y_old[kk].fill(y[kk])
        
        xbar = x + theta * (x - x_old) 
        x_old = x
        y_old = y        
         
        
        
        
###############################################################################          
#        x_tmp = x_old - tau * sum([operator[i].adjoint(y[i]) for i in range(len(operator))])                                    
#        x = g.proximal(x_tmp, tau)
###############################################################################        
        
        
#         Stop Criteria
#        if i%show_iter==0:  

            
#            cmp_error = stop_crit(x, x_old)
#            if cmp_error<tol:
#                return x, t_end - t, error.append(cmp_error)
            
        
        
    t_end = time.time()        
        
    return x, t_end - t  


def compute_opNorm(operator):    
    tmp = 0
    for i in range(len(operator)):
        for j in range(len(operator[0])):
            tmp += operator[i][j].norm()**2
    return np.sqrt(tmp)


def form_Operator(*operator):
    op = []
    for i in operator:
        op.append(i)
    return op 

def create_toy_phantom(N, ig):

    Phantom = ImageData(geometry=ig)

    x = Phantom.as_array()
    x[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
    x[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 1
    
    return x



    
    
                    







