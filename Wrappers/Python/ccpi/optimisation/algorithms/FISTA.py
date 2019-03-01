# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 11:07:30 2019

@author: ofn77899
"""

from ccpi.optimisation.algorithms import Algorithm
from ccpi.optimisation.funcs import ZeroFun
import numpy

class FISTA(Algorithm):
    '''Fast Iterative Shrinkage-Thresholding Algorithm
    
    Beck, A. and Teboulle, M., 2009. A fast iterative shrinkage-thresholding 
    algorithm for linear inverse problems. 
    SIAM journal on imaging sciences,2(1), pp.183-202.
    
    Parameters:
      x_init: initial guess
      f: data fidelity
      g: regularizer
      h:
      opt: additional algorithm 
    '''

    def __init__(self, **kwargs):
        '''initialisation can be done at creation time if all 
        proper variables are passed or later with set_up'''
        super(FISTA, self).__init__()
        self.f = None
        self.g = None
        self.invL = None
        self.t_old = 1
        args = ['x_init', 'f', 'g', 'opt']
        for k,v in kwargs.items():
            if k in args:
                args.pop(args.index(k))
        if len(args) == 0:
            return self.set_up(kwargs['x_init'],
                               f=kwargs['f'],
                               g=kwargs['g'],
                               opt=kwargs['opt'])
    
    def set_up(self, x_init, f=None, g=None, opt=None):
        
        # default inputs
        if f   is None: 
            self.f = ZeroFun()
        else:
            self.f = f
        if g   is None:
            g = ZeroFun()
            self.g = g
        else:
            self.g = g
        
        # algorithmic parameters
        if opt is None: 
            opt = {'tol': 1e-4, 'memopt':False}
        
        self.tol = opt['tol'] if 'tol' in opt.keys() else 1e-4
        memopt = opt['memopt'] if 'memopt' in opt.keys() else False
        self.memopt = memopt
            
        # initialization
        if memopt:
            self.y = x_init.clone()
            self.x_old = x_init.clone()
            self.x = x_init.clone()
            self.u = x_init.clone()
        else:
            self.x_old = x_init.copy()
            self.y = x_init.copy()
        
        #timing = numpy.zeros(max_iter)
        #criter = numpy.zeros(max_iter)
        
    
        self.invL = 1/f.L
        
        self.t_old = 1
            
    def update(self):
    # algorithm loop
    #for it in range(0, max_iter):
    
        if self.memopt:
            # u = y - invL*f.grad(y)
            # store the result in x_old
            self.f.gradient(self.y, out=self.u)
            self.u.__imul__( -self.invL )
            self.u.__iadd__( self.y )
            # x = g.prox(u,invL)
            self.g.proximal(self.u, self.invL, out=self.x)
            
            self.t = 0.5*(1 + numpy.sqrt(1 + 4*(self.t_old**2)))
            
            # y = x + (t_old-1)/t*(x-x_old)
            self.x.subtract(self.x_old, out=self.y)
            self.y.__imul__ ((self.t_old-1)/self.t)
            self.y.__iadd__( self.x )
            
            self.x_old.fill(self.x)
            self.t_old = self.t
            
            
        else:
            u = self.y - self.invL*self.f.grad(self.y)
            
            self.x = self.g.prox(u,self.invL)
            
            self.t = 0.5*(1 + numpy.sqrt(1 + 4*(self.t_old**2)))
            
            self.y = self.x + (self.t_old-1)/self.t*(self.x-self.x_old)
            
            self.x_old = self.x.copy()
            self.t_old = self.t
        
    def update_objective(self):
        self.loss.append( self.f(self.x) + self.g(self.x) )