# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 11:07:30 2019

@author: ofn77899
"""

from ccpi.optimisation.algorithms import Algorithm
from ccpi.optimisation.functions import ZeroFunction
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
      opt: additional options 
    '''
    
    
    def __init__(self, **kwargs):
        '''initialisation can be done at creation time if all 
        proper variables are passed or later with set_up'''
        super(FISTA, self).__init__()
        self.f = kwargs.get('f', None)
        self.g = kwargs.get('g', None)
        self.x_init = kwargs.get('x_init',None)
        self.invL = None
        self.t_old = 1
        if self.f is not None and self.g is not None:
            print ("FISTA initialising from creator")
            self.set_up(self.x_init, self.f, self.g)        

    
    def set_up(self, x_init, f, g, opt=None, **kwargs):
        
        self.f = f
        self.g = g
        
        # algorithmic parameters
        if opt is None: 
            opt = {'tol': 1e-4}
        print(self.x_init.as_array())
        print(x_init.as_array())
        
        self.y = x_init.copy()
        self.x_old = x_init.copy()
        self.x = x_init.copy()
        self.u = x_init.copy()            


        self.invL = 1/f.L
        
        self.t_old = 1
            
    def update(self):

        self.f.gradient(self.y, out=self.u)
        print ('update, self.u' , self.u.as_array())
        self.u.__imul__( -self.invL )
        self.u.__iadd__( self.y )
        print ('update, self.u' , self.u.as_array())
        
        # x = g.prox(u,invL)
        print ('update, self.x pre prox' , self.x.as_array())
        self.g.proximal(self.u, self.invL, out=self.x)
        print ('update, self.x post prox' , self.x.as_array())
        
        self.t = 0.5*(1 + numpy.sqrt(1 + 4*(self.t_old**2)))
        
        self.x.subtract(self.x_old, out=self.y)
        print ('update, self.y' , self.y.as_array())
        
        self.y.__imul__ ((self.t_old-1)/self.t)
        print ('update, self.x' , self.x.as_array())
        self.y.__iadd__( self.x )
        print ('update, self.y' , self.y.as_array())
        
        self.x_old.fill(self.x)
        print ('update, self.x_old' , self.x_old.as_array())
        self.t_old = self.t            
        
    def update_objective(self):
        self.loss.append( self.f(self.x) + self.g(self.x) )    
    

