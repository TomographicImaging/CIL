# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library developed by
#   Visual Analytics and Imaging System Group of the Science Technology
#   Facilities Council, STFC

#   Copyright 2019 Edoardo Pasca

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
import numpy
import time
from ccpi.optimisation.funcs import ZeroFun

class Algorithm(object):
    '''Base class for iterative algorithms

      provides the minimal infrastructure.
      Algorithms are iterables so can be easily run in a for loop. They will
      stop as soon as the stop cryterion is met.
      The user is required to implement the set_up, __init__, update and
      should_stop and update_objective methods
   '''

    def __init__(self):
        self.iteration = 0
        self.stop_cryterion = 'max_iter'
        self.__max_iteration = 0
        self.__loss = []
        self.memopt = False
        self.timing = []
    def set_up(self, *args, **kwargs):
        raise NotImplementedError()
    def update(self):
        raise NotImplementedError()
    
    def should_stop(self):
        '''stopping cryterion'''
        raise NotImplementedError()
    
    def __iter__(self):
        return self
    def next(self):
        '''python2 backwards compatibility'''
        return self.__next__()
    def __next__(self):
        if self.should_stop():
            raise StopIteration()
        else:
            time0 = time.time()
            self.update()
            self.timing.append( time.time() - time0 )
            self.update_objective()
            self.iteration += 1
    def get_output(self):
        '''Returns the solution found'''
        return self.x
    def get_current_loss(self):
        '''Returns the current value of the loss function'''
        return self.__loss[-1]
    def update_objective(self):
        raise NotImplementedError()
    @property
    def loss(self):
        return self.__loss
    @property
    def max_iteration(self):
        return self.__max_iteration
    @max_iteration.setter
    def max_iteration(self, value):
        assert isinstance(value, int)
        self.__max_iteration = value
    
class GradientDescent(Algorithm):
    '''Implementation of a simple Gradient Descent algorithm
    '''

    def __init__(self, **kwargs):
        '''initialisation can be done at creation time if all 
        proper variables are passed or later with set_up'''
        super(GradientDescent, self).__init__()
        self.x = None
        self.rate = 0
        self.objective_function = None
        self.regulariser = None
        args = ['x_init', 'objective_function', 'rate']
        for k,v in kwargs.items():
            if k in args:
                args.pop(args.index(k))
        if len(args) == 0:
            return self.set_up(x_init=kwargs['x_init'],
                               objective_function=kwargs['objective_function'],
                               rate=kwargs['rate'])
    
    def should_stop(self):
        '''stopping cryterion, currently only based on number of iterations'''
        return self.iteration >= self.max_iteration
    
    def set_up(self, x_init, objective_function, rate):
        '''initialisation of the algorithm'''
        self.x = x_init.copy()
        if self.memopt:
            self.x_update = x_init.copy()
        self.objective_function = objective_function
        self.rate = rate
        self.loss.append(objective_function(x_init))
        
    def update(self):
        '''Single iteration'''
        if self.memopt:
            self.objective_function.gradient(self.x, out=self.x_update)
            self.x_update *= -self.rate
            self.x += self.x_update
        else:
            self.x += -self.rate * self.objective_function.grad(self.x)

    def update_objective(self):
        self.loss.append(self.objective_function(self.x))
        


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
            return self.set_up(x_init=kwargs['x_init'],
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
        else:
            self.g = g
        
        # algorithmic parameters
        if opt is None: 
            opt = {'tol': 1e-4, 'iter': 1000, 'memopt':False}
        
        self.max_iteration = opt['iter'] if 'iter' in opt.keys() else 1000
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
        
    def should_stop(self):
        '''stopping cryterion, currently only based on number of iterations'''
        return self.iteration >= self.max_iteration
    
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
        
class FBPD(Algorithm)
    '''FBPD Algorithm
    
    Parameters:
      x_init: initial guess
      f: constraint
      g: data fidelity
      h: regularizer
      opt: additional algorithm 
    '''
    constraint = None
    data_fidelity = None
    regulariser = None
    def __init__(self, **kwargs):
        pass
    def set_up(self, x_init, operator=None, constraint=None, data_fidelity=None,\
         regulariser=None, opt=None):

        # default inputs
        if constraint    is None: 
            self.constraint    = ZeroFun()
        else:
            self.constraint = constraint
        if data_fidelity is None:
            data_fidelity = ZeroFun()
        else:
            self.data_fidelity = data_fidelity
        if regulariser   is None:
            self.regulariser   = ZeroFun()
        else:
            self.regulariser = regulariser
        
        # algorithmic parameters
        
        
        # step-sizes
        self.tau   = 2 / (self.data_fidelity.L + 2)
        self.sigma = (1/self.tau - self.data_fidelity.L/2) / self.regulariser.L
        
        self.inv_sigma = 1/self.sigma
    
        # initialization
        self.x = x_init
        self.y = operator.direct(self.x)
        
    
    def update(self):
    
        # primal forward-backward step
        x_old = self.x
        self.x = self.x - self.tau * ( self.data_fidelity.grad(self.x) + self.operator.adjoint(self.y) )
        self.x = constraint.prox(self.x, self.tau);
    
        # dual forward-backward step
        self.y = self.y + self.sigma * self.operator.direct(2*self.x - x_old);
        self.y = self.y - self.sigma * self.regulariser.prox(self.inv_sigma*self.y, self.inv_sigma);   

        # time and criterion
        self.loss = self.constraint(self.x) + self.data_fidelity(self.x) + self.regulariser(self.operator.direct(self.x))
        
class CGLS(Algorithm):

    '''Conjugate Gradient Least Squares algorithm

    Parameters:
      x_init: initial guess
      operator: operator for forward/backward projections
      data: data to operate on
    '''
    def __init__(self, **kwargs):
        super(CGLS, self).__init__()
        self.x        = kwargs.get('x_init', None)
        self.operator = kwargs.get('operator', None)
        self.data     = kwargs.get('data', None)
        if self.x is not None and self.operator is not None and \
           self.data is not None:
            print ("Calling from creator")
            return self.set_up(x_init  =kwargs['x_init'],
                               operator=kwargs['operator'],
                               data    =kwargs['data'])

    def set_up(self, x_init, operator , data ):

        self.r = data.copy()
        self.x = x_init.copy()

        self.operator = operator
        self.d = operator.adjoint(self.r)

        self.normr2 = self.d.norm()

    def should_stop(self):
        '''stopping cryterion, currently only based on number of iterations'''
        return self.iteration >= self.max_iteration

    def update(self):

        Ad = self.operator.direct(self.d)
        alpha = self.normr2/Ad.norm()
        self.x += alpha * self.d
        self.r -= alpha * Ad
        s  = self.operator.adjoint(self.r)

        normr2_new = s.norm()
        beta = normr2_new/self.normr2
        self.normr2 = normr2_new
        self.d = s + beta*self.d

    def update_objective(self):
        self.loss.append(self.r.norm())
