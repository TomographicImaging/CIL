# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library developed by
#   Visual Analytics and Imaging System Group of the Science Technology
#   Facilities Council, STFC

#   Copyright 2018 Jakob Jorgensen, Daniil Kazantsev and Edoardo Pasca

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

from ccpi.optimisation.funcs import Function
from ccpi.optimisation.funcs import ZeroFun
from ccpi.framework import ImageData, AcquisitionData

def FISTA(x_init, f=None, g=None, opt=None):
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
    # default inputs
    if f   is None: f = ZeroFun()
    if g   is None: g = ZeroFun()
    
    # algorithmic parameters
    if opt is None: 
        opt = {'tol': 1e-4, 'iter': 1000, 'memopt':False}
    
    max_iter = opt['iter'] if 'iter' in opt.keys() else 1000
    tol = opt['tol'] if 'tol' in opt.keys() else 1e-4
    memopt = opt['memopt'] if 'memopt' in opt.keys() else False
        
        
    # initialization
    if memopt:
        y = x_init.clone()
        x_old = x_init.clone()
        x = x_init.clone()
        u = x_init.clone()
    else:
        x_old = x_init
        y = x_init;
    
    timing = numpy.zeros(max_iter)
    criter = numpy.zeros(max_iter)
    
    invL = 1/f.L
    
    t_old = 1
    
    c = f(x_init) + g(x_init)

    # algorithm loop
    for it in range(0, max_iter):
    
        time0 = time.time()
        if memopt:
            # u = y - invL*f.grad(y)
            # store the result in x_old
            f.gradient(y, out=u)
            u.__imul__( -invL )
            u.__iadd__( y )
            # x = g.prox(u,invL)
            g.proximal(u, invL, out=x)
            
            t = 0.5*(1 + numpy.sqrt(1 + 4*(t_old**2)))
            
            # y = x + (t_old-1)/t*(x-x_old)
            x.subtract(x_old, out=y)
            y.__imul__ ((t_old-1)/t)
            y.__iadd__( x )
            
            x_old.fill(x)
            t_old = t
            
            
        else:
            u = y - invL*f.grad(y)
            
            x = g.prox(u,invL)
            
            t = 0.5*(1 + numpy.sqrt(1 + 4*(t_old**2)))
            
            y = x + (t_old-1)/t*(x-x_old)
            
            x_old = x.copy()
            t_old = t
        
        # time and criterion
        timing[it] = time.time() - time0
        criter[it] = f(x) + g(x);
        
        # stopping rule
        #if np.linalg.norm(x - x_old) < tol * np.linalg.norm(x_old) and it > 10:
        #   break
    
        #print(it, 'out of', 10, 'iterations', end='\r');

    #criter = criter[0:it+1];
    timing = numpy.cumsum(timing[0:it+1]);
    
    return x, it, timing, criter

def FBPD(x_init, operator=None, constraint=None, data_fidelity=None,\
         regulariser=None, opt=None):
    '''FBPD Algorithm
    
    Parameters:
      x_init: initial guess
      f: constraint
      g: data fidelity
      h: regularizer
      opt: additional algorithm 
    '''
    # default inputs
    if constraint    is None: constraint    = ZeroFun()
    if data_fidelity is None: data_fidelity = ZeroFun()
    if regulariser   is None: regulariser   = ZeroFun()
    
    # algorithmic parameters
    if opt is None: 
        opt = {'tol': 1e-4, 'iter': 1000}
    else:
        try:
            max_iter = opt['iter']
        except KeyError as ke:
            opt[ke] = 1000
        try:
            opt['tol'] = 1000
        except KeyError as ke:
            opt[ke] = 1e-4
    tol = opt['tol']
    max_iter = opt['iter']
    memopt = opt['memopts'] if 'memopts' in opt.keys() else False
    
    # step-sizes
    tau   = 2 / (data_fidelity.L + 2);
    sigma = (1/tau - data_fidelity.L/2) / regulariser.L;
    
    inv_sigma = 1/sigma

    # initialization
    x = x_init
    y = operator.direct(x);
    
    timing = numpy.zeros(max_iter)
    criter = numpy.zeros(max_iter)

    
    
        
    # algorithm loop
    for it in range(0, max_iter):
    
        t = time.time()
    
        # primal forward-backward step
        x_old = x;
        x = x - tau * ( data_fidelity.grad(x) + operator.adjoint(y) );
        x = constraint.prox(x, tau);
    
        # dual forward-backward step
        y = y + sigma * operator.direct(2*x - x_old);
        y = y - sigma * regulariser.prox(inv_sigma*y, inv_sigma);   

        # time and criterion
        timing[it] = time.time() - t
        criter[it] = constraint(x) + data_fidelity(x) + regulariser(operator.direct(x))
           
        # stopping rule
        #if np.linalg.norm(x - x_old) < tol * np.linalg.norm(x_old) and it > 10:
        #   break

    criter = criter[0:it+1]
    timing = numpy.cumsum(timing[0:it+1])
    
    return x, it, timing, criter

def CGLS(x_init, operator , data , opt=None):
    '''Conjugate Gradient Least Squares algorithm
    
    Parameters:
      x_init: initial guess
      operator: operator for forward/backward projections
      data: data to operate on
      opt: additional algorithm 
    '''
    
    if opt is None: 
        opt = {'tol': 1e-4, 'iter': 1000}
    else:
        try:
            max_iter = opt['iter']
        except KeyError as ke:
            opt[ke] = 1000
        try:
            opt['tol'] = 1000
        except KeyError as ke:
            opt[ke] = 1e-4
    tol = opt['tol']
    max_iter = opt['iter']
    
    r = data.copy()
    x = x_init.copy()
    
    d = operator.adjoint(r)
    
    normr2 = (d**2).sum()
    
    timing = numpy.zeros(max_iter)
    criter = numpy.zeros(max_iter)

    # algorithm loop
    for it in range(0, max_iter):
    
        t = time.time()
        
        Ad = operator.direct(d)
        alpha = normr2/( (Ad**2).sum() )
        x  = x + alpha*d
        r  = r - alpha*Ad
        s  = operator.adjoint(r)
        
        normr2_new = (s**2).sum()
        beta = normr2_new/normr2
        normr2 = normr2_new
        d = s + beta*d
        
        # time and criterion
        timing[it] = time.time() - t
        criter[it] = (r**2).sum()
    
    return x, it, timing, criter

def SIRT(x_init, operator , data , opt=None, constraint=None):
    '''Simultaneous Iterative Reconstruction Technique
    
    Parameters:
      x_init: initial guess
      operator: operator for forward/backward projections
      data: data to operate on
      opt: additional algorithm 
      constraint: func of Indicator type specifying convex constraint.
    '''
    
    if opt is None: 
        opt = {'tol': 1e-4, 'iter': 1000}
    else:
        try:
            max_iter = opt['iter']
        except KeyError as ke:
            opt[ke] = 1000
        try:
            opt['tol'] = 1000
        except KeyError as ke:
            opt[ke] = 1e-4
    tol = opt['tol']
    max_iter = opt['iter']
    
    # Set default constraint to unconstrained
    if constraint==None:
        constraint = Function()
    
    x = x_init.clone()
    
    timing = numpy.zeros(max_iter)
    criter = numpy.zeros(max_iter)
    
    # Relaxation parameter must be strictly between 0 and 2. For now fix at 1.0
    relax_par = 1.0
    
    # Set up scaling matrices D and M.
    im1 = ImageData(geometry=x_init.geometry)
    im1.array[:] = 1.0
    M = 1/operator.direct(im1)
    del im1
    aq1 = AcquisitionData(geometry=M.geometry)
    aq1.array[:] = 1.0
    D = 1/operator.adjoint(aq1)
    del aq1
    
    # algorithm loop
    for it in range(0, max_iter):
        t = time.time()
        r = data - operator.direct(x)
        
        x = constraint.prox(x + relax_par * (D*operator.adjoint(M*r)),None)
        
        timing[it] = time.time() - t
        if it > 0:
            criter[it-1] = (r**2).sum()
    
    r = data - operator.direct(x)
    criter[it] = (r**2).sum()
    return x, it, timing,  criter
    
