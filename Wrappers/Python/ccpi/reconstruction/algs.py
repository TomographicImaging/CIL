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

from ccpi.reconstruction.funcs import BaseFunction

def FISTA(x_init, f=None, g=None, opt=None):

    # default inputs
    if f   is None: f = BaseFunction()
    if g   is None: g = BaseFunction()
    if opt is None: opt = {'tol': 1e-4, 'iter': 1000}

    # algorithmic parameters
    tol      = opt['tol']
    max_iter = opt['iter']
    
    # initialization
    x_old = x_init
    y = x_init;
    
    timing = numpy.zeros(max_iter)
    criter = numpy.zeros(max_iter)
    
    invL = 1/f.L
    
    t_old = 1

    # algorithm loop
    for it in range(0, max_iter):
    
        time0 = time.time()
        
        u = y - invL*f.grad(y)
        
        x = g.prox(u,invL)
        
        t = 0.5*(1 + numpy.sqrt(1 + 4*(t_old**2)))
        
        y = x + (t_old-1)/t*(x-x_old)
        
        x_old = x
        t_old = t
        
        # time and criterion
        timing[it] = time.time() - time0
        criter[it] = f.fun(x) + g.fun(x);
        
        # stopping rule
        #if np.linalg.norm(x - x_old) < tol * np.linalg.norm(x_old) and it > 10:
        #   break
    
        #print(it, 'out of', 10, 'iterations', end='\r');

    criter = criter[0:it+1];
    timing = numpy.cumsum(timing[0:it+1]);
    
    return x, it, timing, criter

def FBPD(x_init, f=None, g=None, h=None, opt=None):

    # default inputs
    if f   is None: f = BaseFunction()
    if g   is None: g = BaseFunction()
    if h   is None: h = BaseFunction()
    if opt is None: opt = {'tol': 1e-4, 'iter': 1000}

    # algorithmic parameters
    tol      = opt['tol']
    max_iter = opt['iter']
    
    # step-sizes
    tau   = 2 / (g.L + 2);
    sigma = (1/tau - g.L/2) / h.L;
    
    inv_sigma = 1/sigma

    # initialization
    x = x_init
    y = h.dir_op(x);
    
    timing = numpy.zeros(max_iter)
    criter = numpy.zeros(max_iter)

    # algorithm loop
    for it in range(0, max_iter):
    
        t = time.time()
    
        # primal forward-backward step
        x_old = x;
        x = x - tau * ( g.grad(x) + h.adj_op(y) );
        x = f.prox(x, tau);
    
        # dual forward-backward step
        y = y + sigma * h.dir_op(2*x - x_old);
        y = y - sigma * h.prox(inv_sigma*y, inv_sigma);   

        # time and criterion
        timing[it] = time.time() - t
        criter[it] = f.fun(x) + g.fun(x) + h.fun(h.dir_op(x));
           
        # stopping rule
        #if np.linalg.norm(x - x_old) < tol * np.linalg.norm(x_old) and it > 10:
        #   break

    criter = criter[0:it+1];
    timing = numpy.cumsum(timing[0:it+1]);
    
    return x, it, timing, criter
