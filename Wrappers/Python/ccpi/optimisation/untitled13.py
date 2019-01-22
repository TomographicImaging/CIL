#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 21:02:37 2019

@author: evangelos
"""

""" min_{u} alpha ||\nabla u||_{1} + 0.5 * ||Au - g||_{2}^{2} """
    

def PDHG_Kop(data, regulariser, fidelity, operator, tau = None, sigma = None, opt = None ):

                              
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
        
        y_tmp = y_old + sigma * operator.direct(xbar)  
        y = regulariser.proximal(y_tmp, regulariser.gamma)
    
        x_tmp = x_old - tau * operator.adjoint(y)
        x = fidelity.proximal(x_tmp, tau)        

        xbar = x + theta * (x - x_old) 
        
        error = error_cmp(x-x_old)     
      
        x_old = x
        y_old = y
                    
        # Compute objective ( primal function ) 
        obj_value.append(regulariser(x_old) + fidelity(x_old))
         
        if error < tol:
           break
        
        if it % show_iter==0:
            print('{} {:<5} || {:<5} {:.4f} {:<5} || {:<5} {:.4g}'.format(it,' ',' ',obj_value[it],' ',' ',error))                 
            
#        plt.imshow(x.as_array())
#        plt.show()        


    # End time        
    t_end = time.time()        
        
    return x, t_end - t, obj_value, error 