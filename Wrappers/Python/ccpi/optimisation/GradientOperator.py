#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 16:21:24 2019

@author: evangelos
"""

import numpy as np
from operators import Operator

from ccpi.optimisation.ops import PowerMethodNonsquare
from ccpi.framework import ImageData, ImageGeometry


class FiniteDiff(Operator):
    
    # Works for Neum/Symmetric &  periodic boundary conditions
    # TODO add central differences???
    # TODO not very well optimised, too many conditions
    # TODO add discretisation step, should get that from imageGeometry
    
    # Grad_order = ['channels', 'direction_z', 'direction_y', 'direction_x']
    # Grad_order = ['channels', 'direction_y', 'direction_x']
    # Grad_order = ['direction_z', 'direction_y', 'direction_x']
    # Grad_order = ['channels', 'direction_z', 'direction_y', 'direction_x']
    
    def __init__(self, gm_domain, gm_range=None, direction=0, bnd_cond = 'Neumann'):
        
        self.gm_domain = gm_domain
        self.gm_range = gm_range
        self.direction = direction
        self.bnd_cond = bnd_cond
        
        if self.gm_range is None:
            self.gm_range = self.gm_domain
            
        if self.direction + 1 > len(gm_domain):
            raise ValueError('Gradient directions more than geometry domain')      

        super(FiniteDiff, self).__init__()                  
        
    def direct(self, x, out=None):
                        
        x_asarr = x.as_array()
        x_sz = len(x.shape)
        
        if out is None:        
            out = np.zeros(x.shape)
        
        fd_arr = out
          
        ######################## Direct for 2D  ###############################
        if x_sz == 2:
            
            if self.direction == 1:
                
                np.subtract( x_asarr[:,1:], x_asarr[:,0:-1], out = fd_arr[:,0:-1] )
                
                if self.bnd_cond == 'Neumann':
                    pass                                        
                elif self.bnd_cond == 'Periodic':
                    np.subtract( x_asarr[:,0], x_asarr[:,-1], out = fd_arr[:,-1] )
                else: 
                    raise ValueError('No valid boundary conditions')
                
            if self.direction == 0:
                
                np.subtract( x_asarr[1:], x_asarr[0:-1], out = fd_arr[0:-1,:] )

                if self.bnd_cond == 'Neumann':
                    pass                                        
                elif self.bnd_cond == 'Periodic':
                    np.subtract( x_asarr[0,:], x_asarr[-1,:], out = fd_arr[-1,:] ) 
                else:    
                    raise ValueError('No valid boundary conditions') 
                    
        ######################## Direct for 3D  ###############################                        
        elif x_sz == 3:
                    
            if self.direction == 0:  
                
                np.subtract( x_asarr[1:,:,:], x_asarr[0:-1,:,:], out = fd_arr[0:-1,:,:] )
                
                if self.bnd_cond == 'Neumann':
                    pass
                elif self.bnd_cond == 'Periodic':
                    np.subtract( x_asarr[0,:,:], x_asarr[-1,:,:], out = fd_arr[-1,:,:] ) 
                else:    
                    raise ValueError('No valid boundary conditions')                      
                                                             
            if self.direction == 1:
                
                np.subtract( x_asarr[:,1:,:], x_asarr[:,0:-1,:], out = fd_arr[:,0:-1,:] ) 
                
                if self.bnd_cond == 'Neumann':
                    pass
                elif self.bnd_cond == 'Periodic':
                    np.subtract( x_asarr[:,0,:], x_asarr[:,-1,:], out = fd_arr[:,-1,:] )
                else:    
                    raise ValueError('No valid boundary conditions')                      
                                
             
            if self.direction == 2:
                
                np.subtract( x_asarr[:,:,1:], x_asarr[:,:,0:-1], out = fd_arr[:,:,0:-1] ) 
                
                if self.bnd_cond == 'Neumann':
                    pass
                elif self.bnd_cond == 'Periodic':
                    np.subtract( x_asarr[:,:,0], x_asarr[:,:,-1], out = fd_arr[:,:,-1] )
                else:    
                    raise ValueError('No valid boundary conditions')  
                    
        ######################## Direct for 4D  ###############################
        elif x_sz == 4:
                    
            if self.direction == 0:                            
                np.subtract( x_asarr[1:,:,:,:], x_asarr[0:-1,:,:,:], out = fd_arr[0:-1,:,:,:] )
                
                if self.bnd_cond == 'Neumann':
                    pass
                elif self.bnd_cond == 'Periodic':
                    np.subtract( x_asarr[0,:,:,:], x_asarr[-1,:,:,:], out = fd_arr[-1,:,:,:] )
                else:    
                    raise ValueError('No valid boundary conditions') 
                                                
            if self.direction == 1:
                np.subtract( x_asarr[:,1:,:,:], x_asarr[:,0:-1,:,:], out = fd_arr[:,0:-1,:,:] ) 
                
                if self.bnd_cond == 'Neumann':
                    pass
                elif self.bnd_cond == 'Periodic':
                    np.subtract( x_asarr[:,0,:,:], x_asarr[:,-1,:,:], out = fd_arr[:,-1,:,:] )
                else:    
                    raise ValueError('No valid boundary conditions')                 
                
            if self.direction == 2:
                np.subtract( x_asarr[:,:,1:,:], x_asarr[:,:,0:-1,:], out = fd_arr[:,:,0:-1,:] ) 
                
                if self.bnd_cond == 'Neumann':
                    pass
                elif self.bnd_cond == 'Periodic':
                    np.subtract( x_asarr[:,:,0,:], x_asarr[:,:,-1,:], out = fd_arr[:,:,-1,:] )
                else:    
                    raise ValueError('No valid boundary conditions')                   
                
            if self.direction == 3:
                np.subtract( x_asarr[:,:,:,1:], x_asarr[:,:,:,0:-1], out = fd_arr[:,:,:,0:-1] )                 

                if self.bnd_cond == 'Neumann':
                    pass
                elif self.bnd_cond == 'Periodic':
                    np.subtract( x_asarr[:,:,:,0], x_asarr[:,:,:,-1], out = fd_arr[:,:,:,-1] )
                else:    
                    raise ValueError('No valid boundary conditions')                   
                                
        else:
            raise NotImplementedError                
         
        if self.scalar!=1: 
            res = type(x)(self.scalar * out)
        res = type(x)(out)    
        return res
                    
    def adjoint(self, x, out=None):
        
        x_asarr = x.as_array()
        x_sz = len(x.shape)
        
        if out is None:        
            out = np.zeros(x.shape)
        
        fd_arr = out
        
        ######################## Adjoint for 2D  ###############################
        if x_sz == 2:        
        
            if self.direction == 1:
                
                np.subtract( x_asarr[:,1:], x_asarr[:,0:-1], out = fd_arr[:,1:] )
                
                if self.bnd_cond == 'Neumann':
                    np.subtract( x_asarr[:,0], 0, out = fd_arr[:,0] )
                    np.subtract( -x_asarr[:,-2], 0, out = fd_arr[:,-1] )
                    
                elif self.bnd_cond == 'Periodic':
                    np.subtract( x_asarr[:,0], x_asarr[:,-1], out = fd_arr[:,0] )                                        
                    
                else:   
                    raise ValueError('No valid boundary conditions') 
                                    
            if self.direction == 0:
                
                np.subtract( x_asarr[1:,:], x_asarr[0:-1,:], out = fd_arr[1:,:] )
                
                if self.bnd_cond == 'Neumann':
                    np.subtract( x_asarr[0,:], 0, out = fd_arr[0,:] )
                    np.subtract( -x_asarr[-2,:], 0, out = fd_arr[-1,:] ) 
                    
                elif self.bnd_cond == 'Periodic':  
                    np.subtract( x_asarr[0,:], x_asarr[-1,:], out = fd_arr[0,:] ) 
                    
                else:   
                    raise ValueError('No valid boundary conditions')     
        
        ######################## Adjoint for 3D  ###############################        
        elif x_sz == 3:                
                
            if self.direction == 0:          
                  
                np.subtract( x_asarr[1:,:,:], x_asarr[0:-1,:,:], out = fd_arr[1:,:,:] )
                
                if self.bnd_cond == 'Neumann':
                    np.subtract( x_asarr[0,:,:], 0, out = fd_arr[0,:,:] )
                    np.subtract( -x_asarr[-2,:,:], 0, out = fd_arr[-1,:,:] )
                elif self.bnd_cond == 'Periodic':                     
                    np.subtract( x_asarr[0,:,:], x_asarr[-1,:,:], out = fd_arr[0,:,:] )
                else:   
                    raise ValueError('No valid boundary conditions')                     
                                    
            if self.direction == 1:
                np.subtract( x_asarr[:,1:,:], x_asarr[:,0:-1,:], out = fd_arr[:,1:,:] )
                
                if self.bnd_cond == 'Neumann':
                    np.subtract( x_asarr[:,0,:], 0, out = fd_arr[:,0,:] )
                    np.subtract( -x_asarr[:,-2,:], 0, out = fd_arr[:,-1,:] )
                elif self.bnd_cond == 'Periodic':                     
                    np.subtract( x_asarr[:,0,:], x_asarr[:,-1,:], out = fd_arr[:,0,:] )
                else:   
                    raise ValueError('No valid boundary conditions')                                 
                
            if self.direction == 2:
                np.subtract( x_asarr[:,:,1:], x_asarr[:,:,0:-1], out = fd_arr[:,:,1:] )
                
                if self.bnd_cond == 'Neumann':
                    np.subtract( x_asarr[:,:,0], 0, out = fd_arr[:,:,0] ) 
                    np.subtract( -x_asarr[:,:,-2], 0, out = fd_arr[:,:,-1] ) 
                elif self.bnd_cond == 'Periodic':                     
                    np.subtract( x_asarr[:,:,0], x_asarr[:,:,-1], out = fd_arr[:,:,0] )
                else:   
                    raise ValueError('No valid boundary conditions')                                 
        
        ######################## Adjoint for 4D  ###############################        
        elif x_sz == 4:                
                
            if self.direction == 0:                            
                np.subtract( x_asarr[1:,:,:,:], x_asarr[0:-1,:,:,:], out = fd_arr[1:,:,:,:] )
                
                if self.bnd_cond == 'Neumann':
                    np.subtract( x_asarr[0,:,:,:], 0, out = fd_arr[0,:,:,:] )
                    np.subtract( -x_asarr[-2,:,:,:], 0, out = fd_arr[-1,:,:,:] )
                    
                elif self.bnd_cond == 'Periodic':
                    np.subtract( x_asarr[0,:,:,:], x_asarr[-1,:,:,:], out = fd_arr[0,:,:,:] )
                else:    
                    raise ValueError('No valid boundary conditions') 
                                
            if self.direction == 1:
                np.subtract( x_asarr[:,1:,:,:], x_asarr[:,0:-1,:,:], out = fd_arr[:,1:,:,:] )
                
                if self.bnd_cond == 'Neumann':
                   np.subtract( x_asarr[:,0,:,:], 0, out = fd_arr[:,0,:,:] )
                   np.subtract( -x_asarr[:,-2,:,:], 0, out = fd_arr[:,-1,:,:] )
                    
                elif self.bnd_cond == 'Periodic':
                    np.subtract( x_asarr[:,0,:,:], x_asarr[:,-1,:,:], out = fd_arr[:,0,:,:] )
                else:    
                    raise ValueError('No valid boundary conditions') 
                    
                
            if self.direction == 2:
                np.subtract( x_asarr[:,:,1:,:], x_asarr[:,:,0:-1,:], out = fd_arr[:,:,1:,:] )
                
                if self.bnd_cond == 'Neumann':
                    np.subtract( x_asarr[:,:,0,:], 0, out = fd_arr[:,:,0,:] ) 
                    np.subtract( -x_asarr[:,:,-2,:], 0, out = fd_arr[:,:,-1,:] ) 
                    
                elif self.bnd_cond == 'Periodic':
                    np.subtract( x_asarr[:,:,0,:], x_asarr[:,:,-1,:], out = fd_arr[:,:,0,:] )
                else:    
                    raise ValueError('No valid boundary conditions')                 
                
            if self.direction == 3:
                np.subtract( x_asarr[:,:,:,1:], x_asarr[:,:,:,0:-1], out = fd_arr[:,:,:,1:] )
                
                if self.bnd_cond == 'Neumann':
                    np.subtract( x_asarr[:,:,:,0], 0, out = fd_arr[:,:,:,0] ) 
                    np.subtract( -x_asarr[:,:,:,-2], 0, out = fd_arr[:,:,:,-1] )   
                    
                elif self.bnd_cond == 'Periodic':
                    np.subtract( x_asarr[:,:,:,0], x_asarr[:,:,:,-1], out = fd_arr[:,:,:,0] )
                else:    
                    raise ValueError('No valid boundary conditions')                  
                              
        else:
            raise NotImplementedError
            
        if self.scalar!=1: 
            res = type(x)(-self.scalar * out)
        res = type(x)(-out)    
        return res            
            
    def range_dim(self):
        return self.gm_range
    
    def domain_dim(self):
        return self.gm_domain
       
    def norm(self):
        x0 = ImageData(np.random.random_sample(self.domain_dim()))
        self.s1, sall, svec = PowerMethodNonsquare(self, 25, x0)
        return self.s1
    
    def __str__ (self):
        
        repres = "Gradient direction:{}\n".format(self.direction)                
        return repres

    
    

if __name__ == '__main__':

    u = ImageData(np.random.randint(10, size = (512,512)))
    u1 = ImageData(np.random.randint(10, size = (10,10,10)))
    u2 = ImageData(np.random.randint(10, size = (10,10,10,10)))
    
    w = ImageData(np.random.randint(10, size = u.shape))
    w1 = ImageData(np.random.randint(10, size = u1.shape))
    w2 = ImageData(np.random.randint(10, size = u2.shape))
    
    bnd_cond = 'Periodic'
    
    Gxu = FiniteDiff(u.shape, direction = 1, bnd_cond = bnd_cond)
    Gyu = FiniteDiff(u.shape, direction = 0, bnd_cond = bnd_cond)
    
    Gxu1 = FiniteDiff(u1.shape, direction = 2, bnd_cond = bnd_cond)
    Gyu1 = FiniteDiff(u1.shape, direction = 1, bnd_cond = bnd_cond) 
    Gzu1 = FiniteDiff(u1.shape, direction = 0, bnd_cond = bnd_cond) 
    
    Gxu2 = FiniteDiff(u2.shape, direction = 3, bnd_cond = bnd_cond)
    Gyu2 = FiniteDiff(u2.shape, direction = 2, bnd_cond = bnd_cond) 
    Gzu2 = FiniteDiff(u2.shape, direction = 1, bnd_cond = bnd_cond) 
    Gcu2 = FiniteDiff(u2.shape, direction = 0, bnd_cond = bnd_cond) 
        
    # Check gradient --> adjoint, direct property
    
    print((Gxu.direct(u)*w).sum() - (u*Gxu.adjoint(w)).sum() == 0)
    print((Gyu.direct(u)*w).sum() - (u*Gyu.adjoint(w)).sum() == 0) 
    
    print((Gxu1.direct(u1)*w1).sum() - (u1*Gxu1.adjoint(w1)).sum() == 0)
    print((Gyu1.direct(u1)*w1).sum() - (u1*Gyu1.adjoint(w1)).sum() == 0)
    print((Gzu1.direct(u1)*w1).sum() - (u1*Gzu1.adjoint(w1)).sum() == 0)   
    
    print((Gxu2.direct(u2)*w2).sum() - (u2*Gxu2.adjoint(w2)).sum() == 0)
    print((Gyu2.direct(u2)*w2).sum() - (u2*Gyu2.adjoint(w2)).sum() == 0)
    print((Gzu2.direct(u2)*w2).sum() - (u2*Gzu2.adjoint(w2)).sum() == 0)
    print((Gcu2.direct(u2)*w2).sum() - (u2*Gcu2.adjoint(w2)).sum() == 0)    
        
    #################  < Grad(u), w > = < u, Grad.T(w) > ######################
    
    Grad2D = CompositeOperator((2,1), Gxu, Gyu)
    Grad3D = CompositeOperator((3,1), Gxu1, Gyu1, Gzu1)
    Grad4D = CompositeOperator((4,1), Gxu2, Gyu2, Gzu2, Gcu2)

    grad_2D = Grad2D.direct([u])
    grad_3D = Grad3D.direct([u1])
    grad_4D = Grad4D.direct([u2])
    
    y = [ImageData(np.random.randint(10, size = u.shape)), ImageData(np.random.randint(10, size = u.shape)) ]
    y1 = [ImageData(np.random.randint(10, size = u1.shape)), \
          ImageData(np.random.randint(10, size = u1.shape)), \
          ImageData(np.random.randint(10, size = u1.shape))]
    
    y2 = [ImageData(np.random.randint(10, size = u2.shape)), \
          ImageData(np.random.randint(10, size = u2.shape)), \
          ImageData(np.random.randint(10, size = u2.shape)), \
          ImageData(np.random.randint(10, size = u2.shape))]
    
    grad_tr_2D = Grad2D.adjoint(y)
    grad_tr_3D = Grad3D.adjoint(y1)
    grad_tr_4D = Grad4D.adjoint(y2)
        
    LHS_2D = ImageData(np.zeros(u.shape))
    LHS_3D = ImageData(np.zeros(u1.shape))
    LHS_4D = ImageData(np.zeros(u2.shape))
    
    for i, j in zip(grad_2D, y):
        LHS_2D += i*j
        
    for i, j in zip(grad_3D, y1):
        LHS_3D += i*j

    for i, j in zip(grad_4D, y2):
        LHS_4D += i*j  
        
    
    RHS_2D = u*grad_tr_2D[0]
    RHS_3D = u1*grad_tr_3D[0]
    RHS_4D = u2*grad_tr_4D[0]
    
    print(' LHS = RHS for 2D: ' , LHS_2D.sum()==RHS_2D.sum())
    print(' LHS = RHS for 3D: ' , LHS_3D.sum()==RHS_3D.sum())
    print(' LHS = RHS for 4D: ' , LHS_4D.sum()==RHS_4D.sum())
    
    
    print(Gxu)
    print(Gyu)
    
    # TODO printing in composite operator should return the __str__ for each operator
    # User will now what the direction the gradient is computed
    
    ######################## Symmetrised Gradient #############################
        
    zeroOp = ZeroOp(u.shape, u.shape)                
    E = CompositeOperator((3,2), Gxu, zeroOp, \
                                 zeroOp, Gyu, \
                                 0.5*Gyu, 0.5*Gxu)     
    
    #######################  < E(v), w > = < v, E.T(w) > ######################
    
#    v1 = cp0.containers
    
    v = [ImageData(np.random.randint(10, size = u.shape)),\
         ImageData(np.random.randint(10, size = u.shape))]
    
    Edirect = E.direct(v)
    w = [ImageData(np.random.randint(10, size = u.shape))]
#    for i, j in zip(E, v):
#        LHS_2D += i*j    
#    
#    z1 = E.direct([v1,v2])
    
    
    
    
    
    


