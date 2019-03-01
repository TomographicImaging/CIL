#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 15:31:47 2019

@author: evangelos
"""
from operators import Operator
from ccpi.optimisation.ops import PowerMethodNonsquare
from ccpi.framework import ImageData, DataContainer
import numpy as np

#%%

class Gradient(Operator):
    
    def __init__(self, gm_domain, gm_range=None, bnd_cond = 'Neumann', **kwargs):
        
        super(Gradient, self).__init__() 
        
        self.gm_domain = gm_domain # Domain of Grad Operator
        self.gm_range = gm_range # Range of Grad Operator
        self.bnd_cond = bnd_cond # Boundary conditions of Finite Differences

        
        if self.gm_range is None:
           self.gm_range =  ((len(self.gm_domain),)+self.gm_domain)
    
        # Kwargs Default options            
        self.memopt = kwargs.get('memopt',False)
        self.correlation = kwargs.get('correlation','Space') 
        
        #TODO not tested yet, operator norm???
        self.voxel_size = kwargs.get('voxel_size',[1]*len(gm_domain))  
                                             
        
    def direct(self, x, out=None):
        
        tmp = np.zeros(self.gm_range)
        for i in range(len(self.gm_domain)):
            tmp[i] = FiniteDiff(self.gm_domain, direction = i, bnd_cond = self.bnd_cond).direct(x.as_array())/self.voxel_size[i]            
#        return type(x)(tmp)
        return type(x)(tmp)
    
    def adjoint(self, x, out=None):
            
        tmp = np.zeros(self.gm_domain)
        for i in range(len(self.gm_domain)):
            tmp+=FiniteDiff(self.gm_domain, direction = i, bnd_cond = self.bnd_cond).adjoint(x.as_array()[i])/self.voxel_size[i]  
        return type(x)(tmp)
        
    def alloc_domain_dim(self):
        return ImageData(np.zeros(self.gm_domain))
    
    def alloc_range_dim(self):
        return ImageData(np.zeros(self.range_dim))
    
    def domain_dim(self):
        return self.gm_domain
    
    def range_dim(self):
        return self.gm_range
                                   
    def norm(self):
#        return np.sqrt(4*len(self.domainDim()))        
        #TODO this takes time for big ImageData
        # for 2D ||grad|| = sqrt(8), 3D ||grad|| = sqrt(12)        
        x0 = ImageData(np.random.random_sample(self.domain_dim()))
        self.s1, sall, svec = PowerMethodNonsquare(self, 25, x0)
        return self.s1
    
class SymmetrizedGradient(Operator):
    
    def __init__(self, gm_domain, gm_range, bnd_cond = 'Neumann', **kwargs):
        
        super(SymmetrizedGradient, self).__init__() 
        
        self.gm_domain = gm_domain # Domain of Grad Operator
        self.gm_range = gm_range # Range of Grad Operator
        self.bnd_cond = bnd_cond # Boundary conditions of Finite Differences
    
        # Kwargs Default options            
        self.memopt = kwargs.get('memopt',False)
        self.correlation = kwargs.get('correlation','Space') 
        
        #TODO not tested yet, operator norm???
        self.voxel_size = kwargs.get('voxel_size',[1]*len(gm_domain))  
                                             
        
    def direct(self, x, out=None):
        
        tmp = np.zeros(self.gm_range)
        tmp[0] = FiniteDiff(self.gm_domain[1:], direction = 1, bnd_cond = self.bnd_cond).adjoint(x.as_array()[0])
        tmp[1] = FiniteDiff(self.gm_domain[1:], direction = 0, bnd_cond = self.bnd_cond).adjoint(x.as_array()[1])
        tmp[2] = 0.5 * (FiniteDiff(self.gm_domain[1:], direction = 0, bnd_cond = self.bnd_cond).adjoint(x.as_array()[0]) +
                        FiniteDiff(self.gm_domain[1:], direction = 1, bnd_cond = self.bnd_cond).adjoint(x.as_array()[1]) )
        
        return type(x)(tmp)
    
    
    def adjoint(self, x, out=None):
        
        tmp = np.zeros(self.gm_domain)
        
        tmp[0] = FiniteDiff(self.gm_domain[1:], direction = 1, bnd_cond = self.bnd_cond).direct(x.as_array()[0]) +  \
                 FiniteDiff(self.gm_domain[1:], direction = 0, bnd_cond = self.bnd_cond).direct(x.as_array()[2])
                 
        tmp[1] = FiniteDiff(self.gm_domain[1:], direction = 1, bnd_cond = self.bnd_cond).direct(x.as_array()[2]) +  \
                 FiniteDiff(self.gm_domain[1:], direction = 0, bnd_cond = self.bnd_cond).direct(x.as_array()[1])                 

        return type(x)(-tmp)          
            
    def alloc_domain_dim(self):
        return ImageData(np.zeros(self.gm_domain))
    
    def alloc_range_dim(self):
        return ImageData(np.zeros(self.range_dim))
    
    def domain_dim(self):
        return self.gm_domain
    
    def range_dim(self):
        return self.gm_range
                                   
    def norm(self):
#        return np.sqrt(4*len(self.domainDim()))        
        #TODO this takes time for big ImageData
        # for 2D ||grad|| = sqrt(8), 3D ||grad|| = sqrt(12)        
        x0 = ImageData(np.random.random_sample(self.domain_dim()))
        self.s1, sall, svec = PowerMethodNonsquare(self, 25, x0)
        return self.s1    
    
 
    
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
                        
#        x_asarr = x.as_array()
        x_asarr = x
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
         
        res = out  
        return res
                    
    def adjoint(self, x, out=None):
        
#        x_asarr = x.as_array()
        x_asarr = x
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
            
        res = -out
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
    
    N, M = (2,3)
    ig = (N,M)
    G = Gradient(ig)
    u = DataContainer(np.random.randint(10, size=G.domain_dim()))
    w = DataContainer(np.random.randint(10, size=G.range_dim()))
#    w = [DataContainer(np.random.randint(10, size=G.domain_dim())),\
#         DataContainer(np.random.randint(10, size=G.domain_dim()))]

    # domain_dim
    print('Domain {}'.format(G.domain_dim()))
    
    # range_dim
    print('Range {}'.format(G.range_dim()))
    
    # Direct
    z = G.direct(u)
    
    # Adjoint
    z1 = G.adjoint(w)

    print(z)
    print(z1)
    
    LHS = (G.direct(u)*w).sum()
    RHS = (u * G.adjoint(w)).sum()
#    
    print(LHS,RHS)
    print(G.norm())
    
  
    
    
    
    
    
    
    
    

    
    