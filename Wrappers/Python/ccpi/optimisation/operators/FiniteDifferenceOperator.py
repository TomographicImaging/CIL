#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 22:51:17 2019

@author: evangelos
"""

from ccpi.optimisation.operators import Operator
from ccpi.optimisation.ops import PowerMethodNonsquare
from ccpi.framework import ImageData, BlockDataContainer
import numpy as np

class FiniteDiff(Operator):
    
    # Works for Neum/Symmetric &  periodic boundary conditions
    # TODO add central differences???
    # TODO not very well optimised, too many conditions
    # TODO add discretisation step, should get that from imageGeometry
    
    # Grad_order = ['channels', 'direction_z', 'direction_y', 'direction_x']
    # Grad_order = ['channels', 'direction_y', 'direction_x']
    # Grad_order = ['direction_z', 'direction_y', 'direction_x']
    # Grad_order = ['channels', 'direction_z', 'direction_y', 'direction_x']
    
    def __init__(self, gm_domain, gm_range=None, direction=0, bnd_cond = 'Neumann', **kwargs):
        ''''''
        super(FiniteDiff, self).__init__() 

        self.gm_domain = gm_domain
        self.gm_range = gm_range
        
        self.direction = direction
        self.bnd_cond = bnd_cond
                
        ''' Domain Geometry = Range Geometry'''
        if self.gm_range is None:
            self.gm_range = self.gm_domain
        
        ''' check direction and "length" of geometry  '''
        if self.direction + 1 > len(self.gm_domain):
            raise ValueError('Gradient directions more than geometry length') 

        self.voxel_size = kwargs.get('voxel_size',1)            
                                                 
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
         
        res = out/self.voxel_size 
        return type(x)(res)
                    
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
            
        res = out/self.voxel_size 
        return type(x)(-res)
            
    def range_geometry(self):
        ''' Returns the range geometry '''
        return self.gm_range
    
    def domain_geometry(self):
        ''' Returns the domain geometry '''
        return self.gm_domain
       
    def norm(self):     
        ''' Computes the operator norm using PowerMethod
            Can we avoid it since it is always 2??? '''            
        ''' Need ImageData to allocate random array '''           
        x0 = self.gm_domain.allocate('random')
#        x = ImageData(np.random.random_sample(x0.shape))
        self.s1, sall, svec = PowerMethodNonsquare(self, 50, x0)
        return self.s1
    
#    def matrix(self):
#        
#        '''' 
#        Create Sparse matrix for direction and boundaries
#        ''''
#        pass
#    
#    def matrix_row_abs_sum(self):
#        pass
#    
#    def matrix_row_abs_sum(self):
#        pass
    
    
if __name__ == '__main__':
    
    N, M = 200, 300
    
    from ccpi.framework import ImageGeometry
    
    ig = ImageGeometry(voxel_num_x = M, voxel_num_y = N)    
    u = ig.allocate('random_int')
    G = FiniteDiff(ig, direction=0, bnd_cond = 'Neumann')
    print(u.as_array())    
    print(G.direct(u).as_array())
    
    # Gradient Operator norm, for one direction should be close to 2
    np.testing.assert_allclose(G.norm(), np.sqrt(4), atol=0.1)
        
    M1, N1, K1 = 200, 300, 2
    ig1 = ImageGeometry(voxel_num_x = M1, voxel_num_y = N1, channels = K1)
    u1 = ig1.allocate('random_int')
    G1 = FiniteDiff(ig1, direction=2, bnd_cond = 'Periodic')
    print(ig1.shape==u1.shape)
    np.testing.assert_allclose(G1.norm(), np.sqrt(4), atol=0.1)


    

    
    