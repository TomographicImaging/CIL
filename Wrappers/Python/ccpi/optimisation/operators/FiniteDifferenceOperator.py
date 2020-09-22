# -*- coding: utf-8 -*-
#  CCP in Tomographic Imaging (CCPi) Core Imaging Library (CIL).

#   Copyright 2017 UKRI-STFC
#   Copyright 2017 University of Manchester

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as xp

from ccpi.optimisation.operators import LinearOperator

###############################################################################
###############################################################################
###############################################################################
############################# New Finite Difference ###########################
###############################################################################
###############################################################################
###############################################################################            
            
        
class FiniteDifferenceOperator(LinearOperator):
    
    def __init__(self, domain_geometry, 
                       range_geometry=None, 
                       direction=0, 
                       method = 'forward',
                       bnd_cond = 'Neumann'):
        
        self.direction = direction
        self.voxel_size = domain_geometry.spacing[self.direction]
        self.boundary_condition = bnd_cond
        self.method = method
                
        # Domain Geometry = Range Geometry if not stated
        if range_geometry is None:
            range_geometry = domain_geometry 
            
        super(FiniteDifferenceOperator, self).__init__(domain_geometry = domain_geometry, 
                                         range_geometry = range_geometry)              
            
        self.size_dom_gm = len(domain_geometry.shape) 
        
        if self.voxel_size <= 0:
            raise ValueError(' Need a positive voxel size ')                      
                    
        # check direction and "length" of geometry
        if self.direction + 1 > self.size_dom_gm:
            raise ValueError('Finite differences direction {} larger than geometry shape length {}'.format(self.direction + 1, self.size_dom_gm))          
                                                 
    def get_slice(self, start, stop, end=None):
        
        tmp = [slice(None)]*self.size_dom_gm
        tmp[self.direction] = slice(start, stop, end)
        return tmp       

    def direct(self, x, out = None):
        
        x_asarr = x.as_array()
        
        outnone = False
        if out is None:
            outnone = True
            ret = self.domain_geometry().allocate()
            outa = ret.as_array()
        else:
            outa = out.as_array()
            outa[:]=0     

        #######################################################################
        ##################### Forward differences #############################
        #######################################################################
                
        if self.method == 'forward':  
            
            # interior nodes
            xp.subtract( x_asarr[tuple(self.get_slice(2, None))], \
                             x_asarr[tuple(self.get_slice(1,-1))], \
                             out = outa[tuple(self.get_slice(1, -1))])               

            if self.boundary_condition == 'Neumann':
                
                # left boundary
                xp.subtract(x_asarr[tuple(self.get_slice(1,2))],\
                            x_asarr[tuple(self.get_slice(0,1))],
                            out = outa[tuple(self.get_slice(0,1))]) 
                
                
            elif self.boundary_condition == 'Periodic':
                
                # left boundary
                xp.subtract(x_asarr[tuple(self.get_slice(1,2))],\
                            x_asarr[tuple(self.get_slice(0,1))],
                            out = outa[tuple(self.get_slice(0,1))])  
                
                # right boundary
                xp.subtract(x_asarr[tuple(self.get_slice(0,1))],\
                            x_asarr[tuple(self.get_slice(-1,None))],
                            out = outa[tuple(self.get_slice(-1,None))])  
                
            else:
                raise ValueError('Not implemented')                
                
        #######################################################################
        ##################### Backward differences ############################
        #######################################################################                

        elif self.method == 'backward':   
                                   
            # interior nodes
            xp.subtract( x_asarr[tuple(self.get_slice(1, -1))], \
                             x_asarr[tuple(self.get_slice(0,-2))], \
                             out = outa[tuple(self.get_slice(1, -1))])              
            
            if self.boundary_condition == 'Neumann':
                    
                    # right boundary
                    xp.subtract( x_asarr[tuple(self.get_slice(-1, None))], \
                                 x_asarr[tuple(self.get_slice(-2,-1))], \
                                 out = outa[tuple(self.get_slice(-1, None))]) 
                    
            elif self.boundary_condition == 'Periodic':
                  
                # left boundary
                xp.subtract(x_asarr[tuple(self.get_slice(0,1))],\
                            x_asarr[tuple(self.get_slice(-1,None))],
                            out = outa[tuple(self.get_slice(0,1))])  
                
                # right boundary
                xp.subtract(x_asarr[tuple(self.get_slice(-1,None))],\
                            x_asarr[tuple(self.get_slice(-2,-1))],
                            out = outa[tuple(self.get_slice(-1,None))]) 
                
            else:
                raise ValueError('Not implemented')                 
        
        #######################################################################
        ##################### Centered differences ############################
        #######################################################################
        
        
        elif self.method == 'centered':
            
            # interior nodes
            xp.subtract( x_asarr[tuple(self.get_slice(2, None))], \
                             x_asarr[tuple(self.get_slice(0,-2))], \
                             out = outa[tuple(self.get_slice(1, -1))]) 
            
            outa[tuple(self.get_slice(1, -1))] /= 2.
            
            if self.boundary_condition == 'Neumann':
            #                
#                # left boundary
                xp.subtract( x_asarr[tuple(self.get_slice(1, 2))], \
                                 x_asarr[tuple(self.get_slice(0,1))], \
                                 out = outa[tuple(self.get_slice(0, 1))])  
                outa[tuple(self.get_slice(0, 1))] /=2.
#                
#                # left boundary
                xp.subtract( x_asarr[tuple(self.get_slice(-1, None))], \
                                 x_asarr[tuple(self.get_slice(-2,-1))], \
                                 out = outa[tuple(self.get_slice(-1, None))])
                outa[tuple(self.get_slice(-1, None))] /=2.                
#                
            elif self.boundary_condition == 'Periodic':
                pass
#                
               # left boundary
                xp.subtract( x_asarr[tuple(self.get_slice(1, 2))], \
                                 x_asarr[tuple(self.get_slice(-1,None))], \
                                 out = outa[tuple(self.get_slice(0, 1))])                  
                outa[tuple(self.get_slice(0, 1))] /= 2.
                
                
                # left boundary
                xp.subtract( x_asarr[tuple(self.get_slice(0, 1))], \
                                 x_asarr[tuple(self.get_slice(-2,-1))], \
                                 out = outa[tuple(self.get_slice(-1, None))]) 
                outa[tuple(self.get_slice(-1, None))] /= 2.

            else:
                raise ValueError('Not implemented')                 
                
        else:
                raise ValueError('Not implemented')                
        
        if self.voxel_size != 1.0:
            outa /= self.voxel_size  

        if outnone:                  
            ret.fill(outa)
            return ret                
                 
        
    def adjoint(self, x, out=None):
        
        # Adjoint operation defined as  
                      
        x_asarr = x.as_array()

        outnone = False 
        if out is None:
            outnone = True
            ret = self.range_geometry().allocate()
            outa = ret.as_array()
        else:
            outa = out.as_array()        
            outa[:]=0 
            
            
        #######################################################################
        ##################### Forward differences #############################
        #######################################################################            
            

        if self.method == 'forward':    
            
            # interior nodes
            xp.subtract( x_asarr[tuple(self.get_slice(1, -1))], \
                             x_asarr[tuple(self.get_slice(0,-2))], \
                             out = outa[tuple(self.get_slice(1, -1))])              
            
            if self.boundary_condition == 'Neumann':            

                # left boundary
                outa[tuple(self.get_slice(0,1))] = x_asarr[tuple(self.get_slice(0,1))]                
                
                # right boundary
                outa[tuple(self.get_slice(-1,None))] = - x_asarr[tuple(self.get_slice(-2,-1))]  
                
            elif self.boundary_condition == 'Periodic':            

                # left boundary
                xp.subtract(x_asarr[tuple(self.get_slice(0,1))],\
                            x_asarr[tuple(self.get_slice(-1,None))],
                            out = outa[tuple(self.get_slice(0,1))])  
                # right boundary
                xp.subtract(x_asarr[tuple(self.get_slice(-1,None))],\
                            x_asarr[tuple(self.get_slice(-2,-1))],
                            out = outa[tuple(self.get_slice(-1,None))]) 
                
            else:
                raise ValueError('Not implemented')                 

        #######################################################################
        ##################### Backward differences ############################
        #######################################################################                
                
        elif self.method == 'backward': 
            
            # interior nodes
            xp.subtract( x_asarr[tuple(self.get_slice(2, None))], \
                             x_asarr[tuple(self.get_slice(1,-1))], \
                             out = outa[tuple(self.get_slice(1, -1))])             
            
            if self.boundary_condition == 'Neumann':             
                
                # left boundary
                outa[tuple(self.get_slice(0,1))] = x_asarr[tuple(self.get_slice(1,2))]                
                
                # right boundary
                outa[tuple(self.get_slice(-1,None))] = - x_asarr[tuple(self.get_slice(-1,None))] 
                
                
            elif self.boundary_condition == 'Periodic':
            
                # left boundary
                xp.subtract(x_asarr[tuple(self.get_slice(1,2))],\
                            x_asarr[tuple(self.get_slice(0,1))],
                            out = outa[tuple(self.get_slice(0,1))])  
                
                # right boundary
                xp.subtract(x_asarr[tuple(self.get_slice(0,1))],\
                            x_asarr[tuple(self.get_slice(-1,None))],
                            out = outa[tuple(self.get_slice(-1,None))])              
                            
            else:
                raise ValueError('Not implemented')
                
                
        #######################################################################
        ##################### Centered differences ############################
        #######################################################################

        elif self.method == 'centered':
            
            # interior nodes
            xp.subtract( x_asarr[tuple(self.get_slice(2, None))], \
                             x_asarr[tuple(self.get_slice(0,-2))], \
                             out = outa[tuple(self.get_slice(1, -1))]) 
            outa[tuple(self.get_slice(1, -1))] /= 2.0
            

            if self.boundary_condition == 'Neumann':
                
                # left boundary
                xp.add(x_asarr[tuple(self.get_slice(0,1))],\
                            x_asarr[tuple(self.get_slice(1,2))],
                            out = outa[tuple(self.get_slice(0,1))])
                outa[tuple(self.get_slice(0,1))] /= 2.0

                # right boundary
                xp.add(x_asarr[tuple(self.get_slice(-1,None))],\
                            x_asarr[tuple(self.get_slice(-2,-1))],
                            out = outa[tuple(self.get_slice(-1,None))])  

                outa[tuple(self.get_slice(-1,None))] /= -2.0               
                                                            
                
            elif self.boundary_condition == 'Periodic':
                
                # left boundary
                xp.subtract(x_asarr[tuple(self.get_slice(1,2))],\
                            x_asarr[tuple(self.get_slice(-1,None))],
                            out = outa[tuple(self.get_slice(0,1))])
                outa[tuple(self.get_slice(0,1))] /= 2.0
                
                # right boundary
                xp.subtract(x_asarr[tuple(self.get_slice(0,1))],\
                            x_asarr[tuple(self.get_slice(-2,-1))],
                            out = outa[tuple(self.get_slice(-1,None))])
                outa[tuple(self.get_slice(-1,None))] /= 2.0
                
                                
            else:
                raise ValueError('Not implemented') 
                                             
        else:
                raise ValueError('Not implemented')                  
                               
        outa *= -1.
        if self.voxel_size != 1.0:
            outa /= self.voxel_size                      
            
        if outnone:                  
            ret.fill(outa)
            return ret       


class OldFiniteDiff(LinearOperator):
    
    '''Finite Difference Operator:
            
        Computes first-order forward/backward differences 
                    on 2D, 3D, 4D ImageData
                    under Neumann/Periodic boundary conditions

        Order of the Gradient ( ImageGeometry may contain channels ):

        .. code:: python          

            Grad_order = ['channels', 'direction_z', 'direction_y', 'direction_x']
            Grad_order = ['channels', 'direction_y', 'direction_x']
            Grad_order = ['direction_z', 'direction_y', 'direction_x']
            Grad_order = ['channels', 'direction_z', 'direction_y', 'direction_x']  
                                        
    
    '''

        
    def __init__(self, gm_domain, gm_range=None, direction=0, bnd_cond = 'Neumann'):
        '''creator

        :param gm_domain: domain of the operator
        :type gm_domain: :code:`AcquisitionGeometry` or :code:`ImageGeometry`
        :param gm_range: optional range of the operator
        :type gm_range: :code:`AcquisitionGeometry` or :code:`ImageGeometry`, optional
        :param direction: optional axis in the input :code:`DataContainer` along which to calculate the finite differences, default 0
        :type direction: int, optional, default 0
        :param bnd_cond: boundary condition, either :code:`Neumann` or :code:`Periodic`.
        :type bnd_cond: str, default :code:`Neumann`
        
        '''
        
        self.gm_domain = gm_domain
        self.gm_range = gm_range
        
        self.direction = direction
        self.bnd_cond = bnd_cond
        
        # Domain Geometry = Range Geometry if not stated
        if self.gm_range is None:
            self.gm_range = self.gm_domain
            
        # check direction and "length" of geometry
        if self.direction + 1 > len(self.gm_domain.shape):
            raise ValueError('Gradient directions more than geometry domain')      
        
        #self.voxel_size = kwargs.get('voxel_size',1)
        # this wrongly assumes a homogeneous voxel size
#        self.voxel_size = self.gm_domain.voxel_size_x
        super(OldFiniteDiff, self).__init__(domain_geometry=gm_domain, 
                                         range_geometry=self.gm_range) 


    def direct(self, x, out=None):
        
        x_asarr = x.as_array()
        x_sz = len(x.shape)
        outnone = False
        if out is None:
            outnone = True
            ret = self.domain_geometry().allocate()
            outa = ret.as_array()
        else:
            outa = out.as_array()
            outa[:]=0
                  
        ######################## Direct for 2D  ###############################
        if x_sz == 2:
            
            if self.direction == 1:
                
                xp.subtract( x_asarr[:,1:], x_asarr[:,0:-1], out = outa[:,0:-1] )
                
                if self.bnd_cond == 'Neumann':
                    pass
                elif self.bnd_cond == 'Periodic':
                    xp.subtract( x_asarr[:,0], x_asarr[:,-1], out = outa[:,-1] )
                else: 
                    raise ValueError('No valid boundary conditions')
                
            if self.direction == 0:
                
                xp.subtract( x_asarr[1:], x_asarr[0:-1], out = outa[0:-1,:] )

                if self.bnd_cond == 'Neumann':
                    pass                                        
                elif self.bnd_cond == 'Periodic':
                    xp.subtract( x_asarr[0,:], x_asarr[-1,:], out = outa[-1,:] ) 
                else:    
                    raise ValueError('No valid boundary conditions') 
                    
        ######################## Direct for 3D  ###############################                        
        elif x_sz == 3:
                    
            if self.direction == 0:  
                
                xp.subtract( x_asarr[1:,:,:], x_asarr[0:-1,:,:], out = outa[0:-1,:,:] )
                
                if self.bnd_cond == 'Neumann':
                    pass
                elif self.bnd_cond == 'Periodic':
                    xp.subtract( x_asarr[0,:,:], x_asarr[-1,:,:], out = outa[-1,:,:] ) 
                else:    
                    raise ValueError('No valid boundary conditions')                      
                                                             
            if self.direction == 1:
                
                xp.subtract( x_asarr[:,1:,:], x_asarr[:,0:-1,:], out = outa[:,0:-1,:] ) 
                
                if self.bnd_cond == 'Neumann':
                    pass
                elif self.bnd_cond == 'Periodic':
                    xp.subtract( x_asarr[:,0,:], x_asarr[:,-1,:], out = outa[:,-1,:] )
                else:    
                    raise ValueError('No valid boundary conditions')                      
                                
             
            if self.direction == 2:
                
                xp.subtract( x_asarr[:,:,1:], x_asarr[:,:,0:-1], out = outa[:,:,0:-1] ) 
                
                if self.bnd_cond == 'Neumann':
                    pass
                elif self.bnd_cond == 'Periodic':
                    xp.subtract( x_asarr[:,:,0], x_asarr[:,:,-1], out = outa[:,:,-1] )
                else:    
                    raise ValueError('No valid boundary conditions')  
                    
        ######################## Direct for 4D  ###############################
        elif x_sz == 4:
                    
            if self.direction == 0:                            
                xp.subtract( x_asarr[1:,:,:,:], x_asarr[0:-1,:,:,:], out = outa[0:-1,:,:,:] )
                
                if self.bnd_cond == 'Neumann':
                    pass
                elif self.bnd_cond == 'Periodic':
                    xp.subtract( x_asarr[0,:,:,:], x_asarr[-1,:,:,:], out = outa[-1,:,:,:] )
                else:    
                    raise ValueError('No valid boundary conditions') 
                                                
            if self.direction == 1:
                xp.subtract( x_asarr[:,1:,:,:], x_asarr[:,0:-1,:,:], out = outa[:,0:-1,:,:] ) 
                
                if self.bnd_cond == 'Neumann':
                    pass
                elif self.bnd_cond == 'Periodic':
                    xp.subtract( x_asarr[:,0,:,:], x_asarr[:,-1,:,:], out = outa[:,-1,:,:] )
                else:    
                    raise ValueError('No valid boundary conditions')                 
                
            if self.direction == 2:
                xp.subtract( x_asarr[:,:,1:,:], x_asarr[:,:,0:-1,:], out = outa[:,:,0:-1,:] ) 
                
                if self.bnd_cond == 'Neumann':
                    pass
                elif self.bnd_cond == 'Periodic':
                    xp.subtract( x_asarr[:,:,0,:], x_asarr[:,:,-1,:], out = outa[:,:,-1,:] )
                else:    
                    raise ValueError('No valid boundary conditions')                   
                
            if self.direction == 3:
                xp.subtract( x_asarr[:,:,:,1:], x_asarr[:,:,:,0:-1], out = outa[:,:,:,0:-1] )                 

                if self.bnd_cond == 'Neumann':
                    pass
                elif self.bnd_cond == 'Periodic':
                    xp.subtract( x_asarr[:,:,:,0], x_asarr[:,:,:,-1], out = outa[:,:,:,-1] )
                else:    
                    raise ValueError('No valid boundary conditions')                   
                                
        else:
            raise NotImplementedError                
         
#        res = out #/self.voxel_size 
        #return type(x)(out)
        if outnone:
            ret.fill(outa)
            return ret

                    
    def adjoint(self, x, out=None):
                      
        x_asarr = x.as_array()
        x_sz = len(x.shape)
        outnone = False 
        if out is None:
            outnone = True
            ret = self.range_geometry().allocate()
            outa = ret.as_array()
            #out = np.zeros_like(x_asarr)
        else:
            outa = out.as_array()        
            outa[:]=0

        
        ######################## Adjoint for 2D  ###############################
        if x_sz == 2:        
        
            if self.direction == 1:
                
                xp.subtract( x_asarr[:,1:], x_asarr[:,0:-1], out = outa[:,1:] )
                
                if self.bnd_cond == 'Neumann':
                    xp.subtract( x_asarr[:,0], 0, out = outa[:,0] )
                    xp.subtract( -x_asarr[:,-2], 0, out = outa[:,-1] )
                    
                elif self.bnd_cond == 'Periodic':
                    xp.subtract( x_asarr[:,0], x_asarr[:,-1], out = outa[:,0] )                                        
                    
                else:   
                    raise ValueError('No valid boundary conditions') 
                                    
            if self.direction == 0:
                
                xp.subtract( x_asarr[1:,:], x_asarr[0:-1,:], out = outa[1:,:] )
                
                if self.bnd_cond == 'Neumann':
                    xp.subtract( x_asarr[0,:], 0, out = outa[0,:] )
                    xp.subtract( -x_asarr[-2,:], 0, out = outa[-1,:] ) 
                    
                elif self.bnd_cond == 'Periodic':  
                    xp.subtract( x_asarr[0,:], x_asarr[-1,:], out = outa[0,:] ) 
                    
                else:   
                    raise ValueError('No valid boundary conditions')     
        
        ######################## Adjoint for 3D  ###############################        
        elif x_sz == 3:                
                
            if self.direction == 0:          
                  
                xp.subtract( x_asarr[1:,:,:], x_asarr[0:-1,:,:], out = outa[1:,:,:] )
                
                if self.bnd_cond == 'Neumann':
                    xp.subtract( x_asarr[0,:,:], 0, out = outa[0,:,:] )
                    xp.subtract( -x_asarr[-2,:,:], 0, out = outa[-1,:,:] )
                elif self.bnd_cond == 'Periodic':                     
                    xp.subtract( x_asarr[0,:,:], x_asarr[-1,:,:], out = outa[0,:,:] )
                else:   
                    raise ValueError('No valid boundary conditions')                     
                                    
            if self.direction == 1:
                xp.subtract( x_asarr[:,1:,:], x_asarr[:,0:-1,:], out = outa[:,1:,:] )
                
                if self.bnd_cond == 'Neumann':
                    xp.subtract( x_asarr[:,0,:], 0, out = outa[:,0,:] )
                    xp.subtract( -x_asarr[:,-2,:], 0, out = outa[:,-1,:] )
                elif self.bnd_cond == 'Periodic':                     
                    xp.subtract( x_asarr[:,0,:], x_asarr[:,-1,:], out = outa[:,0,:] )
                else:   
                    raise ValueError('No valid boundary conditions')                                 
                
            if self.direction == 2:
                xp.subtract( x_asarr[:,:,1:], x_asarr[:,:,0:-1], out = outa[:,:,1:] )
                
                if self.bnd_cond == 'Neumann':
                    xp.subtract( x_asarr[:,:,0], 0, out = outa[:,:,0] ) 
                    xp.subtract( -x_asarr[:,:,-2], 0, out = outa[:,:,-1] ) 
                elif self.bnd_cond == 'Periodic':                     
                    xp.subtract( x_asarr[:,:,0], x_asarr[:,:,-1], out = outa[:,:,0] )
                else:   
                    raise ValueError('No valid boundary conditions')                                 
        
        ######################## Adjoint for 4D  ###############################        
        elif x_sz == 4:                
                
            if self.direction == 0:                            
                xp.subtract( x_asarr[1:,:,:,:], x_asarr[0:-1,:,:,:], out = outa[1:,:,:,:] )
                
                if self.bnd_cond == 'Neumann':
                    xp.subtract( x_asarr[0,:,:,:], 0, out = outa[0,:,:,:] )
                    xp.subtract( -x_asarr[-2,:,:,:], 0, out = outa[-1,:,:,:] )
                    
                elif self.bnd_cond == 'Periodic':
                    xp.subtract( x_asarr[0,:,:,:], x_asarr[-1,:,:,:], out = outa[0,:,:,:] )
                else:    
                    raise ValueError('No valid boundary conditions') 
                                
            if self.direction == 1:
                xp.subtract( x_asarr[:,1:,:,:], x_asarr[:,0:-1,:,:], out = outa[:,1:,:,:] )
                
                if self.bnd_cond == 'Neumann':
                   xp.subtract( x_asarr[:,0,:,:], 0, out = outa[:,0,:,:] )
                   xp.subtract( -x_asarr[:,-2,:,:], 0, out = outa[:,-1,:,:] )
                    
                elif self.bnd_cond == 'Periodic':
                    xp.subtract( x_asarr[:,0,:,:], x_asarr[:,-1,:,:], out = outa[:,0,:,:] )
                else:    
                    raise ValueError('No valid boundary conditions') 
                    
                
            if self.direction == 2:
                xp.subtract( x_asarr[:,:,1:,:], x_asarr[:,:,0:-1,:], out = outa[:,:,1:,:] )
                
                if self.bnd_cond == 'Neumann':
                    xp.subtract( x_asarr[:,:,0,:], 0, out = outa[:,:,0,:] ) 
                    xp.subtract( -x_asarr[:,:,-2,:], 0, out = outa[:,:,-1,:] ) 
                    
                elif self.bnd_cond == 'Periodic':
                    xp.subtract( x_asarr[:,:,0,:], x_asarr[:,:,-1,:], out = outa[:,:,0,:] )
                else:    
                    raise ValueError('No valid boundary conditions')                 
                
            if self.direction == 3:
                xp.subtract( x_asarr[:,:,:,1:], x_asarr[:,:,:,0:-1], out = outa[:,:,:,1:] )
                
                if self.bnd_cond == 'Neumann':
                    xp.subtract( x_asarr[:,:,:,0], 0, out = outa[:,:,:,0] ) 
                    xp.subtract( -x_asarr[:,:,:,-2], 0, out = outa[:,:,:,-1] )   
                    
                elif self.bnd_cond == 'Periodic':
                    xp.subtract( x_asarr[:,:,:,0], x_asarr[:,:,:,-1], out = outa[:,:,:,0] )
                else:    
                    raise ValueError('No valid boundary conditions')                  
                              
        else:
            raise NotImplementedError
            
        outa *= -1 #/self.voxel_size
        if outnone:
            ret.fill(outa)
            return ret
        
        
if __name__ == '__main__':
    
    
    from ccpi.framework import ImageGeometry
    from timeit import default_timer as timer
    
    ig = ImageGeometry(3, 4, 5)
    x = ig.allocate('random_int', max_value = 10, seed = 10)

     
    methods = ['forward', 'backward', 'centered']
    bnd_cond = ['Neumann', 'Periodic']
    vsz = [0.3, 3., 0.01, 4.]
    
    for i in range(len(ig.shape)):
        for j in range(len(methods)):
            for k in range(len(bnd_cond)):
                        FD = FiniteDifferenceOperator(ig, direction=i, method = methods[j], bnd_cond = bnd_cond[k]) 
                        print("FD: Direction = {}, Method = {}, BND = {} , DOT = {}" \
                              .format(i, methods[j], bnd_cond[k], \
                                      FD.dot_test(FD)))                    
                       
#     Checking speed vs the FiniteDiff old, the new one is a bit faster                        
    ig1 = ImageGeometry(300, 300, 400)                        
    FD_old = OldFiniteDiff(ig1, direction=0, bnd_cond = 'Neumann')    
    FD = FiniteDifferenceOperator(ig1, direction=0,  bnd_cond = 'Neumann')
                           
    x = ig1.allocate('random')
    
    res1 = x*0.
    t0 = timer()
    for i in range(1):
        FD_old.direct(x, out = res1)
    t1 = timer()
    print(t1-t0)

    res2 = x*0.
    t2 = timer()
    for i in range(1):
        FD.direct(x, out = res2)
    t3 = timer()
    print(t3-t2)  
     
#    from ccpi.optimisation.operators import Gradient
#    
#    G_numpy = Gradient(ig1, backend = 'numpy')
#    G_C = Gradient(ig1, backend = 'c')
#    
#    res3 = G_numpy.range_geometry().allocate()
#    t4 = timer()
#    G_numpy.direct(x, out = res3)
#    t5 = timer()
#    print(t5-t4) 
#
#    res4 = G_C.range_geometry().allocate()
#    t6 = timer()
#    G_C.direct(x, out = res4)
#    t7 = timer()
#    print(t7-t6)    

#    ig1 = ImageGeometry(voxel_num_x = 300, voxel_num_y = 300, 
#                        voxel_size_x = 0.1, voxel_size_y = 0.1)
#    FD = FiniteDifferenceOperator(ig1, direction=0,  bnd_cond = 'Neumann')  
#
#    x = ig1.allocate('random')
#    res = FD.direct(x)
#
#    print(res.as_array())
#    print(x.as_array())
#
#    print(FD.norm())
#    print(np.sqrt(4/ig1.voxel_size_x**2))
    # print(4/(ig1.voxel_size_x**2))
    
#    
