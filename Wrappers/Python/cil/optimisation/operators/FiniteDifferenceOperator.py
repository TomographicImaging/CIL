# -*- coding: utf-8 -*-
#  Copyright 2019 United Kingdom Research and Innovation
#  Copyright 2019 The University of Manchester
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Authors:
# CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt

import numpy as np

from cil.optimisation.operators import LinearOperator
from cil.utilities.errors import InPlaceError
        
class FiniteDifferenceOperator(LinearOperator):
    
    r'''                  
        Computes forward/backward/centered finite differences of a DataContainer 
        under Neumann/Periodic boundary conditions
        
        :param domain_geometry: Domain geometry for the FiniteDifferenceOperator
        :param direction: Direction to evaluate finite differences
        :type direction: string label from domain geometry or integer number
        :param method: Method for finite differences
        :type method: 'forward', 'backward', 'centered'
        :param bnd_cond: 'Neumann', 'Periodic'
        
     '''        
    
    def __init__(self, domain_geometry, 
                       range_geometry=None, 
                       direction = None, 
                       method = 'forward',
                       bnd_cond = 'Neumann'):
        
        if isinstance(direction, int):
            if direction > len(domain_geometry.shape) or direction<0:
                raise ValueError('Requested direction is not possible. Accepted direction {}, \ngot {}'.format(range(len(domain_geometry.shape)), direction))            
            else:
                self.direction = direction
        else:
           if direction in domain_geometry.dimension_labels:              
                self.direction = domain_geometry.dimension_labels.index(direction)
           else:
               raise ValueError('Requested direction is not possible. Accepted direction is {} or {}, \ngot {}'.format(domain_geometry.dimension_labels, range(len(domain_geometry.shape)),  direction))
                         
        #get voxel spacing, if not use 1s
        try:
            self.voxel_size = domain_geometry.spacing[self.direction]
        except:
            self.voxel_size = 1
        
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
        
        if id(x)==id(out):
            raise InPlaceError
        
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
            np.subtract( x_asarr[tuple(self.get_slice(2, None))], \
                             x_asarr[tuple(self.get_slice(1,-1))], \
                             out = outa[tuple(self.get_slice(1, -1))])               

            if self.boundary_condition == 'Neumann':
                
                # left boundary
                np.subtract(x_asarr[tuple(self.get_slice(1,2))],\
                            x_asarr[tuple(self.get_slice(0,1))],
                            out = outa[tuple(self.get_slice(0,1))]) 
                
                
            elif self.boundary_condition == 'Periodic':
                
                # left boundary
                np.subtract(x_asarr[tuple(self.get_slice(1,2))],\
                            x_asarr[tuple(self.get_slice(0,1))],
                            out = outa[tuple(self.get_slice(0,1))])  
                
                # right boundary
                np.subtract(x_asarr[tuple(self.get_slice(0,1))],\
                            x_asarr[tuple(self.get_slice(-1,None))],
                            out = outa[tuple(self.get_slice(-1,None))])  
                
            else:
                raise ValueError('Not implemented')                
                
        #######################################################################
        ##################### Backward differences ############################
        #######################################################################                

        elif self.method == 'backward':   
                                   
            # interior nodes
            np.subtract( x_asarr[tuple(self.get_slice(1, -1))], \
                             x_asarr[tuple(self.get_slice(0,-2))], \
                             out = outa[tuple(self.get_slice(1, -1))])              
            
            if self.boundary_condition == 'Neumann':
                    
                    # right boundary
                    np.subtract( x_asarr[tuple(self.get_slice(-1, None))], \
                                 x_asarr[tuple(self.get_slice(-2,-1))], \
                                 out = outa[tuple(self.get_slice(-1, None))]) 
                    
            elif self.boundary_condition == 'Periodic':
                  
                # left boundary
                np.subtract(x_asarr[tuple(self.get_slice(0,1))],\
                            x_asarr[tuple(self.get_slice(-1,None))],
                            out = outa[tuple(self.get_slice(0,1))])  
                
                # right boundary
                np.subtract(x_asarr[tuple(self.get_slice(-1,None))],\
                            x_asarr[tuple(self.get_slice(-2,-1))],
                            out = outa[tuple(self.get_slice(-1,None))]) 
                
            else:
                raise ValueError('Not implemented')                 
        
        #######################################################################
        ##################### Centered differences ############################
        #######################################################################
        
        
        elif self.method == 'centered':
            
            # interior nodes
            np.subtract( x_asarr[tuple(self.get_slice(2, None))], \
                             x_asarr[tuple(self.get_slice(0,-2))], \
                             out = outa[tuple(self.get_slice(1, -1))]) 
            
            outa[tuple(self.get_slice(1, -1))] /= 2.
            
            if self.boundary_condition == 'Neumann':
                            
                # left boundary
                np.subtract( x_asarr[tuple(self.get_slice(1, 2))], \
                                 x_asarr[tuple(self.get_slice(0,1))], \
                                 out = outa[tuple(self.get_slice(0, 1))])  
                outa[tuple(self.get_slice(0, 1))] /=2.
                
                # left boundary
                np.subtract( x_asarr[tuple(self.get_slice(-1, None))], \
                                 x_asarr[tuple(self.get_slice(-2,-1))], \
                                 out = outa[tuple(self.get_slice(-1, None))])
                outa[tuple(self.get_slice(-1, None))] /=2.                
                
            elif self.boundary_condition == 'Periodic':
                pass
                
               # left boundary
                np.subtract( x_asarr[tuple(self.get_slice(1, 2))], \
                                 x_asarr[tuple(self.get_slice(-1,None))], \
                                 out = outa[tuple(self.get_slice(0, 1))])                  
                outa[tuple(self.get_slice(0, 1))] /= 2.
                
                
                # left boundary
                np.subtract( x_asarr[tuple(self.get_slice(0, 1))], \
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
        else:
            out.fill(outa)                             
                 
        
    def adjoint(self, x, out=None):
        
        if id(x)==id(out):
            raise InPlaceError
        
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
            np.subtract( x_asarr[tuple(self.get_slice(1, -1))], \
                             x_asarr[tuple(self.get_slice(0,-2))], \
                             out = outa[tuple(self.get_slice(1, -1))])              
            
            if self.boundary_condition == 'Neumann':            

                # left boundary
                outa[tuple(self.get_slice(0,1))] = x_asarr[tuple(self.get_slice(0,1))]                
                
                # right boundary
                outa[tuple(self.get_slice(-1,None))] = - x_asarr[tuple(self.get_slice(-2,-1))]  
                
            elif self.boundary_condition == 'Periodic':            

                # left boundary
                np.subtract(x_asarr[tuple(self.get_slice(0,1))],\
                            x_asarr[tuple(self.get_slice(-1,None))],
                            out = outa[tuple(self.get_slice(0,1))])  
                # right boundary
                np.subtract(x_asarr[tuple(self.get_slice(-1,None))],\
                            x_asarr[tuple(self.get_slice(-2,-1))],
                            out = outa[tuple(self.get_slice(-1,None))]) 
                
            else:
                raise ValueError('Not implemented')                 

        #######################################################################
        ##################### Backward differences ############################
        #######################################################################                
                
        elif self.method == 'backward': 
            
            # interior nodes
            np.subtract( x_asarr[tuple(self.get_slice(2, None))], \
                             x_asarr[tuple(self.get_slice(1,-1))], \
                             out = outa[tuple(self.get_slice(1, -1))])             
            
            if self.boundary_condition == 'Neumann':             
                
                # left boundary
                outa[tuple(self.get_slice(0,1))] = x_asarr[tuple(self.get_slice(1,2))]                
                
                # right boundary
                outa[tuple(self.get_slice(-1,None))] = - x_asarr[tuple(self.get_slice(-1,None))] 
                
                
            elif self.boundary_condition == 'Periodic':
            
                # left boundary
                np.subtract(x_asarr[tuple(self.get_slice(1,2))],\
                            x_asarr[tuple(self.get_slice(0,1))],
                            out = outa[tuple(self.get_slice(0,1))])  
                
                # right boundary
                np.subtract(x_asarr[tuple(self.get_slice(0,1))],\
                            x_asarr[tuple(self.get_slice(-1,None))],
                            out = outa[tuple(self.get_slice(-1,None))])              
                            
            else:
                raise ValueError('Not implemented')
                
                
        #######################################################################
        ##################### Centered differences ############################
        #######################################################################

        elif self.method == 'centered':
            
            # interior nodes
            np.subtract( x_asarr[tuple(self.get_slice(2, None))], \
                             x_asarr[tuple(self.get_slice(0,-2))], \
                             out = outa[tuple(self.get_slice(1, -1))]) 
            outa[tuple(self.get_slice(1, -1))] /= 2.0
            

            if self.boundary_condition == 'Neumann':
                
                # left boundary
                np.add(x_asarr[tuple(self.get_slice(0,1))],\
                            x_asarr[tuple(self.get_slice(1,2))],
                            out = outa[tuple(self.get_slice(0,1))])
                outa[tuple(self.get_slice(0,1))] /= 2.0

                # right boundary
                np.add(x_asarr[tuple(self.get_slice(-1,None))],\
                            x_asarr[tuple(self.get_slice(-2,-1))],
                            out = outa[tuple(self.get_slice(-1,None))])  

                outa[tuple(self.get_slice(-1,None))] /= -2.0               
                                                            
                
            elif self.boundary_condition == 'Periodic':
                
                # left boundary
                np.subtract(x_asarr[tuple(self.get_slice(1,2))],\
                            x_asarr[tuple(self.get_slice(-1,None))],
                            out = outa[tuple(self.get_slice(0,1))])
                outa[tuple(self.get_slice(0,1))] /= 2.0
                
                # right boundary
                np.subtract(x_asarr[tuple(self.get_slice(0,1))],\
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
        else:
            out.fill(outa)        
              
        
    
    
