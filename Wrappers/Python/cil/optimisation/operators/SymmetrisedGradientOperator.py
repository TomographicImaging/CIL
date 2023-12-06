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

from cil.optimisation.operators import LinearOperator
from cil.framework import BlockGeometry, BlockDataContainer
from cil.optimisation.operators import FiniteDifferenceOperator


class SymmetrisedGradientOperator(LinearOperator):
    
    r''' Consider the operator :math:`E: V \rightarrow W` where `V` is `BlockGeometry` and  `W` is the range of the Symmetrized Gradient and 
        
        .. math::
        
            E(v) = 0.5  ( \nabla v + (\nabla v)^{T} ) \\
                        
        
    In 2 dimensions, let :math:`v(x,y)=(v_1(x,y),v_2(x,y))` which gives
        
        .. math::
        
            \nabla v =\left( \begin{matrix} 
                \partial_{x} v_1 & \partial_x v_2\\
                \partial_{y}v_1 & \partial_y v_2
            \end{matrix}\right)
        
    and thus 
        
        .. math::
        
            E(v) = 0.5  ( \nabla\cdot v + (\nabla\cdot v)^{T} )       
            =\left( \begin{matrix} 
                \partial_{x} v_1 & 0.5 * (\partial_{y} v_1 + \partial_{x} v_2) \\
                0.5 * (\partial_{x} v_1 + \partial_{y} v_2) & \partial_{y} v_2 
            \end{matrix}\right)
                          
    Parameters
    ----------
    domain_geometry: `BlockGeometry` with shape (2,1) or (3,1)
        Set up the domain of the function. 
    bnd_cond: str, optional, default :code:`Neumann`
        Boundary condition either :code:`Neumann` or :code:`Periodic`
    correlation: str, optional, default :code:`Channel`
        Correlation either :code:`SpaceChannel` or :code:`Channel`
        
    '''
    
    CORRELATION_SPACE = "Space"
    CORRELATION_SPACECHANNEL = "SpaceChannels"
    
    def __init__(self, domain_geometry, bnd_cond = 'Neumann', **kwargs):

        self.bnd_cond = bnd_cond
        self.correlation = kwargs.get('correlation',SymmetrisedGradientOperator.CORRELATION_SPACE)
                
        tmp_gm = len(domain_geometry.geometries)*domain_geometry.geometries
        
        
        # Define FD operator. We need one geometry from the BlockGeometry of the domain
        self.FD = FiniteDifferenceOperator(domain_geometry.get_item(0), direction = 0, 
                             bnd_cond = self.bnd_cond)
        
        if domain_geometry.shape[0]==2:
            self.order_ind = [0,2,1,3]
        else:
            self.order_ind = [0,3,6,1,4,7,2,5,8]            
        
        super(SymmetrisedGradientOperator, self).__init__(
                                          domain_geometry=domain_geometry, 
                                          range_geometry=BlockGeometry(*tmp_gm))
        
        
    def direct(self, x, out=None):
        
        f'''Returns :math:`E(v) = 0.5 * ( \nabla v + (\nabla v)^{T} )` '''        
        
        if out is None:
            
            tmp = []
            for i in range(self.domain_geometry().shape[0]):
                for j in range(x.shape[0]):
                    self.FD.direction = i
                    tmp.append(self.FD.adjoint(x.get_item(j)))
                    
            tmp1 = [tmp[i] for i in self.order_ind]        
                    
            res = [0.5 * sum(x) for x in zip(tmp, tmp1)]   
                    
            return BlockDataContainer(*res) 
    
        else:
            
            ind = 0
            for i in range(self.domain_geometry().shape[0]):
                for j in range(x.shape[0]):
                    self.FD.direction = i
                    self.FD.adjoint(x.get_item(j), out=out[ind])
                    ind+=1                    
            out1 = BlockDataContainer(*[out[i] for i in self.order_ind])          
            out.fill( 0.5 * (out + out1) )
            
                                               
    def adjoint(self, x, out=None):
        '''Returns the adjoint of the symmetrised gradient operator'''        
       
        if out is None:
            
            tmp = [None]*self.domain_geometry().shape[0]
            i = 0
            
            for k in range(self.domain_geometry().shape[0]):
                tmp1 = 0
                for j in range(self.domain_geometry().shape[0]):
                    self.FD.direction = j
                    tmp1 += self.FD.direct(x[i])                    
                    i+=1
                tmp[k] = tmp1  
            return BlockDataContainer(*tmp)
            

        else:
            
            tmp = self.domain_geometry().allocate() 
            i = 0
            for k in range(self.domain_geometry().shape[0]):
                tmp1 = 0
                for j in range(self.domain_geometry().shape[0]):
                    self.FD.direction = j
                    self.FD.direct(x[i], out=tmp[j])
                    i+=1
                    tmp1+=tmp[j]
                out[k].fill(tmp1)
                    
    