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
from __future__ import unicode_literals


from ccpi.optimisation.operators import LinearOperator
from ccpi.framework import BlockGeometry, BlockDataContainer
from ccpi.optimisation.operators import FiniteDiff


class SymmetrizedGradient(LinearOperator):
    
    r'''Symmetrized Gradient Operator:  E: V -> W
        
            V : range of the Gradient Operator
            W : range of the Symmetrized Gradient          
        
            Example (2D): 
            .. math::
               v = (v1, v2), 
            
                           Ev = 0.5 * ( \nabla\cdot v + (\nabla\cdot c)^{T} )
                           
                           \begin{matrix} 
                               \partial_{y} v1 & 0.5 * (\partial_{x} v1 + \partial_{y} v2) \\
                               0.5 * (\partial_{x} v1 + \partial_{y} v2) & \partial_{x} v2 
                           \end{matrix}
              |                                                      
    '''
    
    CORRELATION_SPACE = "Space"
    CORRELATION_SPACECHANNEL = "SpaceChannels"
    
    def __init__(self, gm_domain, bnd_cond = 'Neumann', **kwargs):
        
        super(SymmetrizedGradient, self).__init__() 
                
        self.gm_domain = gm_domain
        self.bnd_cond = bnd_cond
        self.correlation = kwargs.get('correlation',SymmetrizedGradient.CORRELATION_SPACE)
                
        tmp_gm = len(self.gm_domain.geometries)*self.gm_domain.geometries
        
        self.gm_range = BlockGeometry(*tmp_gm)
        
        # Define FD operator. We need one geometry from the BlockGeometry of the domain
        self.FD = FiniteDiff(self.gm_domain.get_item(0), direction = 0, bnd_cond = self.bnd_cond)
        
        if self.gm_domain.shape[0]==2:
            self.order_ind = [0,2,1,3]
        else:
            self.order_ind = [0,3,6,1,4,7,2,5,8]            
                
        
    def direct(self, x, out=None):
        
        '''Returns E(v)'''        
        
        if out is None:
            
            tmp = []
            for i in range(self.gm_domain.shape[0]):
                for j in range(x.shape[0]):
                    self.FD.direction = i
                    tmp.append(self.FD.adjoint(x.get_item(j)))
                    
            tmp1 = [tmp[i] for i in self.order_ind]        
                    
            res = [0.5 * sum(x) for x in zip(tmp, tmp1)]   
                    
            return BlockDataContainer(*res) 
    
        else:
            
            ind = 0
            for i in range(self.gm_domain.shape[0]):
                for j in range(x.shape[0]):
                    self.FD.direction = i
                    self.FD.adjoint(x.get_item(j), out=out[ind])
                    ind+=1                    
            out1 = BlockDataContainer(*[out[i] for i in self.order_ind])          
            out.fill( 0.5 * (out + out1) )
            
                                               
    def adjoint(self, x, out=None):
        
        if out is None:
            
            tmp = [None]*self.gm_domain.shape[0]
            i = 0
            
            for k in range(self.gm_domain.shape[0]):
                tmp1 = 0
                for j in range(self.gm_domain.shape[0]):
                    self.FD.direction = j
                    tmp1 += self.FD.direct(x[i])                    
                    i+=1
                tmp[k] = tmp1  
            return BlockDataContainer(*tmp)
            

        else:
            
            tmp = self.gm_domain.allocate() 
            i = 0
            for k in range(self.gm_domain.shape[0]):
                tmp1 = 0
                for j in range(self.gm_domain.shape[0]):
                    self.FD.direction = j
                    self.FD.direct(x[i], out=tmp[j])
                    i+=1
                    tmp1+=tmp[j]
                out[k].fill(tmp1)
                    
                     
    def domain_geometry(self):
        return self.gm_domain
    
    def range_geometry(self):
        return self.gm_range

if __name__ == '__main__':   
    
    ###########################################################################  
    ## Symmetrized Gradient Tests
    from ccpi.framework import ImageGeometry
    from ccpi.optimisation.operators import Gradient
    import numpy as np
    
    N, M = 2, 3
    K = 2
    C = 2
    
    ###########################################################################
    # 2D geometry no channels
    ig = ImageGeometry(N, M)
    Grad = Gradient(ig)
    
    E1 = SymmetrizedGradient(Grad.range_geometry())
    np.testing.assert_almost_equal(E1.norm(), np.sqrt(8), 1e-5)
    
    print(E1.domain_geometry().shape, E1.range_geometry().shape)
    u1 = E1.domain_geometry().allocate('random_int')
    w1 = E1.range_geometry().allocate('random_int', symmetry = True)
       
    
    lhs = E1.direct(u1).dot(w1)
    rhs = u1.dot(E1.adjoint(w1))
    np.testing.assert_almost_equal(lhs, rhs)
    
    ###########################################################################
    # 2D geometry with channels
    ig2 = ImageGeometry(N, M, channels = C)
    Grad2 = Gradient(ig2, correlation = 'Space')
    
    E2 = SymmetrizedGradient(Grad2.range_geometry())
    np.testing.assert_almost_equal(E2.norm(), np.sqrt(12), 1e-6)
    
    print(E2.domain_geometry().shape, E2.range_geometry().shape)
    u2 = E2.domain_geometry().allocate('random_int')
    w2 = E2.range_geometry().allocate('random_int', symmetry = True)
#    
    lhs2 = E2.direct(u2).dot(w2)
    rhs2 = u2.dot(E2.adjoint(w2))
    np.testing.assert_almost_equal(lhs2, rhs2)
    
    ###########################################################################
    # 3D geometry no channels
    ig3 = ImageGeometry(N, M, K)
    Grad3 = Gradient(ig3, correlation = 'Space')
    
    E3 = SymmetrizedGradient(Grad3.range_geometry())
    np.testing.assert_almost_equal(E3.norm(), np.sqrt(12), 1e-6)
    
    print(E3.domain_geometry().shape, E3.range_geometry().shape)
    u3 = E3.domain_geometry().allocate('random_int')
    w3 = E3.range_geometry().allocate('random_int', symmetry = True)
#    
    lhs3 = E3.direct(u3).dot(w3)
    rhs3 = u3.dot(E3.adjoint(w3))
    np.testing.assert_almost_equal(lhs3, rhs3)  