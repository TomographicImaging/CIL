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

from ccpi.optimisation.operators import Gradient, Operator, LinearOperator, ScaledOperator
from ccpi.framework import ImageData, ImageGeometry, BlockGeometry, BlockDataContainer
import numpy 
from ccpi.optimisation.operators import FiniteDiff


class SymmetrizedGradient(Gradient):
    
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
    
    
    def __init__(self, gm_domain, bnd_cond = 'Neumann', **kwargs):
        
        super(SymmetrizedGradient, self).__init__(gm_domain, bnd_cond, **kwargs) 
                
        self.gm_domain = self.gm_range 
        self.bnd_cond = bnd_cond
        
        self.channels = self.gm_range.get_item(0).channels
        
        tmp_gm = len(self.gm_domain.geometries)*self.gm_domain.geometries
        
        self.gm_range = BlockGeometry(*tmp_gm)
        
        self.FD = FiniteDiff(self.gm_domain, direction = 0, bnd_cond = self.bnd_cond)
        
        tmp_sh = self.gm_domain.shape[0]
        if tmp_sh==2:
            self.order_ind = [0,2,1,3]
        elif tmp_sh==3:
            self.order_ind = [0,3,6,1,4,7,2,5,8]     
        elif tmp_sh==4:
            self.order_ind = [0,4,8,12,
                              1,5,9,13,
                              2,6,10,14,
                              3,7,11,15]
        else:
            print(" Not the correct dimension ")          
                
        
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
#            tmp = self.adjoint(x)
#            out.fill(tmp)
                    
                     
    def domain_geometry(self):
        return self.gm_domain
    
    def range_geometry(self):
        return self.gm_range

if __name__ == '__main__':   
    
    ###########################################################################  
    ## Symmetrized Gradient Tests
    from ccpi.framework import DataContainer
    from ccpi.optimisation.operators import Gradient, BlockOperator, FiniteDiff
    import numpy as np
    
    N, M = 2, 3
    K = 2
    C = 2
    
    ig1 = ImageGeometry(N, M)
    ig2 = ImageGeometry(N, M, channels=C)
    
    E1 = SymmetrizedGradient(ig1, correlation = 'Space', bnd_cond='Neumann')
    
    try:
        E1 = SymmetrizedGradient(ig1, correlation = 'SpaceChannels', bnd_cond='Neumann')
    except:
        print("No Channels to correlate")
        
    E2 = SymmetrizedGradient(ig2, correlation = 'SpaceChannels', bnd_cond='Neumann')  

    print(E1.domain_geometry().shape, E1.range_geometry().shape)
    print(E2.domain_geometry().shape, E2.range_geometry().shape)   
    
    #check Linear operator property
    
    u1 = E1.domain_geometry().allocate('random_int')
    u2 = E2.domain_geometry().allocate('random_int')
    
    # Need to allocate random_int at the Range of SymGradient
    
    #a1 = ig1.allocate('random_int')
    #a2 = ig1.allocate('random_int')
    #a3 = ig1.allocate('random_int')
    
    #a4 = ig1.allocate('random_int')
    #a5 = ig1.allocate('random_int')    
    #a6 = ig1.allocate('random_int')
    
    # TODO allocate has to create this symmetry by default!!!!!
    #w1 = BlockDataContainer(*[a1, a2, \
    #                           a2, a3]) 
    w1 = E1.range_geometry().allocate('random_int',symmetry=True)
    
    LHS = (E1.direct(u1) * w1).sum()
    RHS = (u1 * E1.adjoint(w1)).sum()
    
    numpy.testing.assert_equal(LHS, RHS)     
    
    u2 = E2.gm_domain.allocate('random_int')
    
    #aa1 = ig2.allocate('random_int')
    #aa2 = ig2.allocate('random_int')
    #aa3 = ig2.allocate('random_int')
    #aa4 = ig2.allocate('random_int')
    #aa5 = ig2.allocate('random_int')
    #aa6 = ig2.allocate('random_int')  
    
    #w2 = BlockDataContainer(*[aa1, aa2, aa3, \
    #                          aa2, aa4, aa5, \
    #                          aa3, aa5, aa6])     
    w2 = E2.range_geometry().allocate('random_int',symmetry=True)
 
    
    LHS1 = (E2.direct(u2) * w2).sum()
    RHS1 = (u2 * E2.adjoint(w2)).sum()
    
    numpy.testing.assert_equal(LHS1, RHS1)      
    
    out = E1.range_geometry().allocate()
    E1.direct(u1, out=out)
    a1 = E1.direct(u1)
    numpy.testing.assert_array_equal(a1[0].as_array(), out[0].as_array()) 
    numpy.testing.assert_array_equal(a1[1].as_array(), out[1].as_array()) 
    numpy.testing.assert_array_equal(a1[2].as_array(), out[2].as_array()) 
    numpy.testing.assert_array_equal(a1[3].as_array(), out[3].as_array()) 
    
    
    out1 = E1.domain_geometry().allocate()
    E1.adjoint(w1, out=out1)
    b1 = E1.adjoint(w1)    
    
    LHS_out = (out * w1).sum()
    RHS_out = (u1 * out1).sum()
    print(LHS_out, RHS_out)
    
    
    out2 = E2.range_geometry().allocate()
    E2.direct(u2, out=out2)
    a2 = E2.direct(u2)
    
    out21 = E2.domain_geometry().allocate()
    E2.adjoint(w2, out=out21)
    b2 = E2.adjoint(w2)    
    
    LHS_out = (out2 * w2).sum()
    RHS_out = (u2 * out21).sum()
    print(LHS_out, RHS_out)    
    
    
    out = E1.range_geometry().allocate()
    E1.direct(u1, out=out)
    E1.adjoint(out, out=out1)
    
    print(E1.norm())
    print(E2.norm())
    
  
