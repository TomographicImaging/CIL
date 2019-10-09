# -*- coding: utf-8 -*-
#========================================================================
# Copyright 2019 Science Technology Facilities Council
# Copyright 2019 University of Manchester
#
# This work is part of the Core Imaging Library developed by Science Technology
# Facilities Council and University of Manchester
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#=========================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from ccpi.optimisation.functions import Function
from ccpi.framework import BlockDataContainer
from numbers import Number

class BlockFunction(Function):
    
    r'''BlockFunction acts as a separable sum function: f = [f_1,...,f_n]
    
      .. math::

          f([x_1,...,x_n]) = f_1(x_1) +  .... + f_n(x_n)
      |

    '''
    
    def __init__(self, *functions):
                
        super(BlockFunction, self).__init__()
        self.functions = functions      
        self.length = len(self.functions)
                                
    def __call__(self, x):
        
        r'''Evaluates the BlockFunction at a BlockDataContainer x
        
            :param: x (BlockDataContainer): must have as many rows as self.length

            returns ..math:: sum(f_i(x_i))
            
        '''
        
        if self.length != x.shape[0]:
            raise ValueError('BlockFunction and BlockDataContainer have incompatible size')
        t = 0                
        for i in range(x.shape[0]):
            t += self.functions[i](x.get_item(i))
        return t
    
    def convex_conjugate(self, x):
        
        r'''Convex conjugate of BlockFunction at x            
        
            .. math:: returns sum(f_i^{*}(x_i))
        
        '''       
        t = 0                
        for i in range(x.shape[0]):
            t += self.functions[i].convex_conjugate(x.get_item(i))
        return t  
    
    
    def proximal_conjugate(self, x, tau, out = None):
        
        r'''Proximal operator of BlockFunction at x: 
                 
                 .. math:: prox_{tau*f}(x) = sum_{i} prox_{tau*f_{i}}(x_{i}) 
        
        
        '''

        if out is not None:
            if isinstance(tau, Number):
                for i in range(self.length):
                    self.functions[i].proximal_conjugate(x.get_item(i), tau, out=out.get_item(i))
            else:
                for i in range(self.length):
                    self.functions[i].proximal_conjugate(x.get_item(i), tau.get_item(i),out=out.get_item(i))
            
        else:
                
            out = [None]*self.length
            if isinstance(tau, Number):
                for i in range(self.length):
                    out[i] = self.functions[i].proximal_conjugate(x.get_item(i), tau)
            else:
                for i in range(self.length):
                    out[i] = self.functions[i].proximal_conjugate(x.get_item(i), tau.get_item(i))
            
            return BlockDataContainer(*out) 

    
    def proximal(self, x, tau, out = None):
        
        r'''Proximal operator of the convex conjugate of BlockFunction at x:
        
            .. math:: prox_{tau*f^{*}}(x) = sum_{i} prox_{tau*f^{*}_{i}}(x_{i}) 
        '''
        
        if out is None:
                
            out = [None]*self.length
            if isinstance(tau, Number):
                for i in range(self.length):
                    out[i] = self.functions[i].proximal(x.get_item(i), tau)
            else:
                for i in range(self.length):
                    out[i] = self.functions[i].proximal(x.get_item(i), tau.get_item(i))                        
            
            return BlockDataContainer(*out)  
        
        else:
            if isinstance(tau, Number):
                for i in range(self.length):
                    self.functions[i].proximal(x.get_item(i), tau, out[i])
            else:
                for i in range(self.length):
                    self.functions[i].proximal(x.get_item(i), tau.get_item(i), out[i])
            
            
    
    def gradient(self,x, out=None):
        
        r'''Evaluates gradient of BlockFunction at x
        
            returns: BlockDataContainer .. math:: [f_{1}'(x_{1}), ... , f_{n}'(x_{n})]
                
        '''
        
        out = [None]*self.length
        for i in range(self.length):
            out[i] = self.functions[i].gradient(x.get_item(i))
            
        return  BlockDataContainer(*out)           
                            
    
    
if __name__ == '__main__':
    
    M, N, K = 2,3,5
    
    from ccpi.optimisation.functions import L2NormSquared, MixedL21Norm
    from ccpi.framework import ImageGeometry, BlockGeometry
    from ccpi.optimisation.operators import Gradient, Identity, BlockOperator
    import numpy
    import numpy as np
    
    
    ig = ImageGeometry(M, N)
    BG = BlockGeometry(ig, ig)
    
    u = ig.allocate('random_int')
    B = BlockOperator( Gradient(ig), Identity(ig) )
    
    U = B.direct(u)
    b = ig.allocate('random_int')
    
    f1 =  10 * MixedL21Norm()
    f2 =  0.5 * L2NormSquared(b=b)    
    
    f = BlockFunction(f1, f2)
    tau = 0.3
    
    print( " without out " )
    res_no_out = f.proximal_conjugate( U, tau)
    res_out = B.range_geometry().allocate()
    f.proximal_conjugate( U, tau, out = res_out)
    
    numpy.testing.assert_array_almost_equal(res_no_out[0][0].as_array(), \
                                            res_out[0][0].as_array(), decimal=4) 
    
    numpy.testing.assert_array_almost_equal(res_no_out[0][1].as_array(), \
                                            res_out[0][1].as_array(), decimal=4) 

    numpy.testing.assert_array_almost_equal(res_no_out[1].as_array(), \
                                            res_out[1].as_array(), decimal=4)     
    
                    
    
    ##########################################################################
    
    
    
    
    
    
    
#    zzz = B.range_geometry().allocate('random_int')
#    www = B.range_geometry().allocate()
#    www.fill(zzz)
    
#    res[0].fill(z)
    
    
    
    
#    f.proximal_conjugate(z, sigma, out = res)
    
#    print(z1[0][0].as_array())    
#    print(res[0][0].as_array())
    
    
    
    
#    U = BG.allocate('random_int')
#    RES = BG.allocate()
#    f = BlockFunction(f1, f2)
#    
#    z = f.proximal_conjugate(U, 0.2)
#    f.proximal_conjugate(U, 0.2, out = RES)
#    
#    print(z[0].as_array())
#    print(RES[0].as_array())
#    
#    print(z[1].as_array())
#    print(RES[1].as_array())    
    
    
    
    
    


    
