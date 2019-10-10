# -*- coding: utf-8 -*-
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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from ccpi.optimisation.functions import Function, ScaledFunction
from ccpi.framework import BlockDataContainer

import functools
import numpy

class MixedL21Norm(Function):
    
    
    r'''MixedL21Norm: .. math:: f(x) = ||x||_{2,1} = \int \|x\|_{2} dx

        where x is a vector/tensor vield
                
    '''      
    
    def __init__(self, **kwargs):

        super(MixedL21Norm, self).__init__()                      
        self.SymTensor = kwargs.get('SymTensor',False)
        
    def __call__(self, x):
        
        '''Evaluates MixedL21Norm at point x
            
           :param: x: is a BlockDataContainer
        '''
        if not isinstance(x, BlockDataContainer):
            raise ValueError('__call__ expected BlockDataContainer, got {}'.format(type(x))) 
        tmp = x.get_item(0) * 0
        for el in x.containers:
            tmp += el.power(2.)
        return tmp.sqrt().sum()

                            
    def gradient(self, x, out=None):
        return ValueError('Not Differentiable')
                            
    def convex_conjugate(self,x):
        
        r'''Convex conjugate of of MixedL21Norm: 
        
        Indicator function of .. math:: ||\cdot||_{2, \infty}
            which is either 0 if .. math:: ||x||_{2, \infty}<1 or \infty 
            
        '''
        
        return 0.0
        
    
    def proximal(self, x, tau, out=None):
        
        r'''Proximal operator of MixedL21Norm at x:
           
           .. math:: prox_{\tau * f(x)
        '''
        pass
    
    def proximal_conjugate(self, x, tau, out=None): 
        
        r'''Proximal operator of the convex conjugate of MixedL21Norm at x:
           
           .. math:: prox_{\tau * f^{*}}(x)

        '''           


        if out is None:                                        
            # tmp = [ el*el for el in x.containers]
            # res = sum(tmp).sqrt().maximum(1.0) 
            # frac = [el/res for el in x.containers]
            # return  BlockDataContainer(*frac)   
            tmp = x.get_item(0) * 0
            for el in x.containers:
                tmp += el.power(2.)
            tmp.sqrt(out=tmp)
            tmp.maximum(1.0, out=tmp)
            frac = [ el.divide(tmp) for el in x.containers ]
            return BlockDataContainer(*frac)
            
                
        else:
                            
            res1 = functools.reduce(lambda a,b: a + b*b, x.containers, x.get_item(0) * 0 )
            if False:
                res = res1.sqrt().maximum(1.0)
                x.divide(res, out=out)
            else:
                res1.sqrt(out=res1)
                res1.maximum(1.0, out=res1)
                x.divide(res1, out=out)
                                          

    def __rmul__(self, scalar):
        
        '''Multiplication of MixedL21Norm with a scalar        
            
            Returns: ScaledFunction
             
        '''         
        return ScaledFunction(self, scalar) 


def sqrt_maximum(x, a):
    y = numpy.sqrt(x)
    if y >= a:
        return y
    else:
        return a
#
if __name__ == '__main__':
    
    M, N, K = 2,3,5
    from ccpi.framework import BlockGeometry
    import numpy
    
    ig = ImageGeometry(M, N)
    
    BG = BlockGeometry(ig, ig)
    
    U = BG.allocate('random_int')
    
    # Define no scale and scaled
    f_no_scaled = MixedL21Norm() 
    f_scaled = 0.5 * MixedL21Norm()  
    
    # call
    
    a1 = f_no_scaled(U)
    a2 = f_scaled(U)    
    print(a1, 2*a2)
        
    
    print( " ####### check without out ######### " )
          
          
    u_out_no_out = BG.allocate('random_int')         
    res_no_out = f_scaled.proximal_conjugate(u_out_no_out, 0.5)          
    print(res_no_out[0].as_array())
    
    print( " ####### check with out ######### " ) 
#          
    res_out = BG.allocate()        
    f_scaled.proximal_conjugate(u_out_no_out, 0.5, out = res_out)
#    
    print(res_out[0].as_array())   
#
    numpy.testing.assert_array_almost_equal(res_no_out[0].as_array(), \
                                            res_out[0].as_array(), decimal=4)

    numpy.testing.assert_array_almost_equal(res_no_out[1].as_array(), \
                                            res_out[1].as_array(), decimal=4)     
#    
    
    
    

    

    
