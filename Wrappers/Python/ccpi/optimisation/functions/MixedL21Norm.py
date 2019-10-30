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

from ccpi.optimisation.functions import Function, ScaledFunction
from ccpi.framework import BlockDataContainer
import numpy as np

import functools

class MixedL21Norm(Function):
    
    
    '''
        f(x) = ||x||_{2,1} = \sum |x|_{2}                   
    '''      
    
    def __init__(self, **kwargs):

        super(MixedL21Norm, self).__init__()                      
        self.SymTensor = kwargs.get('SymTensor',False)
        
    def __call__(self, x):
        
        ''' Evaluates L2,1Norm at point x
            
            :param: x is a BlockDataContainer
                                
        '''
        if not isinstance(x, BlockDataContainer):
            raise ValueError('__call__ expected BlockDataContainer, got {}'.format(type(x))) 
                                         
        tmp = x.get_item(0) * 0.
        for el in x.containers:
            tmp += el.power(2.)
        return tmp.sqrt().sum()

                            
    def gradient(self, x, out=None):
        return ValueError('Not Differentiable')
                            
    def convex_conjugate(self,x):
        
        ''' This is the Indicator function of ||\cdot||_{2, \infty}
            which is either 0 if ||x||_{2, \infty} or \infty        
        '''
        
        return 0.0
        
    
    def proximal(self, x, tau, out=None):
        
        if out is None:
            
            tmp = sum([ el*el for el in x.containers]).sqrt()
            res = (tmp - tau).maximum(0.0) * x/tmp
            return res
            
        else:
                        
            tmp = functools.reduce(lambda a,b: a + b*b, x.containers, x.get_item(0) * 0 ).sqrt()
            res = (tmp - tau).maximum(0.0) * x/tmp

            for el in res.containers:
                el.as_array()[np.isnan(el.as_array())]=0

            out.fill(res)
        
    
    def proximal_conjugate(self, x, tau, out=None): 

        
        if out is None:                                        
            tmp = x.get_item(0) * 0	
            for el in x.containers:	
                tmp += el.power(2.)	
            tmp.sqrt(out=tmp)	
            tmp.maximum(1.0, out=tmp)	
            frac = [ el.divide(tmp) for el in x.containers ]	
            return BlockDataContainer(*frac)
        
    
        else:
                            
            res1 = functools.reduce(lambda a,b: a + b*b, x.containers, x.get_item(0) * 0 )
            res1.sqrt(out=res1)	
            res1.maximum(1.0, out=res1)	
            x.divide(res1, out=out)
                              

    def __rmul__(self, scalar):
        
        ''' Multiplication of MixedL21Norm with a scalar
        
        Returns: ScaledFunction
             
        '''         
        return ScaledFunction(self, scalar) 


#
if __name__ == '__main__':
    
    M, N, K = 2,3,50
    from ccpi.framework import BlockGeometry, ImageGeometry
    import numpy
    
    ig = ImageGeometry(M, N)
    
    BG = BlockGeometry(ig, ig)
    
    U = BG.allocate('random_int')
    
    # Define no scale and scaled
    alpha = 0.5
    f_no_scaled = MixedL21Norm() 
    f_scaled = alpha * MixedL21Norm()  
    
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
    
    
    tau = 0.4
    d1 = f_scaled.proximal(U, tau)
    
    tmp = (U.get_item(0)**2 + U.get_item(1)**2).sqrt()
    
    d2 = (tmp - alpha*tau).maximum(0) * U/tmp
    
    numpy.testing.assert_array_almost_equal(d1.get_item(0).as_array(), \
                                            d2.get_item(0).as_array(), decimal=4) 

    numpy.testing.assert_array_almost_equal(d1.get_item(1).as_array(), \
                                            d2.get_item(1).as_array(), decimal=4)     
    
    out1 = BG.allocate('random_int')
    
    
    f_scaled.proximal(U, tau, out = out1)
    
    numpy.testing.assert_array_almost_equal(out1.get_item(0).as_array(), \
                                            d1.get_item(0).as_array(), decimal=4) 

    numpy.testing.assert_array_almost_equal(out1.get_item(1).as_array(), \
                                            d1.get_item(1).as_array(), decimal=4)      
#    
    
    
    

    

    