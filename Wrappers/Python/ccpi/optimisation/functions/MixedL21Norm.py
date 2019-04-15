# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library developed by
#   Visual Analytics and Imaging System Group of the Science Technology
#   Facilities Council, STFC

#   Copyright 2018-2019 Evangelos Papoutsellis and Edoardo Pasca

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from ccpi.optimisation.functions import Function, ScaledFunction
from ccpi.framework import BlockDataContainer

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
                     
        if self.SymTensor:
            
            #TODO fix this case
            param = [1]*x.shape[0]
            param[-1] = 2
            tmp = [param[i]*(x[i] ** 2) for i in range(x.shape[0])]
            res = sum(tmp).sqrt().sum()   
            
        else:
                        
            #tmp = [ el**2 for el in x.containers ]
            #res = sum(tmp).sqrt().sum()
            res = x.pnorm()

        return res
                            
    def gradient(self, x, out=None):
        return ValueError('Not Differentiable')
                            
    def convex_conjugate(self,x):
        
        ''' This is the Indicator function of ||\cdot||_{2, \infty}
            which is either 0 if ||x||_{2, \infty} or \infty        
        '''
        return 0.0
    
    def proximal(self, x, tau, out=None):
        
        '''
            For this we need to define a MixedL2,2 norm acting on BDC,
            different form L2NormSquared which acts on DC
        
        '''
        pass
    
    def proximal_conjugate(self, x, tau, out=None): 

        if self.SymTensor:
            
            param = [1]*x.shape[0]
            param[-1] = 2
            tmp = [param[i]*(x[i] ** 2) for i in range(x.shape[0])]
            frac = [x[i]/(sum(tmp).sqrt()).maximum(1.0) for i in range(x.shape[0])]
            res = BlockDataContainer(*frac) 
            
            return res
        
        else:

            if out is None:                                        
                tmp = [ el*el for el in x.containers]
                res = sum(tmp).sqrt().maximum(1.0) 
                frac = [el/res for el in x.containers]
                return  BlockDataContainer(*frac)    
                
            #TODO this is slow, why???
#                return x.divide(x.pnorm().maximum(1.0))
            else:
                                
                res1 = functools.reduce(lambda a,b: a + b*b, x.containers, x.get_item(0) * 0 )
                res = res1.sqrt().maximum(1.0)
                x.divide(res, out=out)
                
#                x.divide(sum([el*el for el in x.containers]).sqrt().maximum(1.0), out=out)
                #TODO this is slow, why ???
#                 x.divide(x.norm().maximum(1.0), out=out)
                              

    def __rmul__(self, scalar):
        
        ''' Multiplication of L2NormSquared with a scalar
        
        Returns: ScaledFunction
             
        '''         
        return ScaledFunction(self, scalar) 


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
    
    
    

    

    
