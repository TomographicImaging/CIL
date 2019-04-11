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

import numpy as np
from ccpi.optimisation.functions import Function, ScaledFunction
from ccpi.framework import DataContainer, ImageData, \
                           ImageGeometry, BlockDataContainer 
import functools

############################   mixed_L1,2NORM FUNCTIONS   #####################
class MixedL21Norm(Function):
    
    def __init__(self, **kwargs):

        super(MixedL21Norm, self).__init__()                      
        self.SymTensor = kwargs.get('SymTensor',False)
        
    def __call__(self, x, out=None):
        
        ''' Evaluates L1,2Norm at point x
            
            :param: x is a BlockDataContainer
                                
        '''
        if not isinstance(x, BlockDataContainer):
            raise ValueError('__call__ expected BlockDataContainer, got {}'.format(type(x)))                      
        if self.SymTensor:
            
            param = [1]*x.shape[0]
            param[-1] = 2
            tmp = [param[i]*(x[i] ** 2) for i in range(x.shape[0])]
            res = sum(tmp).sqrt().sum()           
        else:
            
#            tmp = [ x[i]**2 for i in range(x.shape[0])]
            tmp = [ el**2 for el in x.containers ]
            
#            print(x.containers)
#            print(tmp)
#            print(type(sum(tmp)))
#            print(type(tmp))
            res = sum(tmp).sqrt().sum()
#            print(res)
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
                res = BlockDataContainer(*frac)    
                return res
            else:
                res1 = functools.reduce(lambda a,b: a + b*b, x.containers, x.get_item(0) * 0 )
                res = res1.sqrt().maximum(1.0)
                
                if False:
                    # works but not memory efficient as allocating a new BlockDataContainer
                    a = x / res
                    out.fill(a)
                elif False:
                    # this leads to error
#   File "ccpi\framework\BlockDataContainer.py", line 142, in divide
#     return type(self)(*[ el.divide(other, *args, **kwargs) for el in self.containers], shape=self.shape)
#   File "ccpi\framework\BlockDataContainer.py", line 142, in <listcomp>
#     return type(self)(*[ el.divide(other, *args, **kwargs) for el in self.containers], shape=self.shape)
#   File "ccpi\framework\framework.py", line 814, in divide
#     return self.pixel_wise_binary(numpy.divide, other, *args, **kwargs)
#   File "ccpi\framework\framework.py", line 802, in pixel_wise_binary
#     raise ValueError (message(type(self),  "incompatible class:" , pwop.__name__, type(out)))
# ValueError: ImageData: incompatible class: true_divide <class 'ccpi.framework.BlockDataContainer.BlockDataContainer'>
                    x.divide(res, out=out)
                else:
                    for i,el in enumerate(x.containers):
                        #a = out.get_item(i)
                        el.divide(res, out=out.get_item(i))

    def __rmul__(self, scalar):
        return ScaledFunction(self, scalar) 

#class MixedL21Norm_tensor(Function):
#    
#    def __init__(self):
#        print("feerf")
#    
#
if __name__ == '__main__':
    
    M, N, K = 2,3,5
    ig = ImageGeometry(voxel_num_x=M, voxel_num_y = N)
    u1 = ig.allocate('random_int')
    u2 = ig.allocate('random_int')
    
    U = BlockDataContainer(u1, u2, shape=(2,1))
    
    # Define no scale and scaled
    f_no_scaled = MixedL21Norm() 
    f_scaled = 0.5 * MixedL21Norm()  
    
    # call
    
    a1 = f_no_scaled(U)
    a2 = f_scaled(U)    
    
    z1 = f_no_scaled.proximal_conjugate(U, 1)
    z2 = f_scaled.proximal_conjugate(U, 1)
    

    

    
