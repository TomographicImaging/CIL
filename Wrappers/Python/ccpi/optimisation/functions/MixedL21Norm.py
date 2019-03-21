#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 19:43:12 2019

@author: evangelos
"""

import numpy as np
from ccpi.optimisation.functions import Function, ScaledFunction
from ccpi.framework import DataContainer, ImageData, \
                           ImageGeometry, BlockDataContainer 

############################   mixed_L1,2NORM FUNCTIONS   #####################
class MixedL21Norm(Function):
    
    def __init__(self, **kwargs):

        super(MixedL21Norm, self).__init__()                      
        self.sym_tensor = kwargs.get('sym_tensor',False)
        
    def __call__(self, x, out=None):
        
        ''' Evaluates L1,2Norm at point x
            
            :param: x is a BlockDataContainer
                                
        '''                        
        if self.sym_tensor:
            tmp = np.sqrt(tmp1.as_array()[0]**2 +  tmp1.as_array()[1]**2 +  2*tmp1.as_array()[2]**2)            
        else:
            tmp = [ el*el for el in x ]
            res = sum(tmp).sqrt().sum()
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
        
        if self.sym_tensor:
            
                tmp2 = np.sqrt(x.as_array()[0]**2 +  x.as_array()[1]**2 +  2*x.as_array()[2]**2)/self.alpha
                res = x.divide(ImageData(tmp2).maximum(1.0))                                
        else:

                tmp = [ el*el for el in x]
                res = sum(tmp).sqrt().maximum(1.0) # DataContainer
                res = BlockDataContainer(x.get_item(0)/res, x.get_item(1)/res)
                                                   
        return res 
    
    def __rmul__(self, scalar):
        return ScaledFunction(self, scalar)     

if __name__ == '__main__':
    
    M, N, K = 2,3, 5
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
    
    # grad in the scaled function it gives ValueError
#    res1 = f_no_scaled.gradient(U)
#    res2 = f_scaled.gradient(U)
    
    # conj this will return 0
    print(f_no_scaled.convex_conjugate(U), f_scaled.convex_conjugate(U))
    
    # proximal conjugate
    tau = 10
    a1 = 0.5 * f_no_scaled.proximal_conjugate(U, tau)
    a2 = f_scaled.proximal_conjugate(U, tau)
    
    a1 == a2
    
    
    

    
    
    
        

#class mixed_L12Norm(Function):
#    
#    def __init__(self, alpha, **kwargs):
#
#        super(mixed_L12Norm, self).__init__() 
#        
#        self.alpha = alpha 
#        self.b = kwargs.get('b',None)                
#        self.sym_grad = kwargs.get('sym_grad',False)
#        
#    def __call__(self,x):
#        
#        if self.b is None:
#            tmp1 = x
#        else:
#            tmp1 = x - self.b            
##        
#        if self.sym_grad:
#            tmp = np.sqrt(tmp1.as_array()[0]**2 +  tmp1.as_array()[1]**2 +  2*tmp1.as_array()[2]**2)
#        else:
#            tmp = ImageData(tmp1.power(2).sum(axis=0)).sqrt()
#            
#        return self.alpha*tmp.sum()          
#                            
#    def gradient(self,x):
#        return ValueError('Not Differentiable')
#                            
#    def convex_conjugate(self,x):
#        return 0
#    
#    def proximal(self, x, tau):
#        pass
#    
#    def proximal_conjugate(self, x, tau): 
#        
#        if self.sym_grad:
#                tmp2 = np.sqrt(x.as_array()[0]**2 +  x.as_array()[1]**2 +  2*x.as_array()[2]**2)/self.alpha
#                res = x.divide(ImageData(tmp2).maximum(1.0))                                
#        else:
##                res = x.divide((ImageData(x.power(2).sum(axis=0)).sqrt()/self.alpha).maximum(1.0))
#                
#                a1 = x.get_item(0)
#                a2 = x.get_item(1)
#                c = ((((a1*a1) + (a2*a2)).sqrt())/self.alpha).maximum(1.0)
#                res = BlockDataContainer(x.get_item(0)/c, x.get_item(1)/c)
#                
#                
##                res = x.divide(x.squared_norm()).sqrt()/self.alpha).maximum(1.0))
#                                                   
#        return res 
