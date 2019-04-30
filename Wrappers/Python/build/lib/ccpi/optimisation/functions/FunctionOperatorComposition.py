#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 09:55:36 2019

@author: evangelos
"""

from ccpi.optimisation.functions import Function
from ccpi.optimisation.functions import ScaledFunction


class FunctionOperatorComposition(Function):
    
    ''' Function composition with Operator, i.e., f(Ax)
    
        A: operator
        f: function
    
    '''
    
    def __init__(self, function, operator):
        
        super(FunctionOperatorComposition, self).__init__()
        
        self.function = function     
        self.operator = operator
        self.L = function.L * operator.norm()**2 
        
        
    def __call__(self, x):
        
        ''' Evaluate FunctionOperatorComposition at x
        
        returns f(Ax)
        
        '''
    
        return self.function(self.operator.direct(x))  
    
    def gradient(self, x, out=None):
#        
        ''' Gradient takes into account the Operator'''
        if out is None:
            return self.operator.adjoint(self.function.gradient(self.operator.direct(x)))
        else: 
            tmp = self.operator.range_geometry().allocate()
            self.operator.direct(x, out=tmp)
            self.function.gradient(tmp, out=tmp)
            self.operator.adjoint(tmp, out=out)

    
    

    #TODO do not know if we need it
    #def call_adjoint(self, x):
    #
    #    return self.function(self.operator.adjoint(x))  


    #def convex_conjugate(self, x):
    #    
    #        ''' convex_conjugate does not take into account the Operator'''
    #    return self.function.convex_conjugate(x)

    


                
if __name__ == '__main__':   

    from ccpi.framework import ImageGeometry
    from ccpi.optimisation.operators import Gradient
    from ccpi.optimisation.functions import L2NormSquared
    
    M, N, K = 2,3
    ig = ImageGeometry(voxel_num_x=M, voxel_num_y = N)
    
    G = Gradient(ig)
    alpha = 0.5
    
    f = L2NormSquared()    
    f_comp = FunctionOperatorComposition(G, alpha * f)
    x = ig.allocate('random_int')
    print(f_comp.gradient(x).shape
          
          )
    

             
        
                       