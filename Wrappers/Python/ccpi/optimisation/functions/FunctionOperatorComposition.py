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

    

                
if __name__ == '__main__':   

    from ccpi.framework import ImageGeometry, AcquisitionGeometry
    from ccpi.optimisation.operators import Gradient
    from ccpi.optimisation.functions import L2NormSquared
    from ccpi.astra.ops import AstraProjectorSimple
    import numpy as np
        
    M, N= 50, 50
    ig = ImageGeometry(voxel_num_x=M, voxel_num_y = N)
    
    detectors = N
    angles_num = N    
    det_w = 1.0
    
    angles = np.linspace(0, np.pi, angles_num, endpoint=False)
    ag = AcquisitionGeometry('parallel',
                             '2D',
                             angles,
                             detectors,det_w)
    
    
    Aop = AstraProjectorSimple(ig, ag, 'cpu')    

    u = ig.allocate('random_int')
    u1 = ig.allocate('random_int')
    b = Aop.direct(u1)
    
        
#    G = Gradient(ig)
    alpha = 0.5
    
    f1 =  alpha * L2NormSquared(b=b)    

    f_comp = FunctionOperatorComposition(f1, Aop)
    
    print(f_comp(u))
    
    
    z1 = Aop.direct(u)
    tmp = 0.5 * ((z1 - b)**2).sum()
    
   
    print(tmp)
    
    
    
    
    
    
          
    

             
        
                       