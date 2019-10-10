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

from ccpi.optimisation.functions import Function
from ccpi.optimisation.functions import ScaledFunction


class FunctionOperatorComposition(Function):
    
    '''Function composition with Operator: (f o A)(x) = f(Ax)
    
            : parameter A: operator
            : parameter f: function
    
    '''
    
    def __init__(self, function, operator):
        
        super(FunctionOperatorComposition, self).__init__()
        
        self.function = function     
        self.operator = operator
        try:
            self.L = function.L * operator.norm()**2 
        except Error as er:
            self.L = None
            warnings.warn("Lipschitz constant was not calculated")
        
    def __call__(self, x):
        
        '''Evaluates f(Ax)'''
    
        return self.function(self.operator.direct(x))  
    
    def gradient(self, x, out=None):
        
        '''Evaluates gradient of f(Ax):
        
        ..math ::  A^{T}f'(Ax)
            
        '''
        
        tmp = self.operator.range_geometry().allocate()
        self.operator.direct(x, out=tmp)
        self.function.gradient(tmp, out=tmp)
        if out is None:
            #return self.operator.adjoint(self.function.gradient(self.operator.direct(x)))
            return self.operator.adjoint(tmp)
        else: 
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

    u = ig.allocate('random_int', seed=15)
    u1 = ig.allocate('random_int', seed=10)
    b = Aop.direct(u1)
    
        
#    G = Gradient(ig)
    alpha = 0.5
    
    f1 =  alpha * L2NormSquared(b=b)    

    f_comp = FunctionOperatorComposition(f1, Aop)
    
    print(f_comp(u))
    
    
    z1 = Aop.direct(u)
    tmp = 0.5 * ((z1 - b)**2).sum()
    
   
    print(tmp)
    
    
    
    
    
    
          
    

             
        
                       
