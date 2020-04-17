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
from ccpi.optimisation.functions import Function
from ccpi.optimisation.operators import Operator, ScaledOperator

import warnings

class FunctionOperatorComposition(Function):
    
    """ Composition of a function with an operator as : :math:`(F \otimes A)(x) = F(Ax)`
    
            :parameter function: :code:`Function` F
            :parameter operator: :code:`Operator` A
            
            
        For general operator, we have no explicit formulas for convex_conjugate,
        proximal and proximal_conjugate            
    
    """
    
    def __init__(self, function, operator, **kwargs):
        '''creator

    :param A: operator
    :type A: :code:`Operator`
    :param f: function
    :type f: :code:`Function`
    '''

        super(FunctionOperatorComposition, self).__init__()
        
        self.function = function     
        self.operator = operator
        
        if not isinstance(self.function, Function):
            raise ValueError('{} is not function '.format(type(self.function)))
                   
        if not isinstance(self.operator, Operator):
            raise ValueError('{} is not function '.format(type(self.operator))) 
            
        
        self.L = kwargs.get('L', None)   
        
        if self.L is None:            
            self.L = self.computeLipshitz
    
    @property    
    def computeLipshitz(self):
        
        try:
            self.L = self.function.L * self.operator.norm()**2 
        except ValueError as er:
            self.L = None
            warnings.warn("Lipschitz constant was not calculated") 
        return self.L            

        
    def __call__(self, x):
        
        """ Returns :math:`F(Ax)`
        """
    
        return self.function(self.operator.direct(x))  
    
    def gradient(self, x, out=None):
        
        """ Return the gradient of F(Ax), 
        
        ..math ::  (F(Ax))' = A^{T}F'(Ax)
            
        """
        
        tmp = self.operator.range_geometry().allocate()
        self.operator.direct(x, out=tmp)
        self.function.gradient(tmp, out=tmp)
        if out is None:
            return self.operator.adjoint(tmp)
        else: 
            self.operator.adjoint(tmp, out=out)
            
            
            

if __name__ == '__main__':
    
    from ccpi.framework import ImageGeometry, AcquisitionGeometry
    from ccpi.optimisation.operators import Identity, DiagonalOperator, LinearOperatorMatrix
    from ccpi.optimisation.functions import L2NormSquared
    from timeit import default_timer as timer
    
    ig = ImageGeometry(200,200,200) 
    
    A = Identity(ig)
    b = ig.allocate('random')     
    
    tmp = L2NormSquared(b=b)  
    
    t0 = timer()
    f = FunctionOperatorComposition(tmp, A)
    t1 = timer()
    print(f.L, t1-t0) 


    t2 = timer()
    f = FunctionOperatorComposition(tmp, A, L=4)
    t3 = timer()
    print(f.L, t3-t2) 
    
    import os
    from ccpi.astra.operators import AstraProjectorSimple 
    import tomophantom
    from tomophantom import TomoP2D    
    import numpy as np
    
    # Load Shepp-Logan phantom 
    model = 1 # select a model number from the library
    N = 256 
    path = os.path.dirname(tomophantom.__file__)
    path_library2D = os.path.join(path, "Phantom2DLibrary.dat")
    
    #This will generate a N_size x N_size phantom (2D)
    phantom_2D = TomoP2D.Model(model, N, path_library2D)
    
    # Create image geometry
    ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N)
    data = ig.allocate()
    data.fill(phantom_2D)
    
    # Create Acquisition data
    detectors = N
    angles = np.linspace(0, np.pi, 180, dtype=np.float32)
    
    ag = AcquisitionGeometry('parallel','2D', angles, detectors)  
    
    Aop = AstraProjectorSimple(ig, ag, 'cpu')      
    
    t2 = timer()
    f = FunctionOperatorComposition(tmp, Aop)
    t3 = timer()
    print(f.L, t3-t2)  
    
    t2 = timer()
    f = FunctionOperatorComposition(tmp, Aop, L=88184.546875)
    t3 = timer()
    print(f.L, t3-t2)      
    
    


      
    
    