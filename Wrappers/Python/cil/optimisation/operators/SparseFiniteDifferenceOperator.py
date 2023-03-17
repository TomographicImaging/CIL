# -*- coding: utf-8 -*-
#  Copyright 2019 United Kingdom Research and Innovation
#  Copyright 2019 The University of Manchester
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Authors:
# CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt

import scipy.sparse as sp
import numpy as np
from cil.framework import ImageData
from cil.optimisation.operators import Operator

class SparseFiniteDifferenceOperator(Operator):
    
    
    '''Create Sparse Matrices for the Finite Difference Operator'''
    
    
    def __init__(self, domain_geometry, range_geometry=None, 
      direction=0, bnd_cond = 'Neumann'):
        
        super(SparseFiniteDifferenceOperator, self).__init__(domain_geometry=domain_geometry,
                                               range_geometry=range_geometry) 
        self.direction = direction
        self.bnd_cond = bnd_cond
        
        if self.range_geometry is None:
            self.range_geometry = self.domain_geometry
            
        self.get_dims = [i for i in domain_geometry.shape]  
        
        if self.direction + 1 > len(self.domain_geometry().shape):
            raise ValueError('Gradient directions more than geometry domain')         
            
    def matrix(self):    
        
            i = self.direction 
            
            mat = sp.spdiags(np.vstack([-np.ones((1,self.get_dims[i])),np.ones((1,self.get_dims[i]))]), [0,1], self.get_dims[i], self.get_dims[i], format = 'lil')

            if self.bnd_cond == 'Neumann':
                mat[-1,:] = 0
            elif self.bnd_cond == 'Periodic':
                mat[-1,0] = 1    
                
            tmpGrad = mat if i == 0 else sp.eye(self.get_dims[0])
            
            for j in range(1, self.domain_geometry().length):

                tmpGrad = sp.kron(mat, tmpGrad ) if j == i else sp.kron(sp.eye(self.get_dims[j]), tmpGrad ) 
                
            return tmpGrad    
                         
    def T(self):        
        return self.matrix().T
     
    def direct(self, x):
        
        x_asarr = x.as_array()
        res = np.reshape( self.matrix() * x_asarr.flatten('F'), self.domain_geometry().shape, 'F')
        return type(x)(res)
    
    def adjoint(self, x):
        
        x_asarr = x.as_array()
        res = np.reshape( self.matrix().T * x_asarr.flatten('F'), self.domain_geometry().shape, 'F')
        return type(x)(res) 
    
    def sum_abs_row(self):
        
        res = np.array(np.reshape(abs(self.matrix()).sum(axis=0), self.domain_geometry().shape, 'F'))
        #res[res==0]=0
        return ImageData(res)
    
    def sum_abs_col(self):
        
        res = np.array(np.reshape(abs(self.matrix()).sum(axis=1), self.domain_geometry().shape, 'F') )
        #res[res==0]=0
        return ImageData(res)

if __name__ == '__main__':
    from cil.framework import ImageGeometry        
    M, N= 2, 3
    ig = ImageGeometry(M, N)
    arr = ig.allocate('random_int')
    sFD_neum1 = SparseFiniteDifferenceOperator(ig, direction=0, bnd_cond='Neumann')
    sFD_neum2 = SparseFiniteDifferenceOperator(ig, direction=1, bnd_cond='Neumann')
    DY = sFD_neum1.matrix().toarray()
    DX = sFD_neum2.matrix().toarray()
    

    rows = sFD_neum1.sum_abs_row()
    cols = sFD_neum1.sum_abs_col()  

    print(rows.as_array())
    print(cols.as_array())