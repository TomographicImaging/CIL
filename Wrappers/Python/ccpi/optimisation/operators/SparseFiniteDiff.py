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

import scipy.sparse as sp
import numpy as np
from ccpi.framework import ImageData
from ccpi.optimisation.operators import Operator

class SparseFiniteDiff(Operator):
    
    
    '''Create Sparse Matrices for the Finite Difference Operator'''
    
    
    def __init__(self, domain_geometry, range_geometry=None, 
      direction=0, bnd_cond = 'Neumann'):
        
        super(SparseFiniteDiff, self).__init__(domain_geometry=domain_geometry,
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
     
    def direct(self, x, out = None):
        
        x_asarr = x.as_array()
        res = np.reshape( self.matrix() * x_asarr.flatten('F'), self.domain_geometry().shape, 'F')
        
        if out is None:
            return res
        else:
            out.fill(res)
                        
    
    def adjoint(self, x, out = None):
        
        x_asarr = x.as_array()
        res = np.reshape( self.matrix().T * x_asarr.flatten('F'), self.domain_geometry().shape, 'F')
        
        if out is None:
            return res
        else:
            out.fill(res)
    
    def sum_abs_row(self):
        
        tmp = self.domain_geometry().allocate()
        res = np.array(np.reshape(abs(self.matrix()).sum(axis=0), self.domain_geometry().shape, 'F'))
        tmp.fill(res)

        return tmp
    
    def sum_abs_col(self):
        
        tmp = self.domain_geometry().allocate()
        res = np.array(np.reshape(abs(self.matrix()).sum(axis=1), self.domain_geometry().shape, 'F') )
        tmp.fill(res)
        return tmp

#
#from ccpi.framework import ImageGeometry        
#M, N, Κ = 2, 3, 2
#ig = ImageGeometry(M, N, Κ)
#arr = ig.allocate('random_int')
#sFD_neum1 = SparseFiniteDiff(ig, direction=0, bnd_cond='Periodic')
#sFD_neum2 = SparseFiniteDiff(ig, direction=1, bnd_cond='Periodic')
#DY = sFD_neum1.matrix().toarray()
#DX = sFD_neum2.matrix().toarray()
##  
##
#rows = sFD_neum1.sum_abs_row()
#cols = sFD_neum1.sum_abs_col()  
#
#print(rows.as_array())
#print(cols.as_array())

#%%

#from ccpi.framework import BlockOp











#%%    
if __name__ == '__main__':
    
    from ccpi.framework import ImageGeometry
    from ccpi.optimisation.operators import FiniteDiff
    
    # 2D
    M, N= 2, 3
    ig = ImageGeometry(M, N)
    arr = ig.allocate('random_int')
    
    for i in [0,1]:
    
        # Neumann
        sFD_neum = SparseFiniteDiff(ig, direction=i, bnd_cond='Neumann')
        G_neum = FiniteDiff(ig, direction=i, bnd_cond='Neumann')
        
        # Periodic
        sFD_per = SparseFiniteDiff(ig, direction=i, bnd_cond='Periodic')
        G_per = FiniteDiff(ig, direction=i, bnd_cond='Periodic')    
    
        u_neum_direct = G_neum.direct(arr)
        u_neum_sp_direct = sFD_neum.direct(arr)
        np.testing.assert_array_almost_equal(u_neum_direct.as_array(), u_neum_sp_direct.as_array(), decimal=4)
        
        u_neum_adjoint = G_neum.adjoint(arr)
        u_neum_sp_adjoint = sFD_neum.adjoint(arr)
        np.testing.assert_array_almost_equal(u_neum_adjoint.as_array(), u_neum_sp_adjoint.as_array(), decimal=4)    
        
        u_per_direct = G_neum.direct(arr)
        u_per_sp_direct = sFD_neum.direct(arr)  
        np.testing.assert_array_almost_equal(u_per_direct.as_array(), u_per_sp_direct.as_array(), decimal=4)
        
        u_per_adjoint = G_per.adjoint(arr)
        u_per_sp_adjoint = sFD_per.adjoint(arr)
        np.testing.assert_array_almost_equal(u_per_adjoint.as_array(), u_per_sp_adjoint.as_array(), decimal=4)      
    
    # 3D
    M, N, K = 2, 3, 4
    ig3D = ImageGeometry(M, N, K)
    arr3D = ig3D.allocate('random_int')
    
    for i in [0,1,2]:
    
        # Neumann
        sFD_neum3D = SparseFiniteDiff(ig3D, direction=i, bnd_cond='Neumann')
        G_neum3D = FiniteDiff(ig3D, direction=i, bnd_cond='Neumann')
        
        # Periodic
        sFD_per3D = SparseFiniteDiff(ig3D, direction=i, bnd_cond='Periodic')
        G_per3D = FiniteDiff(ig3D, direction=i, bnd_cond='Periodic')    
    
        u_neum_direct3D = G_neum3D.direct(arr3D)
        u_neum_sp_direct3D = sFD_neum3D.direct(arr3D)
        np.testing.assert_array_almost_equal(u_neum_direct3D.as_array(), u_neum_sp_direct3D.as_array(), decimal=4)
        
        u_neum_adjoint3D = G_neum3D.adjoint(arr3D)
        u_neum_sp_adjoint3D = sFD_neum3D.adjoint(arr3D)
        np.testing.assert_array_almost_equal(u_neum_adjoint3D.as_array(), u_neum_sp_adjoint3D.as_array(), decimal=4)    
        
        u_per_direct3D = G_neum3D.direct(arr3D)
        u_per_sp_direct3D = sFD_neum3D.direct(arr3D)  
        np.testing.assert_array_almost_equal(u_per_direct3D.as_array(), u_per_sp_direct3D.as_array(), decimal=4)
        
        u_per_adjoint3D = G_per3D.adjoint(arr3D)
        u_per_sp_adjoint3D = sFD_per3D.adjoint(arr3D)
        np.testing.assert_array_almost_equal(u_per_adjoint3D.as_array(), u_per_sp_adjoint3D.as_array(), decimal=4)      
        
            
    
    
