#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 13:10:09 2019

@author: evangelos
"""

from ccpi.optimisation.operators import FiniteDiff, SparseFiniteDiff, BlockOperator, Gradient
from ccpi.framework import ImageGeometry, AcquisitionGeometry, BlockDataContainer, ImageData
from ccpi.astra.ops import AstraProjectorSimple

from scipy import sparse
import numpy as np

N = 3
ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N)
u = ig.allocate('random_int')

# Compare FiniteDiff with SparseFiniteDiff

DY = FiniteDiff(ig, direction = 0, bnd_cond = 'Neumann')
DX = FiniteDiff(ig, direction = 1, bnd_cond = 'Neumann')

DXu = DX.direct(u)
DYu = DY.direct(u)

DX_sparse = SparseFiniteDiff(ig, direction=1, bnd_cond='Neumann')
DY_sparse = SparseFiniteDiff(ig, direction=0, bnd_cond='Neumann')

DXu_sparse = DX_sparse.direct(u)
DYu_sparse = DY_sparse.direct(u)

#np.testing.assert_array_almost_equal(DYu.as_array(), DYu_sparse.as_array(), decimal=4)
#np.testing.assert_array_almost_equal(DXu.as_array(), DXu_sparse.as_array(), decimal=4)

#%%  Tau/ Sigma

A1 = DY_sparse.matrix()
A2 = DX_sparse.matrix()
A3 = sparse.eye(np.prod(ig.shape))

sum_rows1 = np.array(np.sum(abs(A1), axis=1))
sum_rows2 = np.array(np.sum(abs(A2), axis=1))
sum_rows3 = np.array(np.sum(abs(A3), axis=1))

sum_cols1 = np.array(np.sum(abs(A1), axis=0))
sum_cols2 = np.array(np.sum(abs(A2), axis=0))
sum_cols3 = np.array(np.sum(abs(A2), axis=0))

# Check if Grad sum row/cols is OK
Grad = Gradient(ig)

Sum_Block_row = Grad.sum_abs_row()
Sum_Block_col = Grad.sum_abs_col()

tmp1 = BlockDataContainer( ImageData(np.reshape(sum_rows1, ig.shape, order='F')),\
                           ImageData(np.reshape(sum_rows2, ig.shape, order='F')))


#np.testing.assert_array_almost_equal(tmp1[0].as_array(), Sum_Block_row[0].as_array(), decimal=4)
#np.testing.assert_array_almost_equal(tmp1[1].as_array(), Sum_Block_row[1].as_array(), decimal=4)

tmp2 = ImageData(np.reshape(sum_cols1 + sum_cols2, ig.shape, order='F'))

#np.testing.assert_array_almost_equal(tmp2.as_array(), Sum_Block_col.as_array(), decimal=4)


#%% BlockOperator with Gradient, Identity

Id = Identity(ig)
Block_GrId = BlockOperator(Grad, Id, shape=(2,1))


Sum_Block_GrId_row = Block_GrId.sum_abs_row()














