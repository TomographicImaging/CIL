#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:22:06 2020

@author: vaggelis
"""

from ccpi.framework import ImageGeometry_cupy as ImageGeometry
from ccpi.optimisation.operators import Gradient_cupy
from ccpi.optimisation.operators import Identity, BlockOperator_cupy

ig = ImageGeometry(voxel_num_x = 30, voxel_num_y=20)

x = ig.allocate('random_int')

G = Gradient_cupy(ig)
Id = Identity(ig)
B = BlockOperator_cupy(G, Id)


##%%
res1 = G.range_geometry().allocate()
res2 = G.domain_geometry().allocate()
#
G.direct(x, out = res1)
G.adjoint(res1, out = res2)


##%%
res1 = Id.range_geometry().allocate()
res2 = Id.domain_geometry().allocate()
#
Id.direct(x, out = res1)
Id.adjoint(res1, out = res2)

res1 = B.range_geometry().allocate()
res2 = B.domain_geometry().allocate()
#
B.direct(x, out = res1)
B.adjoint(res1, out = res2)

#%%

#from ccpi.framework import ImageGeometry
#from ccpi.optimisation.operators import Gradient
#from ccpi.optimisation.operators import Identity, BlockOperator
#
#ig = ImageGeometry(voxel_num_x = 30, voxel_num_y=20)
#
#x = ig.allocate('random_int')
#
#G = Gradient(ig)
#Id = Identity(ig)
#B = BlockOperator(G, Id)
#
#
###%%
#res1 = G.range_geometry().allocate()
#res2 = G.domain_geometry().allocate()
##
#G.direct(x, out = res1)
#G.adjoint(res1, out = res2)
#
#
###%%
#res1 = Id.range_geometry().allocate()
#res2 = Id.domain_geometry().allocate()
##
#Id.direct(x, out = res1)
#Id.adjoint(res1, out = res2)
#
#res1 = B.range_geometry().allocate()
#res2 = B.domain_geometry().allocate()
##
#B.direct(x, out = res1)
#B.adjoint(res1, out = res2)
#
#
#
#
#
##%%