#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 22:50:04 2019

@author: evangelos
"""

from ccpi.optimisation.operators import Operator, LinearOperator, ScaledOperator
from ccpi.optimisation.ops import PowerMethodNonsquare
from ccpi.framework import ImageData, ImageGeometry, BlockGeometry, BlockDataContainer
import numpy 
from ccpi.optimisation.operators import FiniteDiff, SparseFiniteDiff

#%%

class Gradient(LinearOperator):
    
    def __init__(self, gm_domain, bnd_cond = 'Neumann', **kwargs):
        
        super(Gradient, self).__init__() 
                
        self.gm_domain = gm_domain # Domain of Grad Operator
        
        self.correlation = kwargs.get('correlation','Space')
        
        if self.correlation=='Space':
            if self.gm_domain.channels>1:
                self.gm_range = BlockGeometry(*[self.gm_domain for _ in range(self.gm_domain.length-1)] ) 
                self.ind = numpy.arange(1,self.gm_domain.length)
            else:    
                self.gm_range = BlockGeometry(*[self.gm_domain for _ in range(self.gm_domain.length) ] )
                self.ind = numpy.arange(self.gm_domain.length)
        elif self.correlation=='SpaceChannels':
            if self.gm_domain.channels>1:
                self.gm_range = BlockGeometry(*[self.gm_domain for _ in range(self.gm_domain.length)])
                self.ind = range(self.gm_domain.length)
            else:
                raise ValueError('No channels to correlate')
         
        self.bnd_cond = bnd_cond    
                                                         
        
    def direct(self, x, out=None):
        
        tmp = self.gm_range.allocate()        
        for i in range(tmp.shape[0]):
            tmp.get_item(i).fill(FiniteDiff(self.gm_domain, direction = self.ind[i], bnd_cond = self.bnd_cond).direct(x))
        return tmp    
        
    def adjoint(self, x, out=None):
        
        tmp = self.gm_domain.allocate()
        for i in range(x.shape[0]):
            tmp+=FiniteDiff(self.gm_domain, direction = self.ind[i], bnd_cond = self.bnd_cond).adjoint(x.get_item(i))
        return tmp    
            
    
    def domain_geometry(self):
        return self.gm_domain
    
    def range_geometry(self):
        return self.gm_range
                                   
    def norm(self):

        x0 = self.gm_domain.allocate('random')
        self.s1, sall, svec = PowerMethodNonsquare(self, 10, x0)
        return self.s1
    
    def __rmul__(self, scalar):
        return ScaledOperator(self, scalar) 
    
    
    def matrix(self):
        
        tmp = self.gm_range.allocate()
        
        mat = []
        for i in range(tmp.shape[0]):
            
            spMat = SparseFiniteDiff(self.gm_domain, direction=self.ind[i], bnd_cond=self.bnd_cond)
            mat.append(spMat.matrix())
    
        return BlockDataContainer(*mat)    


    def sum_abs_row(self):
        
        tmp = self.gm_range.allocate()
        res = self.gm_domain.allocate()
        for i in range(tmp.shape[0]):
            spMat = SparseFiniteDiff(self.gm_domain, direction=self.ind[i], bnd_cond=self.bnd_cond)
            res += spMat.sum_abs_row()
        return res
    
    def sum_abs_col(self):
        
        tmp = self.gm_range.allocate()
        res = []
        for i in range(tmp.shape[0]):
            spMat = SparseFiniteDiff(self.gm_domain, direction=self.ind[i], bnd_cond=self.bnd_cond)
            res.append(spMat.sum_abs_col())
        return BlockDataContainer(*res)
   
       
if __name__ == '__main__':
    
    
    from ccpi.optimisation.operators import Identity, BlockOperator
    
    M, N = 2, 3
    ig = ImageGeometry(M, N)
    arr = ig.allocate('random_int' )
    
    # check direct of Gradient and sparse matrix
    G = Gradient(ig)
    G_sp = G.matrix()
    
    res1 = G.direct(arr)
    res1y = numpy.reshape(G_sp[0].toarray().dot(arr.as_array().flatten('F')), ig.shape, 'F')
    
    print(res1[0].as_array())
    print(res1y)
    
    res1x = numpy.reshape(G_sp[1].toarray().dot(arr.as_array().flatten('F')), ig.shape, 'F')
    
    print(res1[1].as_array())
    print(res1x)    
    
    #check sum abs row
    conc_spmat = numpy.abs(numpy.concatenate( (G_sp[0].toarray(), G_sp[1].toarray() )))
    print(numpy.reshape(conc_spmat.sum(axis=0), ig.shape, 'F'))    
    print(G.sum_abs_row().as_array())
    
    print(numpy.reshape(conc_spmat.sum(axis=1), ((2,) + ig.shape), 'F'))
    
    print(G.sum_abs_col()[0].as_array())
    print(G.sum_abs_col()[1].as_array())   
    
    # Check Blockoperator sum abs col and row
    
    op1 = Gradient(ig)
    op2 = Identity(ig)
    
    B = BlockOperator( op1, op2)
    
    Brow = B.sum_abs_row()
    Bcol = B.sum_abs_col()
    
    concB = numpy.concatenate( (numpy.abs(numpy.concatenate( (G_sp[0].toarray(), G_sp[1].toarray() ))), op2.matrix().toarray()))
    
    print(numpy.reshape(concB.sum(axis=0), ig.shape, 'F'))
    print(Brow.as_array())
    
    print(numpy.reshape(concB.sum(axis=1)[0:12], ((2,) + ig.shape), 'F'))
    print(Bcol[1].as_array())    
    
        
#    print(numpy.concatene(G_sp[0].toarray()+ ))
#    print(G_sp[1].toarray())
#    
#    d1 = G.sum_abs_row()
#    print(d1.as_array())
#    
#    d2 = G_neum.sum_abs_col()
##    print(d2)    
#    
#    
#    ###########################################################
    a = BlockDataContainer( BlockDataContainer(arr, arr), arr)
    b = BlockDataContainer( BlockDataContainer(arr+5, arr+3), arr+2)
    c = a/b
    
    print(c[0][0].as_array(), (arr/(arr+5)).as_array())
    print(c[0][1].as_array(), (arr/(arr+3)).as_array())
    print(c[1].as_array(), (arr/(arr+2)).as_array())
    
    
    a1 = BlockDataContainer( arr, BlockDataContainer(arr, arr))
#    
    c1 = arr + a
#    c2 = arr + a
#    c2 = a1 + arr
#    
