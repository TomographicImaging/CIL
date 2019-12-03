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
from __future__ import unicode_literals

from ccpi.optimisation.operators import Operator, LinearOperator, ScaledOperator
from ccpi.framework import ImageData, ImageGeometry, BlockGeometry, BlockDataContainer
import numpy 
from ccpi.optimisation.operators import FiniteDiff, SparseFiniteDiff

#%%

NEUMANN = 'Neumann'
PERIODIC = 'Periodic'
C = 'c'
NUMPY = 'numpy'
CORRELATION_SPACE = "Space"
CORRELATION_SPACECHANNEL = "SpaceChannels"

class Gradient(LinearOperator):

    r'''Gradient Operator: .. math:: \nabla : X -> Y           
            
            Computes first-order forward/backward differences 
                     on 2D, 3D, 4D ImageData
                     under Neumann/Periodic boundary conditions
                                                             
                Example (2D): u\in X, \nabla(u) = [\partial_{y} u, \partial_{x} u]
                              u^{*}\in Y, \nabla^{*}(u^{*}) = \partial_{y} v1 + \partial_{x} v2           
    '''

    #kept here for backwards compatability
    CORRELATION_SPACE = CORRELATION_SPACE
    CORRELATION_SPACECHANNEL = CORRELATION_SPACECHANNEL

    def __init__(self, gm_domain, bnd_cond = 'Neumann', **kwargs):
        
        super(Gradient, self).__init__() 

        backend = kwargs.get('backend',C)

        correlation = kwargs.get('correlation',CORRELATION_SPACE)

        if correlation == CORRELATION_SPACE and gm_domain.channels > 1:
            #numpy implementation only for now
            backend = NUMPY

        if backend == NUMPY:
            self.Operator = Gradient_numpy(gm_domain, bnd_cond=bnd_cond, **kwargs)
        else:
            self.Operator = Gradient_C(gm_domain, bnd_cond=bnd_cond)


    def direct(self, x, out=None):
        if out is None:
            out = self.Operator.direct(x, out=out)
            return out
        else:
            self.Operator.direct(x, out=out)
        
        
    def adjoint(self, x, out=None):
        if out is None:
            out = self.Operator.adjoint(x, out=out)
            return out
        else:
            self.Operator.adjoint(x, out=out)

    def domain_geometry(self):
        
        '''Returns domain_geometry of Gradient'''
        
        return self.Operator.gm_domain
    
    def range_geometry(self):
        
        '''Returns range_geometry of Gradient'''
        
        return self.Operator.gm_range

class Gradient_numpy(LinearOperator):
    
    
    r'''Gradient Operator: .. math:: \nabla : X -> Y           
            
            Computes first-order forward/backward differences 
                     on 2D, 3D, 4D ImageData
                     under Neumann/Periodic boundary conditions
                                                             
                Example (2D): u\in X, \nabla(u) = [\partial_{y} u, \partial_{x} u]
                              u^{*}\in Y, \nabla^{*}(u^{*}) = \partial_{y} v1 + \partial_{x} v2
        
                Grad_order = ['channels', 'direction_z', 'direction_y', 'direction_x']
                Grad_order = ['channels', 'direction_y', 'direction_x']
                Grad_order = ['direction_z', 'direction_y', 'direction_x']
                Grad_order = ['channels', 'direction_z', 'direction_y', 'direction_x']
                

    '''

    def __init__(self, gm_domain, bnd_cond = 'Neumann', **kwargs):
        
        super(Gradient_numpy, self).__init__() 
                
        self.gm_domain = gm_domain # Domain of Grad Operator
        
        self.correlation = kwargs.get('correlation',CORRELATION_SPACE)
        
        if self.correlation==CORRELATION_SPACE:
            if self.gm_domain.channels > 1:
                self.gm_range = BlockGeometry(*[self.gm_domain for _ in range(self.gm_domain.length-1)] )

                if self.gm_domain.length == 4:
                    # 3D + Channel
                    # expected Grad_order = ['channels', 'direction_z', 'direction_y', 'direction_x']
                    expected_order = [ImageGeometry.CHANNEL, ImageGeometry.VERTICAL, ImageGeometry.HORIZONTAL_Y, ImageGeometry.HORIZONTAL_X]
                else:
                    # 2D + Channel
                    # expected Grad_order = ['channels', 'direction_y', 'direction_x']
                    expected_order = [ImageGeometry.CHANNEL, ImageGeometry.HORIZONTAL_Y, ImageGeometry.HORIZONTAL_X]

                order = self.gm_domain.get_order_by_label(self.gm_domain.dimension_labels, expected_order)
                self.ind = order[1:]
                #self.ind = numpy.arange(1,self.gm_domain.length)
            else:
                # no channel info
                self.gm_range = BlockGeometry(*[self.gm_domain for _ in range(self.gm_domain.length) ] )
                if self.gm_domain.length == 3:
                    # 3D
                    # expected Grad_order = ['direction_z', 'direction_y', 'direction_x']
                    expected_order = [ImageGeometry.VERTICAL, ImageGeometry.HORIZONTAL_Y, ImageGeometry.HORIZONTAL_X]
                else:
                    # 2D
                    expected_order = [ImageGeometry.HORIZONTAL_Y, ImageGeometry.HORIZONTAL_X]    

                self.ind = self.gm_domain.get_order_by_label(self.gm_domain.dimension_labels, expected_order)
                # self.ind = numpy.arange(self.gm_domain.length)
                
        elif self.correlation==CORRELATION_SPACECHANNEL:
            if self.gm_domain.channels > 1:
                self.gm_range = BlockGeometry(*[self.gm_domain for _ in range(self.gm_domain.length)])
                self.ind = range(self.gm_domain.length)
            else:
                raise ValueError('No channels to correlate')
         
        self.bnd_cond = bnd_cond 
        
        # Call FiniteDiff operator        
        self.FD = FiniteDiff(self.gm_domain, direction = 0, bnd_cond = self.bnd_cond)
                                                         
        
    def direct(self, x, out=None):
        
                
        if out is not None:
            
            for i in range(self.gm_range.shape[0]):
                self.FD.direction = self.ind[i]
                self.FD.direct(x, out = out[i])
        else:
            tmp = self.gm_range.allocate()        
            for i in range(tmp.shape[0]):
                self.FD.direction=self.ind[i]
                tmp.get_item(i).fill(self.FD.direct(x))
            return tmp    
        
    def adjoint(self, x, out=None):
        
        if out is not None:

            tmp = self.gm_domain.allocate()            
            for i in range(x.shape[0]):
                self.FD.direction=self.ind[i] 
                self.FD.adjoint(x.get_item(i), out = tmp)
                if i == 0:
                    out.fill(tmp)
                else:
                    out += tmp
        else:            
            tmp = self.gm_domain.allocate()
            for i in range(x.shape[0]):
                self.FD.direction=self.ind[i]

                tmp += self.FD.adjoint(x.get_item(i))
            return tmp    
            
    
    def domain_geometry(self):
        
        '''Returns domain_geometry of Gradient'''
        
        return self.gm_domain
    
    def range_geometry(self):
        
        '''Returns range_geometry of Gradient'''
        
        return self.gm_range
    
    def __rmul__(self, scalar):
        
        '''Multiplication of Gradient with a scalar        
            
            Returns: ScaledOperator
        '''        
        
        return ScaledOperator(self, scalar) 
    
    ###########################################################################
    ###############  For preconditioning ######################################
    ###########################################################################
    def matrix(self):
        
        tmp = self.gm_range.allocate()
        
        mat = []
        for i in range(tmp.shape[0]):
            
            spMat = SparseFiniteDiff(self.gm_domain, direction=self.ind[i], bnd_cond=self.bnd_cond)
            mat.append(spMat.matrix())
    
        return BlockDataContainer(*mat)    


    def sum_abs_col(self):
        
        tmp = self.gm_range.allocate()
        res = self.gm_domain.allocate()
        for i in range(tmp.shape[0]):
            spMat = SparseFiniteDiff(self.gm_domain, direction=self.ind[i], bnd_cond=self.bnd_cond)
            res += spMat.sum_abs_row()
        return res
    
    def sum_abs_row(self):
        
        tmp = self.gm_range.allocate()
        res = []
        for i in range(tmp.shape[0]):
            spMat = SparseFiniteDiff(self.gm_domain, direction=self.ind[i], bnd_cond=self.bnd_cond)
            res.append(spMat.sum_abs_col())
        return BlockDataContainer(*res)
   
import ctypes, platform

# check for the extension
if platform.system() == 'Linux':
    dll = 'libcilacc.so'
elif platform.system() == 'Windows':
    dll = 'cilacc.dll'
elif platform.system() == 'Darwin':
    dll = 'libcilacc.dylib'
else:
    raise ValueError('Not supported platform, ', platform.system())

#print ("dll location", dll)
cilacc = ctypes.cdll.LoadLibrary(dll)

#FD = ctypes.CDLL(dll)

c_float_p = ctypes.POINTER(ctypes.c_float)

cilacc.openMPtest.restypes = ctypes.c_int32

cilacc.fdiff4D.argtypes = [ctypes.POINTER(ctypes.c_float),
                       ctypes.POINTER(ctypes.c_float),
                       ctypes.POINTER(ctypes.c_float),
                       ctypes.POINTER(ctypes.c_float),
                       ctypes.POINTER(ctypes.c_float),
                       ctypes.c_long,
                       ctypes.c_long,
                       ctypes.c_long,
                       ctypes.c_long,
                       ctypes.c_int32,
                       ctypes.c_int32]

cilacc.fdiff3D.argtypes = [ctypes.POINTER(ctypes.c_float),
                       ctypes.POINTER(ctypes.c_float),
                       ctypes.POINTER(ctypes.c_float),
                       ctypes.POINTER(ctypes.c_float),
                       ctypes.c_long,
                       ctypes.c_long,
                       ctypes.c_long,
                       ctypes.c_int32,
                       ctypes.c_int32]

cilacc.fdiff2D.argtypes = [ctypes.POINTER(ctypes.c_float),
                       ctypes.POINTER(ctypes.c_float),
                       ctypes.POINTER(ctypes.c_float),
                       ctypes.c_long,
                       ctypes.c_long,
                       ctypes.c_int32,
                       ctypes.c_int32]


class Gradient_C(LinearOperator):
    
    '''Finite Difference Operator:
            
            Computes first-order forward/backward differences 
                     on 2D, 3D, 4D ImageData
                     under Neumann/Periodic boundary conditions'''

    def __init__(self, gm_domain, gm_range=None, bnd_cond = NEUMANN):

        super(Gradient_C, self).__init__() 

        self.gm_domain = gm_domain
        self.gm_range = gm_range
        
        #default is 'Neumann'
        self.bnd_cond = 0
        
        if bnd_cond == PERIODIC:
            self.bnd_cond = 1
        
        # Domain Geometry = Range Geometry if not stated
        if self.gm_range is None:
            self.gm_range = BlockGeometry(*[gm_domain for _ in range(len(gm_domain.shape))])
        

        if len(gm_domain.shape) == 4:
            self.fd = cilacc.fdiff4D
        elif len(gm_domain.shape) == 3:
            self.fd = cilacc.fdiff3D
        elif len(gm_domain.shape) == 2:
            self.fd = cilacc.fdiff2D
        else:
            raise ValueError('Number of dimensions not supported, expected 2, 3 or 4, got {}'.format(len(gm_domain.shape)))
        
                        
    @staticmethod 
    def datacontainer_as_c_pointer(x):
        ndx = x.as_array()
        return ndx, ndx.ctypes.data_as(c_float_p)
        
    def direct(self, x, out=None):
        ndx , x_p = Gradient_C.datacontainer_as_c_pointer(x)
        
        return_val = False
        if out is None:
            out = self.gm_range.allocate(None)
            return_val = True

        arg1 = [Gradient_C.datacontainer_as_c_pointer(out.get_item(i))[1] for i in range(self.gm_range.shape[0])]
        arg2 = [el for el in x.shape]
        args = arg1 + arg2 + [self.bnd_cond, 1]
        self.fd(x_p, *args)
        
        if return_val is True:
            return out


    def adjoint(self, x, out=None):

        return_val = False
        if out is None:
            out = self.gm_domain.allocate(None)
            return_val = True

        ndout , out_p = Gradient_C.datacontainer_as_c_pointer(out)

        arg1 = [Gradient_C.datacontainer_as_c_pointer(x.get_item(i))[1] for i in range(self.gm_range.shape[0])]
        arg2 = [el for el in out.shape]
        args = arg1 + arg2 + [self.bnd_cond, 0]

        self.fd(out_p, *args)

        if return_val is True:
            return out

    def domain_geometry(self):
        
        '''Returns domain_geometry of Gradient'''
        
        return self.gm_domain
    
    def range_geometry(self):
        
        '''Returns range_geometry of Gradient'''
        
        return self.gm_range

            

       
if __name__ == '__main__':
    
    
    from ccpi.optimisation.operators import Identity, BlockOperator
    
    
    M, N = 20, 30
    ig = ImageGeometry(M, N)
    arr = ig.allocate('random_int' )
    
    # check direct of Gradient and sparse matrix
    G = Gradient(ig)
    norm1 = G.norm(iterations=300)
    print ("should be sqrt(8) {} {}".format(numpy.sqrt(8), norm1))
    G_sp = G.matrix()
    ig4 = ImageGeometry(M,N, channels=3)
    G4 = Gradient(ig4, correlation=Gradient.CORRELATION_SPACECHANNEL)
    norm4 = G4.norm(iterations=300)
    print ("should be sqrt(12) {} {}".format(numpy.sqrt(12), norm4))
    

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
#    c1 = arr + a
#    c2 = arr + a
#    c2 = a1 + arr
    
    from ccpi.framework import ImageGeometry
#    from ccpi.optimisation.operators import Gradient
#
    N, M = 2, 3
#    
    ig = ImageGeometry(N, M)
#
    G = Gradient(ig)
#
    u = G.domain_geometry().allocate('random_int')
    w = G.range_geometry().allocate('random_int')
    
    
    print( "################   without out #############")
          
    print( (G.direct(u)*w).sum(),  (u*G.adjoint(w)).sum() ) 
        

    print( "################   with out #############")
    
    res = G.range_geometry().allocate()
    res1 = G.domain_geometry().allocate()
    G.direct(u, out = res)          
    G.adjoint(w, out = res1)
          
    print( (res*w).sum(),  (u*res1).sum() )
    