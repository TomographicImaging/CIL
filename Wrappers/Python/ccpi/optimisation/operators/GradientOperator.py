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

from ccpi.optimisation.operators import LinearOperator
from ccpi.optimisation.operators import FiniteDiff, SparseFiniteDiff
from ccpi.framework import ImageGeometry, BlockGeometry
from ccpi.utilities import NUM_THREADS

import warnings

#default nThreads
# import multiprocessing
# cpus = multiprocessing.cpu_count()
# NUM_THREADS = max(int(cpus/2),1)

NEUMANN = 'Neumann'
PERIODIC = 'Periodic'
C = 'c'
NUMPY = 'numpy'
CORRELATION_SPACE = "Space"
CORRELATION_SPACECHANNEL = "SpaceChannels"

class Gradient(LinearOperator):


    r'''Gradient Operator: Computes first-order forward/backward differences on 
        2D, 3D, 4D ImageData under Neumann/Periodic boundary conditions
    
    :param gm_domain: Set up the domain of the function
    :type gm_domain: `ImageGeometry`
    :param bnd_cond: Set the boundary conditions to use 'Neumann' or 'Periodic', defaults to 'Neumann'
    :type bnd_cond: str, optional    
    
    :param \**kwargs:
        See below

    :Keyword Arguments:
        * *correlation* (``str``) --
          'Space' or 'SpaceChannels', defaults to 'Space'
        * *backend* (``str``) --
          'c' or 'numpy', defaults to 'c' if correlation is 'SpaceChannels' or channels = 1
        * *num_threads* (``int``) --
          If backend is 'c' specify the number of threads to use. Default is number of cpus/2          
                 
                 
        Example (2D): 
        .. math::
        
          \nabla : X -> Y \\
          u\in X, \nabla(u) = [\partial_{y} u, \partial_{x} u] \\
          u^{*}\in Y, \nabla^{*}(u^{*}) = \partial_{y} v1 + \partial_{x} v2
            

    '''

    #kept here for backwards compatability
    CORRELATION_SPACE = CORRELATION_SPACE
    CORRELATION_SPACECHANNEL = CORRELATION_SPACECHANNEL

    def __init__(self, domain_geometry, bnd_cond = 'Neumann', **kwargs):
        """Constructor method
        """        
        
        backend = kwargs.get('backend',C)

        correlation = kwargs.get('correlation',CORRELATION_SPACE)

        if correlation == CORRELATION_SPACE and domain_geometry.channels > 1:
            #numpy implementation only for now
            backend = NUMPY
            warnings.warn("Warning: correlation='Space' on multi-channel dataset will use `numpy` backend")

        if backend == NUMPY:
            self.operator = Gradient_numpy(domain_geometry, bnd_cond=bnd_cond, **kwargs)
        else:
            self.operator = Gradient_C(domain_geometry, bnd_cond=bnd_cond, **kwargs)
        
        super(Gradient, self).__init__(domain_geometry=domain_geometry, 
                                       range_geometry=self.operator.range_geometry()) 

        self.gm_range = self.range_geometry()
        self.gm_domain = self.domain_geometry()


    def direct(self, x, out=None):
        """Computes the first-order forward differences

        :param x: Image data
        :type x: `ImageData`
        :param out: pre-allocated output memory to store result
        :type out: `BlockDataContainer`, optional        
        :return: result data if not passed as parameter
        :rtype: `BlockDataContainer`
        """        
        return self.operator.direct(x, out=out)
        
        
    def adjoint(self, x, out=None):
        """Computes the first-order backward differences

        :param x: Gradient images for each dimension in ImageGeometry domain
        :type x: `BlockDataContainer`
        :param out: pre-allocated output memory to store result
        :type out: `ImageData`, optional      
        :return: result data if not passed as parameter
        :rtype: `ImageData`
        """            
        return self.operator.adjoint(x, out=out)

class Gradient_numpy(LinearOperator):
    
    def __init__(self, domain_geometry, method = 'forward', bnd_cond = 'Neumann', **kwargs):
        '''creator
        
        :param gm_domain: domain of the operator
        :type gm_domain: :code:`AcquisitionGeometry` or :code:`ImageGeometry`
        :param bnd_cond: boundary condition, either :code:`Neumann` or :code:`Periodic`.
        :type bnd_cond: str, optional, default :code:`Neumann`
        :param correlation: optional, :code:`SpaceChannel` or :code:`Space`
        :type correlation: str, optional, default :code:`Space`
        '''                
        
        self.size_dom_gm = len(domain_geometry.shape)         
        self.correlation = kwargs.get('correlation',CORRELATION_SPACE)        
        self.bnd_cond = bnd_cond 
        
        # Call FiniteDiff operator 
        self.method = method
        self.FD = FiniteDiff(domain_geometry, direction = 0, method = self.method, voxel_size = 1.0, bnd_cond = self.bnd_cond)
                
        
        if self.correlation==CORRELATION_SPACE:
            
            if domain_geometry.channels > 1:
                
                range_geometry = BlockGeometry(*[domain_geometry for _ in range(domain_geometry.length-1)] )

                if self.size_dom_gm == 4:
                    # 3D + Channel
                    # expected Grad_order = ['channels', 'direction_z', 'direction_y', 'direction_x']
                    expected_order = [ImageGeometry.CHANNEL, ImageGeometry.VERTICAL, ImageGeometry.HORIZONTAL_Y, ImageGeometry.HORIZONTAL_X]
                    self.voxel_size_order = [domain_geometry.voxel_size_z, domain_geometry.voxel_size_y, domain_geometry.voxel_size_x ]

                else:
                    # 2D + Channel
                    # expected Grad_order = ['channels', 'direction_y', 'direction_x']
                    expected_order = [ImageGeometry.CHANNEL, ImageGeometry.HORIZONTAL_Y, ImageGeometry.HORIZONTAL_X]
                    self.voxel_size_order = [domain_geometry.voxel_size_y, domain_geometry.voxel_size_x ]                    

                order = domain_geometry.get_order_by_label(domain_geometry.dimension_labels, expected_order)
                
                self.ind = order[1:]
                
                #self.ind = numpy.arange(1,self.gm_domain.length)
            else:
                # no channel info
                range_geometry = BlockGeometry(*[domain_geometry for _ in range(domain_geometry.length) ] )
                if self.size_dom_gm == 3:
                    # 3D
                    # expected Grad_order = ['direction_z', 'direction_y', 'direction_x']
                    expected_order = [ImageGeometry.VERTICAL, ImageGeometry.HORIZONTAL_Y, ImageGeometry.HORIZONTAL_X]
                    self.voxel_size_order = [domain_geometry.voxel_size_z, domain_geometry.voxel_size_y, domain_geometry.voxel_size_x ]                    
                    
                else:
                    # 2D
                    expected_order = [ImageGeometry.HORIZONTAL_Y, ImageGeometry.HORIZONTAL_X]    
                    self.voxel_size_order = [domain_geometry.voxel_size_y, domain_geometry.voxel_size_x ]                     

                self.ind = domain_geometry.get_order_by_label(domain_geometry.dimension_labels, expected_order)
                # self.ind = numpy.arange(self.gm_domain.length)
                
        elif self.correlation==CORRELATION_SPACECHANNEL:
            
            if domain_geometry.channels > 1:
                range_geometry = BlockGeometry(*[domain_geometry for _ in range(domain_geometry.length)])
                self.ind = range(domain_geometry.length)                
                if self.size_dom_gm == 4:
                    # Voxel size wrt to channel direction == 1.0
                    self.voxel_size_order = [1.0, domain_geometry.voxel_size_z, domain_geometry.voxel_size_y, domain_geometry.voxel_size_x ]                 
                elif self.size_dom_gm == 3:
                    # Voxel size wrt to channel direction == 1.0
                    self.voxel_size_order = [1.0, domain_geometry.voxel_size_y, domain_geometry.voxel_size_x ]                     
            else:
                raise ValueError('No channels to correlate')
         
        super(Gradient_numpy, self).__init__(domain_geometry = domain_geometry, 
                                             range_geometry = range_geometry) 
        
        print("Initialised GradientOperator with numpy backend")               
        
    def direct(self, x, out=None):
        
                
        if out is not None:
            
            for i in range(self.range_geometry().shape[0]):
                self.FD.direction = self.ind[i]
                self.FD.voxel_size = self.voxel_size_order[i]
                self.FD.direct(x, out = out[i])
        else:
            tmp = self.range_geometry().allocate()        
            for i in range(tmp.shape[0]):
                self.FD.direction = self.ind[i]
                self.FD.voxel_size = self.voxel_size_order[i]
                tmp.get_item(i).fill(self.FD.direct(x))
            return tmp    
        
    def adjoint(self, x, out=None):
        
        if out is not None:

            tmp = self.domain_geometry().allocate()            
            for i in range(x.shape[0]):
                self.FD.direction=self.ind[i] 
                self.FD.voxel_size = self.voxel_size_order[i]
                self.FD.adjoint(x.get_item(i), out = tmp)
                if i == 0:
                    out.fill(tmp)
                else:
                    out += tmp
        else:            
            tmp = self.domain_geometry().allocate()
            for i in range(x.shape[0]):
                self.FD.direction=self.ind[i]
                self.FD.voxel_size = self.voxel_size_order[i]
                tmp += self.FD.adjoint(x.get_item(i))
            return tmp    
               
    ###########################################################################
    ###############  For preconditioning ######################################
    ###########################################################################
#    def matrix(self):
#        
#        tmp = self.gm_range.allocate()
#        
#        mat = []
#        for i in range(tmp.shape[0]):
#            
#            spMat = SparseFiniteDiff(self.gm_domain, direction=self.ind[i], bnd_cond=self.bnd_cond)
#            mat.append(spMat.matrix())
#    
#        return BlockDataContainer(*mat)    
#
#
#    def sum_abs_col(self):
#        
#        tmp = self.gm_range.allocate()
#        res = self.gm_domain.allocate()
#        for i in range(tmp.shape[0]):
#            spMat = SparseFiniteDiff(self.gm_domain, direction=self.ind[i], bnd_cond=self.bnd_cond)
#            res += spMat.sum_abs_row()
#        return res
#    
#    def sum_abs_row(self):
#        
#        tmp = self.gm_range.allocate()
#        res = []
#        for i in range(tmp.shape[0]):
#            spMat = SparseFiniteDiff(self.gm_domain, direction=self.ind[i], bnd_cond=self.bnd_cond)
#            res.append(spMat.sum_abs_col())
#        return BlockDataContainer(*res)


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


cilacc = ctypes.cdll.LoadLibrary(dll)

c_float_p = ctypes.POINTER(ctypes.c_float)

cilacc.openMPtest.restypes = ctypes.c_int32
cilacc.openMPtest.argtypes = [ctypes.c_int32]

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
                       ctypes.c_int32,
                       ctypes.c_int32]

cilacc.fdiff2D.argtypes = [ctypes.POINTER(ctypes.c_float),
                       ctypes.POINTER(ctypes.c_float),
                       ctypes.POINTER(ctypes.c_float),
                       ctypes.c_long,
                       ctypes.c_long,
                       ctypes.c_int32,
                       ctypes.c_int32,
                       ctypes.c_int32]


class Gradient_C(LinearOperator):
    
    '''Finite Difference Operator:
            
            Computes first-order forward/backward differences 
                     on 2D, 3D, 4D ImageData
                     under Neumann/Periodic boundary conditions'''

    def __init__(self, gm_domain, gm_range=None, bnd_cond = NEUMANN, **kwargs):

        self.num_threads = kwargs.get('num_threads',NUM_THREADS)

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
            # Voxel size wrt to channel direction == 1.0
            self.fd = cilacc.fdiff4D
            self.voxel_size_order = [1.0, gm_domain.voxel_size_z, gm_domain.voxel_size_y, gm_domain.voxel_size_x ]            
        elif len(gm_domain.shape) == 3:
            # Voxel size wrt to channel direction == 1.0
            self.fd = cilacc.fdiff3D
            self.voxel_size_order = [gm_domain.voxel_size_z, gm_domain.voxel_size_y, gm_domain.voxel_size_x ]            
        elif len(gm_domain.shape) == 2:
            self.fd = cilacc.fdiff2D
            # Voxel size wrt to channel direction == 1.0
            self.voxel_size_order = [gm_domain.voxel_size_y, gm_domain.voxel_size_x ]            
        else:
            raise ValueError('Number of dimensions not supported, expected 2, 3 or 4, got {}'.format(len(gm_domain.shape)))
      #self.num_threads
        # super(Gradient_C, self).__init__() 
        super(Gradient_C, self).__init__(domain_geometry=self.gm_domain, 
                                             range_geometry=self.gm_range) 
        print("Initialised GradientOperator with C backend running with ", cilacc.openMPtest(self.num_threads)," threads")               

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

        #pass list of all arguments
        arg1 = [Gradient_C.datacontainer_as_c_pointer(out.get_item(i))[1] for i in range(self.gm_range.shape[0])]
        arg2 = [el for el in x.shape]
        args = arg1 + arg2 + [self.bnd_cond, 1, self.num_threads]
        self.fd(x_p, *args)
#        out /= self.voxel_size_order
        
        if return_val is True:
            return out

    def adjoint(self, x, out=None):

        return_val = False
        if out is None:
            out = self.gm_domain.allocate(None)
            return_val = True

        ndout, out_p = Gradient_C.datacontainer_as_c_pointer(out)

        arg1 = [Gradient_C.datacontainer_as_c_pointer(x.get_item(i))[1] for i in range(self.gm_range.shape[0])]
        arg2 = [el for el in out.shape]
        args = arg1 + arg2 + [self.bnd_cond, 0, self.num_threads]

        self.fd(out_p, *args)
#        out /= self.voxel_size_order

        if return_val is True:
            return out

       
if __name__ == '__main__':
    
    
    ig = ImageGeometry(3, 4)
    x = ig.allocate('random_int', max_value = 10)
    
    G_numpy = Gradient(ig, method = 'forward', bnd_cond = 'Neumann', backend = 'numpy')    
    print(G_numpy.dot_test(G_numpy))
    
    G_numpy = Gradient(ig, method = 'backward', bnd_cond = 'Neumann', backend = 'numpy')    
    print(G_numpy.dot_test(G_numpy))   
        
    G_numpy = Gradient(ig, method = 'centered', bnd_cond = 'Neumann', backend = 'numpy')    
    print(G_numpy.dot_test(G_numpy))    
    
    G_numpy = Gradient(ig, method = 'forward', bnd_cond = 'Periodic', backend = 'numpy')    
    print(G_numpy.dot_test(G_numpy))
    
    G_numpy = Gradient(ig, method = 'backward', bnd_cond = 'Periodic', backend = 'numpy')    
    print(G_numpy.dot_test(G_numpy))
    
    G_numpy = Gradient(ig, method = 'centered', bnd_cond = 'Periodic', backend = 'numpy')    
    print(G_numpy.dot_test(G_numpy))    
    
        
#    ig = ImageGeometry(voxel_num_x = 3, voxel_num_y = 4, voxel_size_x = 15, voxel_size_y = 12, channels = 2)
#    x = ig.allocate('random_int', max_value = 10, seed = 10)  
#    G_numpy = Gradient(ig, method = 'forward', bnd_cond = 'Neumann', backend = 'numpy')
#    print(G_numpy.direct(x).get_item(0).as_array())
    
    ig = ImageGeometry(voxel_num_x = 3, voxel_num_y = 4, voxel_size_x = 0.5, voxel_size_y = 0.2)
    x = ig.allocate('random', seed = 10)   
    
    G_np = Gradient(ig, method = 'forward', bnd_cond = 'Neumann', backend = 'numpy')
    G_c = Gradient(ig, method = 'forward', bnd_cond = 'Neumann', backend = 'c')
#    
#    res1a = G_np.direct(x)
#    print(res1a.get_item(0).as_array())   
#    r
#    res2a = G_c.direct(x)
#    print(res2a.get_item(0).as_array()) 
#
#    res1b = G_np.adjoint(res1a)
#    print(res1b.as_array())   
#
#    res2b = G_c.adjoint(res2a)
#    print(res2b.as_array())       
    
    
    
    
    
    
#    import numpy as np
#    nc, nz, ny, nx = 3, 4, 5, 6
#    size = nc * nz * ny * nx
#    dim = [nc, nz, ny, nx]
#
#    ig = ImageGeometry(voxel_num_x=nx, voxel_num_y=ny, voxel_num_z=nz, channels=nc)
#
#    arr = np.arange(size).reshape(dim).astype(np.float32)**2
#
#    data = ig.allocate()
#    data.fill(arr)
##
##    #neumann
#    grad_py = Gradient(ig, bnd_cond='Neumann', method = 'forward', correlation="SpaceChannels", backend='numpy')
#    gold_direct = grad_py.direct(data)
#    gold_adjoint = grad_py.adjoint(gold_direct)    
    
    
    
    
    
    
    
#    G_numpy = Gradient(ig, method = 'backward', correlation = "SpaceChannels", bnd_cond = 'Periodic', backend = 'numpy')    
#    print(G_numpy.norm())   
    
#    print(G_numpy.direct(x).get_item(0).as_array())


    
#    
#    G_numpy = Gradient(ig, method = 'backward', bnd_cond = 'Periodic', backend = 'numpy')    
#    print(G_numpy.dot_test(G_numpy)) 
#
#
#    ig = ImageGeometry(3, 4, channels = 10, voxel_size_x = 10, voxel_size_y = 10 )
#    x = ig.allocate('random_int', max_value = 10)
#    
#    G_numpy = Gradient(ig, method = 'forward', correlation = 'SpaceChannels', bnd_cond = 'Neumann', backend = 'numpy')    
#    print(G_numpy.dot_test(G_numpy), G_numpy)
#    
#    G_numpy = Gradient(ig, method = 'forward', correlation = 'SpaceChannels', bnd_cond = 'Periodic', backend = 'numpy')    
#    print(G_numpy.dot_test(G_numpy), G_numpy)
    
    
    
        
#    G_numpy = Gradient(ig, method = 'backward', boundary_condition = 'Neumann', backend = 'numpy')    
#    print(G_numpy.dot_test(G_numpy))   
#    
#    G_numpy = Gradient(ig, method = 'forward', boundary_condition = 'Periodic', backend = 'numpy')    
#    print(G_numpy.dot_test(G_numpy))
#    
#    G_numpy = Gradient(ig, method = 'backward', boundary_condition = 'Periodic', backend = 'numpy')    
#    print(G_numpy.dot_test(G_numpy))       
#    
    
#    res1 = G_numpy.direct(x)
#    
#    print(res1.get_item(0).as_array())
#    print(res1.get_item(1).as_array())
#    
#    G = Gradient_C(ig)
#    
#    res2 = G.direct(x)
#    
#    print(res2.get_item(0).as_array())
#    print(res2.get_item(1).as_array())    
#    
#    print(G_numpy.dot_test(G_numpy))
    
    
    
    
#    arr = ig.allocate('random_int' )
#    
#    # check direct of Gradient and sparse matrix
#    G = Gradient_numpy(ig, method = 'forward')
#    norm1 = G.norm(iterations=300)
#    print ("should be sqrt(8) {} {}".format(numpy.sqrt(8), norm1))
#    G_sp = G.matrix()
#    ig4 = ImageGeometry(M,N, channels=3)
#    G4 = Gradient(ig4, correlation=Gradient.CORRELATION_SPACECHANNEL)
#    norm4 = G4.norm(iterations=300)
#    print ("should be sqrt(12) {} {}".format(numpy.sqrt(12), norm4))
#    
#
#    res1 = G.direct(arr)
#    res1y = numpy.reshape(G_sp[0].toarray().dot(arr.as_array().flatten('F')), ig.shape, 'F')
#    
#    print(res1[0].as_array())
#    print(res1y)
#    
#    res1x = numpy.reshape(G_sp[1].toarray().dot(arr.as_array().flatten('F')), ig.shape, 'F')
#    
#    print(res1[1].as_array())
#    print(res1x)    
#    
#    #check sum abs row
#    conc_spmat = numpy.abs(numpy.concatenate( (G_sp[0].toarray(), G_sp[1].toarray() )))
#    print(numpy.reshape(conc_spmat.sum(axis=0), ig.shape, 'F'))    
#    print(G.sum_abs_row().as_array())
#    
#    print(numpy.reshape(conc_spmat.sum(axis=1), ((2,) + ig.shape), 'F'))
#    
#    print(G.sum_abs_col()[0].as_array())
#    print(G.sum_abs_col()[1].as_array())   
#    
#    # Check Blockoperator sum abs col and row
#    
#    op1 = Gradient(ig)
#    op2 = Identity(ig)
#    
#    B = BlockOperator( op1, op2)
#    
#    Brow = B.sum_abs_row()
#    Bcol = B.sum_abs_col()
#    
#    concB = numpy.concatenate( (numpy.abs(numpy.concatenate( (G_sp[0].toarray(), G_sp[1].toarray() ))), op2.matrix().toarray()))
#    
#    print(numpy.reshape(concB.sum(axis=0), ig.shape, 'F'))
#    print(Brow.as_array())
#    
#    print(numpy.reshape(concB.sum(axis=1)[0:12], ((2,) + ig.shape), 'F'))
#    print(Bcol[1].as_array())    
#    
#        
##    print(numpy.concatene(G_sp[0].toarray()+ ))
##    print(G_sp[1].toarray())
##    
##    d1 = G.sum_abs_row()
##    print(d1.as_array())
##    
##    d2 = G_neum.sum_abs_col()
###    print(d2)    
##    
##    
##    ###########################################################
#    a = BlockDataContainer( BlockDataContainer(arr, arr), arr)
#    b = BlockDataContainer( BlockDataContainer(arr+5, arr+3), arr+2)
#    c = a/b
#    
#    print(c[0][0].as_array(), (arr/(arr+5)).as_array())
#    print(c[0][1].as_array(), (arr/(arr+3)).as_array())
#    print(c[1].as_array(), (arr/(arr+2)).as_array())
#    
#    
#    a1 = BlockDataContainer( arr, BlockDataContainer(arr, arr))
##    
##    c1 = arr + a
##    c2 = arr + a
##    c2 = a1 + arr
#    
#    from ccpi.framework import ImageGeometry
##    from ccpi.optimisation.operators import Gradient
##
#    N, M = 2, 3
##    
#    ig = ImageGeometry(N, M)
##
#    G = Gradient(ig)
##
#    u = G.domain_geometry().allocate('random_int')
#    w = G.range_geometry().allocate('random_int')
#    
#    
#    print( "################   without out #############")
#          
#    print( (G.direct(u)*w).sum(),  (u*G.adjoint(w)).sum() ) 
#        
#
#    print( "################   with out #############")
#    
#    res = G.range_geometry().allocate()
#    res1 = G.domain_geometry().allocate()
#    G.direct(u, out = res)          
#    G.adjoint(w, out = res1)
#          
#    print( (res*w).sum(),  (u*res1).sum() )
#    
#    

#    
