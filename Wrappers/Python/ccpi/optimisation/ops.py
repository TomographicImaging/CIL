# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library developed by
#   Visual Analytics and Imaging System Group of the Science Technology
#   Facilities Council, STFC

#   Copyright 2018 Jakob Jorgensen, Daniil Kazantsev and Edoardo Pasca

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy
from scipy.sparse.linalg import svds
from ccpi.framework import DataContainer
from ccpi.framework import AcquisitionData
from ccpi.framework import ImageData
from ccpi.framework import ImageGeometry
from ccpi.framework import AcquisitionGeometry
from numbers import Number
# Maybe operators need to know what types they take as inputs/outputs
# to not just use generic DataContainer


class Operator(object):
    '''Operator that maps from a space X -> Y'''
    def __init__(self, **kwargs):
        self.scalar = 1
    def is_linear(self):
        '''Returns if the operator is linear'''
        return False
    def direct(self,x, out=None):
        raise NotImplementedError
    def size(self):
        # To be defined for specific class
        raise NotImplementedError
    def norm(self):
        raise NotImplementedError
    def allocate_direct(self):
        '''Allocates memory on the Y space'''
        raise NotImplementedError
    def allocate_adjoint(self):
        '''Allocates memory on the X space'''
        raise NotImplementedError
    def range_dim(self):
        raise NotImplementedError
    def domain_dim(self):
        raise NotImplementedError
    def __rmul__(self, other):
        '''reverse multiplication of Operator with number sets the variable scalar in the Operator'''
        assert isinstance(other, Number)
        self.scalar = other
        return self

class LinearOperator(Operator):
    '''Operator that maps from a space X -> Y'''
    def is_linear(self):
        '''Returns if the operator is linear'''
        return True
    def adjoint(self,x, out=None):
        raise NotImplementedError
        
class Identity(Operator):
    def __init__(self):
        self.s1 = 1.0
        self.L = 1
        super(Identity, self).__init__()
        
    def direct(self,x,out=None):
        if out is None:
            return x.copy()
        else:
            out.fill(x)
    
    def adjoint(self,x, out=None):
        if out is None:
            return x.copy()
        else:
            out.fill(x)
    
    def size(self):
        return NotImplemented
    
    def get_max_sing_val(self):
        return self.s1

class TomoIdentity(Operator):
    def __init__(self, geometry, **kwargs):
        self.s1 = 1.0
        self.geometry = geometry
        super(TomoIdentity, self).__init__()
        
    def direct(self,x,out=None):
        
        if out is None:
            if self.scalar != 1:
                return x * self.scalar
            return x.copy()
        else:
            if self.scalar != 1:
                out.fill(x * self.scalar)
                return
            out.fill(x)
            return
    
    def adjoint(self,x, out=None):
        return self.direct(x, out)
    
    def size(self):
        return NotImplemented
    
    def get_max_sing_val(self):
        return self.s1
    def allocate_direct(self):
        if issubclass(type(self.geometry), ImageGeometry):
            return ImageData(geometry=self.geometry)
        elif issubclass(type(self.geometry), AcquisitionGeometry):
            return AcquisitionData(geometry=self.geometry)
        else:
            raise ValueError("Wrong geometry type: expected ImageGeometry of AcquisitionGeometry, got ", type(self.geometry))
    def allocate_adjoint(self):
        return self.allocate_direct()
    
    

class FiniteDiff2D(Operator):
    def __init__(self):
        self.s1 = 8.0
        super(FiniteDiff2D, self).__init__()
        
    def direct(self,x, out=None):
        '''Forward differences with Neumann BC.'''
        # FIXME this seems to be working only with numpy arrays
        
        d1 = numpy.zeros_like(x.as_array())
        d1[:,:-1] = x.as_array()[:,1:] - x.as_array()[:,:-1]
        d2 = numpy.zeros_like(x.as_array())
        d2[:-1,:] = x.as_array()[1:,:] - x.as_array()[:-1,:]
        d = numpy.stack((d1,d2),0)
        #x.geometry.voxel_num_z = 2
        return type(x)(d,False,geometry=x.geometry)
    
    def adjoint(self,x, out=None):
        '''Backward differences, Neumann BC.'''
        Nrows = x.get_dimension_size('horizontal_x')
        Ncols = x.get_dimension_size('horizontal_y')
        Nchannels = 1
        if len(x.shape) == 4:
            Nchannels = x.get_dimension_size('channel')
        zer = numpy.zeros((Nrows,1))
        xxx = x.as_array()[0,:,:-1]
        #
        h = numpy.concatenate((zer,xxx), 1) 
        h -= numpy.concatenate((xxx,zer), 1)
        
        zer = numpy.zeros((1,Ncols))
        xxx = x.as_array()[1,:-1,:]
        #
        v  = numpy.concatenate((zer,xxx), 0) 
        v -= numpy.concatenate((xxx,zer), 0)
        return type(x)(h + v, False, geometry=x.geometry)
    
    def size(self):
        return NotImplemented
    
    def get_max_sing_val(self):
        return self.s1

def PowerMethodNonsquareOld(op,numiters):
    # Initialise random
    # Jakob's
    #inputsize = op.size()[1]
    #x0 = ImageContainer(numpy.random.randn(*inputsize)
    # Edo's
    #vg = ImageGeometry(voxel_num_x=inputsize[0],
    #                   voxel_num_y=inputsize[1], 
    #                   voxel_num_z=inputsize[2])
    #
    #x0 = ImageData(geometry = vg, dimension_labels=['vertical','horizontal_y','horizontal_x'])
    #print (x0)
    #x0.fill(numpy.random.randn(*x0.shape))
    
    x0 = op.create_image_data()
    
    s = numpy.zeros(numiters)
    # Loop
    for it in numpy.arange(numiters):
        x1 = op.adjoint(op.direct(x0))
        x1norm = numpy.sqrt((x1**2).sum())
        #print ("x0 **********" ,x0)
        #print ("x1 **********" ,x1)
        s[it] = (x1*x0).sum() / (x0*x0).sum()
        x0 = (1.0/x1norm)*x1
    return numpy.sqrt(s[-1]), numpy.sqrt(s), x0

#def PowerMethod(op,numiters):
#    # Initialise random
#    x0 = np.random.randn(400)
#    s = np.zeros(numiters)
#    # Loop
#    for it in np.arange(numiters):
#        x1 = np.dot(op.transpose(),np.dot(op,x0))
#        x1norm = np.sqrt(np.sum(np.dot(x1,x1)))
#        s[it] = np.dot(x1,x0) / np.dot(x1,x0)
#        x0 = (1.0/x1norm)*x1
#    return s, x0
    

def PowerMethodNonsquare(op,numiters , x0=None):
    # Initialise random
    # Jakob's
    # inputsize , outputsize = op.size()
    #x0 = ImageContainer(numpy.random.randn(*inputsize)
    # Edo's
    #vg = ImageGeometry(voxel_num_x=inputsize[0],
    #                   voxel_num_y=inputsize[1], 
    #                   voxel_num_z=inputsize[2])
    #
    #x0 = ImageData(geometry = vg, dimension_labels=['vertical','horizontal_y','horizontal_x'])
    #print (x0)
    #x0.fill(numpy.random.randn(*x0.shape))
    
    if x0 is None:
        #x0 = op.create_image_data()
        x0 = op.allocate_direct()
        x0.fill(numpy.random.randn(*x0.shape))
    
    s = numpy.zeros(numiters)
    # Loop
    for it in numpy.arange(numiters):
        x1 = op.adjoint(op.direct(x0))
        #x1norm = numpy.sqrt((x1*x1).sum())
        x1norm = x1.norm()
        #print ("x0 **********" ,x0)
        #print ("x1 **********" ,x1)
        s[it] = (x1*x0).sum() / (x0.squared_norm())
        x0 = (1.0/x1norm)*x1
    return numpy.sqrt(s[-1]), numpy.sqrt(s), x0

class LinearOperatorMatrix(Operator):
    def __init__(self,A):
        self.A = A
        self.s1 = None   # Largest singular value, initially unknown
        super(LinearOperatorMatrix, self).__init__()
        
    def direct(self,x, out=None):
        if out is None:
            return type(x)(numpy.dot(self.A,x.as_array()))
        else:
            numpy.dot(self.A, x.as_array(), out=out.as_array())
            
    
    def adjoint(self,x, out=None):
        if out is None:
            return type(x)(numpy.dot(self.A.transpose(),x.as_array()))
        else:
            numpy.dot(self.A.transpose(),x.as_array(), out=out.as_array())
            
    
    def size(self):
        return self.A.shape
    
    def get_max_sing_val(self):
        # If unknown, compute and store. If known, simply return it.
        if self.s1 is None:
            self.s1 = svds(self.A,1,return_singular_vectors=False)[0]
            return self.s1
        else:
            return self.s1
    def allocate_direct(self):
        '''allocates the memory to hold the result of adjoint'''
        #numpy.dot(self.A.transpose(),x.as_array())
        M_A, N_A = self.A.shape
        out = numpy.zeros((N_A,1))
        return DataContainer(out)
    def allocate_adjoint(self):
        '''allocate the memory to hold the result of direct'''
        #numpy.dot(self.A.transpose(),x.as_array())
        M_A, N_A = self.A.shape
        out = numpy.zeros((M_A,1))
        return DataContainer(out)
