# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy
from ccpi.framework import DataContainer, ImageData, ImageGeometry, AcquisitionData
from ccpi.astra.processors import AstraForwardProjector, AstraBackProjector
from ccpi.optimisation.ops import PowerMethodNonsquare
from Operators.operators import Operator
from numbers import Number
import functools
#from operators import *
###############################################################################


#class Operator(object):
#    '''Operator that maps from a space X -> Y'''
#    def __init__(self, **kwargs):
#        self.scalar = 1
#    def is_linear(self):
#        '''Returns if the operator is linear'''
#        return False
#    def direct(self,x, out=None):
#        raise NotImplementedError
#    def size(self):
#        # To be defined for specific class
#        raise NotImplementedError
#    def norm(self):
#        raise NotImplementedError
#    def allocate_direct(self):
#        '''Allocates memory on the Y space'''
#        raise NotImplementedError
#    def allocate_adjoint(self):
#        '''Allocates memory on the X space'''
#        raise NotImplementedError
#    def range_dim(self):
#        raise NotImplementedError
#    def domain_dim(self):
#        raise NotImplementedError
#    def __rmul__(self, other):
#        assert isinstance(other, Number)
#        self.scalar = other
#        return self    
#    
#class LinearOperator(Operator):
#    '''Operator that maps from a space X -> Y'''
#    def is_linear(self):
#        '''Returns if the operator is linear'''
#        return True
#    def adjoint(self,x, out=None):
#        raise NotImplementedError   
        
class CompositeDataContainer(object):
    '''Class to hold a composite operator'''
    __array_priority__ = 1
    def __init__(self, *args, shape=None):
        '''containers must be passed row by row'''
        self.containers = args
        self.index = 0
        if shape is None:
            shape = (len(args),1)
        self.shape = shape
        n_elements = functools.reduce(lambda x,y: x*y, shape, 1)
        if len(args) != n_elements:
            raise ValueError(
                    'Dimension and size do not match: expected {} got {}'
                    .format(n_elements,len(args)))
#        for i in range(shape[0]):
#            b.append([])
#            for j in range(shape[1]):
#                b[-1].append(args[i*shape[1]+j])
#                indices.append(i*shape[1]+j)
#        self.containers = b
        
    def __iter__(self):
        return self
    def next(self):
        '''python2 backwards compatibility'''
        return self.__next__()
    def __next__(self):
        try:
            out = self[self.index]
        except IndexError as ie:
            raise StopIteration()
        self.index+=1
        return out
    
    def is_compatible(self, other):
        '''basic check if the size of the 2 objects fit'''
        if isinstance(other, Number):
            return True   
        elif isinstance(other, list):
            # TODO look elements should be numbers
            for ot in other:
                if not isinstance(ot, (Number,\
                                 numpy.int, numpy.int8, numpy.int16, numpy.int32, numpy.int64,\
                                 numpy.float, numpy.float16, numpy.float32, numpy.float64, \
                                 numpy.complex)):
                    raise ValueError('List/ numpy array can only contain numbers {}'\
                                     .format(type(ot)))
            return len(self.containers) == len(other)
        elif isinstance(other, numpy.ndarray):
            return self.shape == other.shape
        return len(self.containers) == len(other.containers)
    def get_item(self, row, col=0):
        if row > self.shape[0]:
            raise ValueError('Requested row {} > max {}'.format(row, self.shape[0]))
        if col > self.shape[1]:
            raise ValueError('Requested col {} > max {}'.format(col, self.shape[1]))
        
        index = row*self.shape[1]+col
        return self.containers[index]
                
    def add(self, other, out=None, *args, **kwargs):
        assert self.is_compatible(other)
        if isinstance(other, Number):
            return type(self)(*[ el.add(other, out, *args, **kwargs) for el in self.containers])
        elif isinstance(other, list) or isinstance(other, numpy.ndarray):
            return type(self)(*[ el.add(ot, out, *args, **kwargs) for el,ot in zip(self.containers,other)])
        return type(self)(*[ el.add(ot, out, *args, **kwargs) for el,ot in zip(self.containers,other.containers)])
        
    def subtract(self, other, out=None , *args, **kwargs):
        assert self.is_compatible(other)
        if isinstance(other, Number):
            return type(self)(*[ el.subtract(other, out, *args, **kwargs) for el in self.containers])
        elif isinstance(other, list) or isinstance(other, numpy.ndarray):
            return type(self)(*[ el.subtract(ot, out, *args, **kwargs) for el,ot in zip(self.containers,other)])
        return type(self)(*[ el.subtract(ot, out, *args, **kwargs) for el,ot in zip(self.containers,other.containers)])

    def multiply(self, other , out=None, *args, **kwargs):
        self.is_compatible(other)
        if isinstance(other, Number):
            return type(self)(*[ el.multiply(other, out, *args, **kwargs) for el in self.containers])
        elif isinstance(other, list):
            return type(self)(*[ el.multiply(ot, out, *args, **kwargs) for el,ot in zip(self.containers,other)])
        elif isinstance(other, numpy.ndarray):           
            return type(self)(*[ el.multiply(ot, out, *args, **kwargs) for el,ot in zip(self.containers,other)])
        return type(self)(*[ el.multiply(ot, out, *args, **kwargs) for el,ot in zip(self.containers,other.containers)])
    
    def divide(self, other , out=None ,*args, **kwargs):
        self.is_compatible(other)
        if isinstance(other, Number):
            return type(self)(*[ el.divide(other, out, *args, **kwargs) for el in self.containers])
        elif isinstance(other, list) or isinstance(other, numpy.ndarray):
            return type(self)(*[ el.divide(ot, out, *args, **kwargs) for el,ot in zip(self.containers,other)])
        return type(self)(*[ el.divide(ot, out, *args, **kwargs) for el,ot in zip(self.containers,other.containers)])
    
    def power(self, other , out=None, *args, **kwargs):
        assert self.is_compatible(other)
        if isinstance(other, Number):
            return type(self)(*[ el.power(other, out, *args, **kwargs) for el in self.containers])
        elif isinstance(other, list) or isinstance(other, numpy.ndarray):
            return type(self)(*[ el.power(ot, out, *args, **kwargs) for el,ot in zip(self.containers,other)])
        return type(self)(*[ el.power(ot, out, *args, **kwargs) for el,ot in zip(self.containers,other.containers)])
    
    def maximum(self,other, out=None, *args, **kwargs):
        assert self.is_compatible(other)
        if isinstance(other, Number):
            return type(self)(*[ el.maximum(other, out, *args, **kwargs) for el in self.containers])
        elif isinstance(other, list) or isinstance(other, numpy.ndarray):
            return type(self)(*[ el.maximum(ot, out, *args, **kwargs) for el,ot in zip(self.containers,other)])
        return type(self)(*[ el.maximum(ot, out, *args, **kwargs) for el,ot in zip(self.containers,other.containers)])
    
    ## unary operations    
    def abs(self, out=None, *args,  **kwargs):
        return type(self)(*[ el.abs(out, *args, **kwargs) for el in self.containers]) 
    def sign(self, out=None, *args,  **kwargs):
        return type(self)(*[ el.sign(out, *args, **kwargs) for el in self.containers])
    def sqrt(self, out=None, *args,  **kwargs):
        return type(self)(*[ el.sqrt(out, *args, **kwargs) for el in self.containers])
    def conjugate(self, out=None):
        return type(self)(*[el.conjugate() for el in self.containers])
    
    ## reductions
    def sum(self, out=None, *args, **kwargs):
        return numpy.asarray([ el.sum(*args, **kwargs) for el in self.containers])
    def norm(self):
        y = numpy.asarray([el**2 for el in self.containers])
        return y.sum()    
    def copy(self):
        '''alias of clone'''    
        return self.clone()
    def clone(self):
        return type(self)(*[el.copy() for el in self.containers])
    
    def __add__(self, other):
        return self.add( other )
    # __radd__
    
    def __sub__(self, other):
        return self.subtract( other )
    # __rsub__
    
    def __mul__(self, other):
        return self.multiply(other)
    # __rmul__
    
    def __div__(self, other):
        return self.divide(other)
    # __rdiv__
    def __truediv__(self, other):
        return self.divide(other)
    
    def __pow__(self, other):
        return self.power(other)
    # reverse operand
    def __radd__(self, other):
        return self + other
    # __radd__
    
    def __rsub__(self, other):
        return (-1 * self) + other
    # __rsub__
    
    def __rmul__(self, other):
        '''Reverse multiplication
        
        to make sure that this method is called rather than the __mul__ of a numpy array
        the class constant __array_priority__ must be set > 0
        https://docs.scipy.org/doc/numpy-1.15.1/reference/arrays.classes.html#numpy.class.__array_priority__
        '''
        return self * other
    # __rmul__
    
    def __rdiv__(self, other):
        return pow(self / other, -1)
    # __rdiv__
    def __rtruediv__(self, other):
        return self.__rdiv__(other)
    
    def __rpow__(self, other):
        return other.power(self)
    
    def __iadd__(self, other):
        if isinstance (other, CompositeDataContainer):
            for el,ot in zip(self.containers, other.containers):
                el += ot
        elif isinstance(other, Number):
            for el in self.containers:
                el += other
        elif isinstance(other, list) or isinstance(other, numpy.ndarray):
            self.is_compatible(other)
            for el,ot in zip(self.containers, other):
                el += ot
        return self
    # __radd__
    
    def __isub__(self, other):
        if isinstance (other, CompositeDataContainer):
            for el,ot in zip(self.containers, other.containers):
                el -= ot
        elif isinstance(other, Number):
            for el in self.containers:
                el -= other
        elif isinstance(other, list) or isinstance(other, numpy.ndarray):
            assert self.is_compatible(other)
            for el,ot in zip(self.containers, other):
                el -= ot
        return self
    # __rsub__
    
    def __imul__(self, other):
        if isinstance (other, CompositeDataContainer):
            for el,ot in zip(self.containers, other.containers):
                el *= ot
        elif isinstance(other, Number):
            for el in self.containers:
                el *= other
        elif isinstance(other, list) or isinstance(other, numpy.ndarray):
            assert self.is_compatible(other)
            for el,ot in zip(self.containers, other):
                el *= ot
        return self
    # __imul__
    
    def __idiv__(self, other):
        if isinstance (other, CompositeDataContainer):
            for el,ot in zip(self.containers, other.containers):
                el /= ot
        elif isinstance(other, Number):
            for el in self.containers:
                el /= other
        elif isinstance(other, list) or isinstance(other, numpy.ndarray):
            assert self.is_compatible(other)
            for el,ot in zip(self.containers, other):
                el /= ot
        return self
    # __rdiv__
    def __itruediv__(self, other):
        return self.__idiv__(other)
                                            
class CompositeOperator(Operator):
    
    def __init__(self, shape, *operators):
        self.shape = shape
        self.operators = operators
                
        n = self.shape[0] * self.shape[1]
        assert n == len (self.operators)
        self.compMat = [[ None for i in range(self.shape[1])] for j in range(self.shape[0])]
        s = 0
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.compMat[i][j] = self.operators[s]
                s +=1
                                        
    def opMatrix(self):
        
        n = self.shape[0] * self.shape[1]
        assert n == len (self.operators)
        out = [[ None for i in range(self.shape[1])] for j in range(self.shape[0])]
        s = 0
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                out[i][j] = self.operators[s]
                s +=1
        return out    
    
    def norm(self):
        tmp = 0
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                tmp += self.compMat[i][j].norm()**2
        return np.sqrt(tmp)   

    def direct(self, x, out=None):
        
        out = [None]*self.shape[0]
        for i in range(self.shape[0]):                        
            z1 = ImageData(np.zeros(self.compMat[i][0].range_dim()))
            for j in range(self.shape[1]):
                z1 += self.compMat[i][j].direct(x.get_item(j))
            out[i] = z1    
                                
        return CompositeDataContainer(*out)
                                          
    def adjoint(self, x, out=None):        
        
        out = [None]*self.shape[1]
        for i in range(self.shape[1]):
            z2 = ImageData(np.zeros(self.compMat[0][i].domain_dim()))
            for j in range(self.shape[0]):
                z2 += self.compMat[j][i].adjoint(x.get_item(j))
            out[i] = z2
                                
        return CompositeDataContainer(*out)
            
    def range_dim(self):

        tmp = [ [] for i in range(self.shape[0])]
        for i in range(self.shape[0]):
            tmp[i]=self.compMat[i][0].range_dim()
#            if isinstance(tmp[i],tuple):
#                tmp[i]=[tmp[i]]
        return tmp
    
    def domain_dim(self):

        tmp = [[] for k in range(self.shape[1])]
        for i in range(self.shape[1]):
            tmp[i]=self.compMat[0][i].domain_dim()
        return tmp
    
    def alloc_domain_dim(self):     
        tmp = [[] for k in range(self.shape[1])] 
        for k in range(self.shape[1]):
            tmp[k] = ImageData(np.zeros(self.compMat[0][k].domain_dim()))
        return CompositeDataContainer(*tmp)
        
    
    def alloc_range_dim(self):
        tmp = [ [] for i in range(self.shape[0])]
        for k in range(self.shape[0]):            
            tmp[k] = ImageData(np.zeros(self.compMat[k][0].range_dim()))
        return CompositeDataContainer(*tmp) 
    
        


################################################################################
#
#class test_sym_gradient(Operator):
#    
##    def __init__(self, memopt = False):
#    def __init__(self, geometry, memopt = False):
#        self.memopt = memopt
#        self.geometry = geometry
#        super(sym_gradient, self).__init__()
#        
#    def domain_dim(self):       
#        return self.geometry
##        return ImageData(geometry = self.geometry).shape         
#        
#    def range_dim(self):
#        return (len(self.domain_dim()),) + self.domain_dim()[1:]        
#        
#    def direct(self, x, out=None):
#                
#        grad = np.zeros((len(x.shape),)+x.shape[1:])
#        
#        grad[0] = finite_diff(x.as_array()[0], direction = 1, method = 'back')
#        grad[1] = finite_diff(x.as_array()[1], direction = 0, method = 'back')
#        grad[2] = 0.5 * (finite_diff(x.as_array()[0], direction = 0, method = 'back') + \
#                      finite_diff(x.as_array()[1], direction = 1, method = 'back') )
#                            
#        return type(x)(grad)
#    
#    def adjoint(self, x, out=None):
#        
#        div = np.zeros((len(x.shape[1:]),) + x.shape[1:])
#        
#        div[0] = finite_diff(x.as_array()[0], direction = 1, method = 'for') + \
#                 finite_diff(x.as_array()[2], direction = 0, method = 'for')
#         
#        div[1] = finite_diff(x.as_array()[2], direction = 1, method = 'for') + \
#                 finite_diff(x.as_array()[1], direction = 0, method = 'for')
#                 
#        return type(x)(-div) 
#    
#    def norm(self):
##        return np.sqrt(4*len(self.domainDim()))        
#        #TODO this takes time for big ImageData
#        # for 2D ||grad|| = sqrt(8), 3D ||grad|| = sqrt(12)        
#        x0 = ImageData(np.random.random_sample(self.domainDim()), geometry = self.geometry)
#        self.s1, sall, svec = PowerMethodNonsquare(self, 25, x0)
#        return self.s1


#class sym_gradient(Operator):
#    
##    def __init__(self, memopt = False):
#    def __init__(self, geometry, memopt = False):
#        self.memopt = memopt
#        self.geometry = geometry
#        super(sym_gradient, self).__init__()
#        
#    def domain_dim(self):       
#        return self.geometry
##        return ImageData(geometry = self.geometry).shape         
#        
#    def range_dim(self):
#        return (len(self.domain_dim()),) + self.domain_dim()[1:]        
#        
#    def direct(self, x, out=None):
#                
#        grad = np.zeros((len(x.shape),)+x.shape[1:])
#        
#        grad[0] = finite_diff(x.as_array()[0], direction = 1, method = 'back')
#        grad[1] = finite_diff(x.as_array()[1], direction = 0, method = 'back')
#        grad[2] = 0.5 * (finite_diff(x.as_array()[0], direction = 0, method = 'back') + \
#                      finite_diff(x.as_array()[1], direction = 1, method = 'back') )
#                            
#        return type(x)(grad)
#    
#    def adjoint(self, x, out=None):
#        
#        div = np.zeros((len(x.shape[1:]),) + x.shape[1:])
#        
#        div[0] = finite_diff(x.as_array()[0], direction = 1, method = 'for') + \
#                 finite_diff(x.as_array()[2], direction = 0, method = 'for')
#         
#        div[1] = finite_diff(x.as_array()[2], direction = 1, method = 'for') + \
#                 finite_diff(x.as_array()[1], direction = 0, method = 'for')
#                 
#        return type(x)(-div) 
#    
#    def norm(self):
##        return np.sqrt(4*len(self.domainDim()))        
#        #TODO this takes time for big ImageData
#        # for 2D ||grad|| = sqrt(8), 3D ||grad|| = sqrt(12)        
#        x0 = ImageData(np.random.random_sample(self.domain_dim()), geometry = self.geometry)
#        self.s1, sall, svec = PowerMethodNonsquare(self, 25, x0)
#        return self.s1
    
###########################   ZeroOp    #######################################

#class ZeroOp(Operator):
#    
#    def __init__(self, gm_domain, gm_range):
#        self.gm_domain = gm_domain
#        self.gm_range = gm_range
#        super(ZeroOp, self).__init__()
#        
#    def direct(self,x,out=None):
#        if out is None:
#            return ImageData(np.zeros(self.gm_range))
#        else:
#            return ImageData(np.zeros(self.gm_range))
#    
#    def adjoint(self,x, out=None):
#        if out is None:
#            return ImageData(np.zeros(self.gm_domain))
#        else:
#            return ImageData(np.zeros(self.gm_domain))
#        
#    def norm(self):
#        return 0
#    
#    def domain_dim(self):       
#        return self.gm_domain  
#        
#    def range_dim(self):
#        return self.gm_range
#    
#####################    Identity Operator      ################################    
#            
#class Identity(Operator):
#    
#    def __init__(self, gm_domain, gm_range=None):
#
#        self.gm_domain = gm_domain
#        self.gm_range = gm_range  
#        if self.gm_range is None:
#            self.gm_range = self.gm_domain
#        
#        super(Identity, self).__init__()
#        
#    def direct(self,x,out=None):
#        if out is None:
#            return x.copy()
#        else:
#            out.fill(x)
#    
#    def adjoint(self,x, out=None):
#        if out is None:
#            return x.copy()
#        else:
#            out.fill(x)
#        
#    def norm(self):
#        return 1.0
#        
#    def domain_dim(self):       
#        return self.gm_domain
#        
#    def range_dim(self):
#        return self.gm_range
#    
################################################################################
#    
#
#class AstraProjectorSimple(Operator):
#    
#    """ASTRA projector modified to use DataSet and geometry."""
#    def __init__(self, geomv, geomp, device):
#        super(AstraProjectorSimple, self).__init__()
#        
#        # Store volume and sinogram geometries.
#        self.sinogram_geometry = geomp
#        self.volume_geometry = geomv
#        
#        self.fp = AstraForwardProjector(volume_geometry=geomv,
#                                        sinogram_geometry=geomp,
#                                        proj_id=None,
#                                        device=device)
#        
#        self.bp = AstraBackProjector(volume_geometry=geomv,
#                                        sinogram_geometry=geomp,
#                                        proj_id=None,
#                                        device=device)
#                
#        # Initialise empty for singular value.
#        self.s1 = None
#    
#    def direct(self, IM):
#        self.fp.set_input(IM)
#        out = self.fp.get_output()
#        return out
#    
#    def adjoint(self, DATA):
#        self.bp.set_input(DATA)
#        out = self.bp.get_output()
#        return out
#    
#    #def delete(self):
#    #    astra.data2d.delete(self.proj_id)
#    
#    def domain_dim(self):
#        return (self.volume_geometry.voxel_num_x, \
#                  self.volume_geometry.voxel_num_y)
#        
#    def range_dim(self):  
#        return (self.sinogram_geometry.angles.size, \
#                  self.sinogram_geometry.pixel_num_h)    
#    
#    def norm(self):
#        x0 = ImageData(np.random.random_sample(self.domain_dim()))
#        self.s1, sall, svec = PowerMethodNonsquare(self,10,x0)
#        return self.s1
#    
#    def size(self):
#        # Only implemented for 2D
#        return ( (self.sinogram_geometry.angles.size, \
#                  self.sinogram_geometry.pixel_num_h), \
#                 (self.volume_geometry.voxel_num_x, \
#                  self.volume_geometry.voxel_num_y) )    



        
        
    
    







