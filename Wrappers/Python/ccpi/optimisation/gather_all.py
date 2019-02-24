#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 21:02:07 2019

@author: evangelos
"""

from ccpi.framework import ImageData, DataContainer
import numpy
import functools
from GradientOperator import FiniteDiff
from numbers import Number
from operators import MyTomoIdentity, finite_diff
from ccpi.optimisation.ops import PowerMethodNonsquare

from skimage.util import random_noise
from functions import Function
from my_changes import Norm2sq_new, ZeroFun


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
        assert isinstance(other, Number)
        self.scalar = other
        return self  
    
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
    '''Class to hold a composite operator'''
    def __init__(self, *args, shape=None):
        self.operators = args
        if shape is None:
            shape = (len(args),1)
        self.shape = shape
        n_elements = functools.reduce(lambda x,y: x*y, shape, 1)
        if len(args) != n_elements:
            raise ValueError(
                    'Dimension and size do not match: expected {} got {}'
                    .format(n_elements,len(args)))
    def get_item(self, row, col):
        if row > self.shape[0]:
            raise ValueError('Requested row {} > max {}'.format(row, self.shape[0]))
        if col > self.shape[1]:
            raise ValueError('Requested col {} > max {}'.format(col, self.shape[1]))
        
        index = row*self.shape[1]+col
        return self.operators[index]
    
    
    def direct(self, x, out=None):
        shape = self.get_output_shape(x.shape)
        res = []
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                if col == 0:
                    prod = self.get_item(row,col).direct(x.get_item(col))
                else:
                    prod += self.get_item(row,col).direct(x.get_item(col))
            res.append(prod)
        return CompositeDataContainer(*res, shape=shape)
    
    def adjoint(self, x, out=None):
        shape = self.get_output_shape(x.shape, adjoint=True)
        res = []
        for row in range(self.shape[1]):
            for col in range(self.shape[0]):
                if col == 0:
                    prod = self.get_item(row,col).adjoint(x.get_item(col))
                else:
                    prod += self.get_item(row,col).adjoint(x.get_item(col))
            res.append(prod)
        return CompositeDataContainer(*res, shape=shape)
    
    def get_output_shape(self, xshape, adjoint=False):
        sshape = self.shape[1]
        oshape = self.shape[0]
        if adjoint:
            sshape = self.shape[0]
            oshape = self.shape[1]
        if sshape != xshape[0]:
            raise ValueError('Incompatible shapes {} {}'.format(self.shape, xshape))
        return (oshape, xshape[-1])
    
    def range_dim(self):
        
        tmp = []    
        tmp1 = []
        for i in range(self.shape[0]): 
            tmp.append(self.get_item(i,0).range_dim())                                                       
        return tmp
                
    def domain_dim(self):
        tmp = []
        for i in range(self.shape[1]):
            tmp.append(self.get_item(0,i).domain_dim())                                
        return tmp
    
    def init_primal_var(self):
        tmp = []
        for i in range(self.shape[1]):
            tmp.append(ImageData(numpy.zeros(self.get_item(0,i).domain_dim()[i])))
        return CompositeDataContainer(*tmp)    
            
    def init_dual_var(self):
        tmp = []
        for i in range(self.shape[0]):
            tmp.append(ImageData(numpy.zeros(self.get_item(i,0).range_dim()[i]))) 
        return CompositeDataContainer(*tmp) 
    
    def norm(self):
        tmp = 0
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                tmp += self.get_item(i,j).norm()**2
        return numpy.sqrt(tmp)     
    
class MyTomoIdentity(Operator):
    
    def __init__(self, gm_domain, gm_range):

        self.gm_domain = gm_domain
        self.gm_range = gm_range    
        super(MyTomoIdentity, self).__init__()
        
    def direct(self,x,out=None):
        if out is None:
            return CompositeDataContainer(x.copy())
        else:
            out.fill(CompositeDataContainer(x))
    
    def adjoint(self,x, out=None):
        if out is None:
            return CompositeDataContainer(x.copy())
        else:
            out.fill(CompositeDataContainer(x))
        
    def norm(self):
        return 1.0
             
    def domain_dim(self):       
        return self.gm_domain
        
    def range_dim(self):
        return self.gm_range    
    
    
class Gradient(Operator):
    
    def __init__(self, gm_domain, gm_range=None, bnd_cond = 'Neumann', **kwargs):
        
        self.gm_domain = gm_domain
        self.gm_range = gm_range
        self.bnd_cond = bnd_cond
        
        if self.gm_range is None:
             self.gm_range = (self.gm_domain,) * (len(list(self.gm_domain)))
            
        self.memopt = kwargs.get('memopt',False)
        self.correlation = kwargs.get('correlation','Space') 
                            
        super(Gradient, self).__init__()  
            
    def direct(self, x, out=None):
         
        if isinstance(x, ImageData):
            x = CompositeDataContainer(x)
                          
        res = []
        for i in range(len(self.gm_range)):
            tmp = FiniteDiff(self.gm_domain, direction = i, bnd_cond = self.bnd_cond).direct(x.get_item(0))
            res.append(tmp)            
        return CompositeDataContainer(*res)
    
    def adjoint(self, x, out=None):
        
        if isinstance(x, list):
            x = CompositeDataContainer(*x)
            
        if isinstance(x, ImageData):
            x = CompositeDataContainer(x)            
                            
        res = []
        for i in range(len(self.gm_domain)):
            tmp = FiniteDiff(self.gm_domain, direction = i, bnd_cond = self.bnd_cond).adjoint(x.get_item(i,0))
            res.append(tmp)  
        return CompositeDataContainer(sum(res))

    def norm(self):
#        return np.sqrt(4*len(self.domainDim()))        
        #TODO this takes time for big ImageData
        # for 2D ||grad|| = sqrt(8), 3D ||grad|| = sqrt(12)        
        x0 = CompositeDataContainer(ImageData(numpy.random.random_sample(self.gm_domain)))
        self.s1, sall, svec = PowerMethodNonsquare(self, 25, x0)
        return self.s1   
    
    def domain_dim(self):
        return self.gm_domain
    
    def range_dim(self):
        return self.gm_range
    
    
class CompositeFunction(Function):
    
    def __init__(self, *args):
        
        self.functions = args
        self.length = len(self.functions)
        
    def get_item(self, ind):        
        return self.functions[ind]        
            
    def proximal_conj(self, x, tau):
        
        tmp = []
        for i in range(self.length):
            tmp.append(self.functions[i].proximal_conj(x, tau))            
        return tmp 
    
class L1Norm(Function):
#
    def __init__(self,A,b=0,alpha=1.0,memopt = False):
        self.A = A
        self.b = b
        self.alpha = alpha
        self.memopt = memopt
        super(L1Norm, self).__init__() 
                       
    def proximal_conj(self, x, tau, out = None):  
#                
        res = x.divide((ImageData(x.power(2).sum(axis=0)).sqrt()/self.alpha).maximum(1.0))
        return res
#        
#        if self.memopt:    
#            out.fill(type(x)(res, geometry = x.geometry))  
#        else:
#            return type(x)(res, geometry = x.geometry)     
            
            
            
#        for i in self.functions:
#            print(i)
        
    
                  
###############################################################################  
#%%
#N, M = (2,3)
#ig = (N,M)
#ag = ig
#
#Grad2D = Gradient(ig)    
#
#u = ImageData(numpy.random.randint(10, size=(N,M)))
#w = [ImageData(numpy.random.randint(10, size=(N,M))),ImageData(numpy.random.randint(10, size=(N,M)))] 
#
#u_DC = CompositeDataContainer(u)
#w_DC = CompositeDataContainer(*w)
#
#Grad2D = Gradient((N,M))
#
#a1 = Grad2D.direct(u)
#a2 = Grad2D.adjoint(w)
#
#b1 = Grad2D.direct(u_DC)
#b2 = Grad2D.adjoint(w_DC)
#
#K = CompositeOperator(Grad2D, MyTomoIdentity(ig, ag), shape=(2,1))
#
#c1 = K.direct(u_DC)
#c2 = K.adjoint(CompositeDataContainer(w_DC, u_DC))
#
## test arrays same
#numpy.testing.assert_array_equal(a1.get_item(0,0).as_array(), \
#                                 c1.get_item(0,0).get_item(0,0).as_array())

#%%
############################# Start PDHG    ###################################

N = 100
ig = (N,N)
ag = ig
Grad2D = Gradient((N,N))
K = CompositeOperator(Grad2D, MyTomoIdentity(ig, ag), shape=(2,1))
    
x_old = CompositeDataContainer(ImageData(numpy.zeros(K.domain_dim()[0])))
y_tmp = [ImageData(numpy.zeros(K.range_dim()[0][0])), 
         ImageData(numpy.zeros(K.range_dim()[0][1]))]
y_tmp1 = ImageData(numpy.zeros(K.range_dim()[1]))

y_old = CompositeDataContainer(CompositeDataContainer(*y_tmp), y_tmp1 )

n_it = 10

xbar = x_old
x_tmp = x_old
x = x_old
y_tmp = y_old
y = y_tmp

theta = 1
alpha = 1

ig = (N,N)
ag = ig

#
phantom = numpy.zeros((N,N))
#
x1 = numpy.linspace(0, int(N/2), N)
x2 = numpy.linspace(int(N/2), 0., N)
xv, yv = numpy.meshgrid(x1, x2)
#
xv[int(N/4):int(3*N/4)-1, int(N/4):int(3*N/4)-1] = yv[int(N/4):int(3*N/4)-1, int(N/4):int(3*N/4)-1].T
#
phantom = xv
## Normalize to [0 1]. Better use of regularizing parameters and MOSEK solver
phantom = phantom/phantom.max()
noisy_data = ImageData(random_noise(phantom,'gaussian', mean = 0, var = 0.05))

f = [ L1Norm(Grad2D, alpha ), \
      Norm2sq_new(MyTomoIdentity(ig,ag), noisy_data, c = 0.5, memopt = False) ]
f_CP = CompositeFunction( L1Norm(Grad2D, alpha ), \
                         Norm2sq_new(MyTomoIdentity(ig,ag), noisy_data, c = 0.5, memopt = False))
g = ZeroFun()

sigma = 1/K.norm()
tau = 1/K.norm()

#for i in range(n_it):
#    
#    opDirect = K.direct(xbar)
#    for i in range(K.shape[0]):
#            y_tmp = y_old + sigma *  opDirect
            
#    z = f[i].proximal_conj(y_tmp.get_item(0,0).get_item(0,0), sigma)
#            y[i] = f[i].proximal_conj(y_tmp[i], sigma)
    


###############################################################################


    
    
    
    