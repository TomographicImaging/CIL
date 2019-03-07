#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import numpy
from ccpi.framework import DataContainer, ImageData, ImageGeometry, AcquisitionData
#from ccpi.astra.processors import AstraForwardProjector, AstraBackProjector
#from ccpi.optimisation.ops import PowerMethodNonsquare
#from Operators.operators import Operator
from numbers import Number
import functools


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

