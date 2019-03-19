# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 16:04:45 2019

@author: ofn77899
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy
from numbers import Number
import functools
#from ccpi.framework import AcquisitionData, ImageData
#from ccpi.optimisation.operators import Operator, LinearOperator
 
class BlockDataContainer(object):
    '''Class to hold DataContainers as column vector'''
    __array_priority__ = 1
    def __init__(self, *args, **kwargs):
        ''''''
        self.containers = args
        self.index = 0
        #shape = kwargs.get('shape', None)
        #if shape is None:
        #   shape = (len(args),1)
        shape = (len(args),1)
        self.shape = shape
        #print (self.shape)
        n_elements = functools.reduce(lambda x,y: x*y, shape, 1)
        if len(args) != n_elements:
            raise ValueError(
                    'Dimension and size do not match: expected {} got {}'
                    .format(n_elements, len(args)))

        
    def __iter__(self):
        '''BlockDataContainer is Iterable'''
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

    def get_item(self, row):
        if row > self.shape[0]:
            raise ValueError('Requested row {} > max {}'.format(row, self.shape[0]))
        return self.containers[row]

    def __getitem__(self, row):
        return self.get_item(row)
                
    def add(self, other, *args, **kwargs):
        assert self.is_compatible(other)
        out = kwargs.get('out', None)
        if isinstance(other, Number):
            return type(self)(*[ el.add(other, out, *args, **kwargs) for el in self.containers], shape=self.shape)
        elif isinstance(other, list) or isinstance(other, numpy.ndarray):
            return type(self)(*[ el.add(ot, out, *args, **kwargs) for el,ot in zip(self.containers,other)], shape=self.shape)
        return type(self)(
            *[ el.add(ot, out, *args, **kwargs) for el,ot in zip(self.containers,other.containers)],
            shape=self.shape)
        
    def subtract(self, other, *args, **kwargs):
        assert self.is_compatible(other)
        out = kwargs.get('out', None)
        if isinstance(other, Number):
            return type(self)(*[ el.subtract(other, out, *args, **kwargs) for el in self.containers], shape=self.shape)
        elif isinstance(other, list) or isinstance(other, numpy.ndarray):
            return type(self)(*[ el.subtract(ot, out, *args, **kwargs) for el,ot in zip(self.containers,other)], shape=self.shape)
        return type(self)(*[ el.subtract(ot, out, *args, **kwargs) for el,ot in zip(self.containers,other.containers)],
                          shape=self.shape)

    def multiply(self, other, *args, **kwargs):
        self.is_compatible(other)
        out = kwargs.get('out', None)
        if isinstance(other, Number):
            return type(self)(*[ el.multiply(other, out, *args, **kwargs) for el in self.containers], shape=self.shape)
        elif isinstance(other, list):
            return type(self)(*[ el.multiply(ot, out, *args, **kwargs) for el,ot in zip(self.containers,other)], shape=self.shape)
        elif isinstance(other, numpy.ndarray):           
            return type(self)(*[ el.multiply(ot, out, *args, **kwargs) for el,ot in zip(self.containers,other)], shape=self.shape)
        return type(self)(*[ el.multiply(ot, out, *args, **kwargs) for el,ot in zip(self.containers,other.containers)],
                          shape=self.shape)
    
    def divide(self, other, *args, **kwargs):
        self.is_compatible(other)
        out = kwargs.get('out', None)
        if isinstance(other, Number):
            return type(self)(*[ el.divide(other, out, *args, **kwargs) for el in self.containers], shape=self.shape)
        elif isinstance(other, list) or isinstance(other, numpy.ndarray):
            return type(self)(*[ el.divide(ot, out, *args, **kwargs) for el,ot in zip(self.containers,other)], shape=self.shape)
        return type(self)(*[ el.divide(ot, out, *args, **kwargs) for el,ot in zip(self.containers,other.containers)],
                          shape=self.shape)
    
    def power(self, other, *args, **kwargs):
        assert self.is_compatible(other)
        out = kwargs.get('out', None)
        if isinstance(other, Number):
            return type(self)(*[ el.power(other, out, *args, **kwargs) for el in self.containers], shape=self.shape)
        elif isinstance(other, list) or isinstance(other, numpy.ndarray):
            return type(self)(*[ el.power(ot, out, *args, **kwargs) for el,ot in zip(self.containers,other)], shape=self.shape)
        return type(self)(*[ el.power(ot, out, *args, **kwargs) for el,ot in zip(self.containers,other.containers)], shape=self.shape)
    
    def maximum(self,other, *args, **kwargs):
        assert self.is_compatible(other)
        out = kwargs.get('out', None)
        if isinstance(other, Number):
            return type(self)(*[ el.maximum(other, out, *args, **kwargs) for el in self.containers], shape=self.shape)
        elif isinstance(other, list) or isinstance(other, numpy.ndarray):
            return type(self)(*[ el.maximum(ot, out, *args, **kwargs) for el,ot in zip(self.containers,other)], shape=self.shape)
        return type(self)(*[ el.maximum(ot, out, *args, **kwargs) for el,ot in zip(self.containers,other.containers)], shape=self.shape)
    
    ## unary operations    
    def abs(self, *args,  **kwargs):
        out = kwargs.get('out', None)
        return type(self)(*[ el.abs(out, *args, **kwargs) for el in self.containers], shape=self.shape)
    def sign(self, *args,  **kwargs):
        out = kwargs.get('out', None)
        return type(self)(*[ el.sign(out, *args, **kwargs) for el in self.containers], shape=self.shape)
    def sqrt(self, *args,  **kwargs):
        out = kwargs.get('out', None)
        return type(self)(*[ el.sqrt(out, *args, **kwargs) for el in self.containers], shape=self.shape)
    def conjugate(self, out=None):
        return type(self)(*[el.conjugate() for el in self.containers], shape=self.shape)
    
    ## reductions
    def sum(self, *args, **kwargs):
        return numpy.sum([ el.sum(*args, **kwargs) for el in self.containers])
    def squared_norm(self):
        y = numpy.asarray([el.squared_norm() for el in self.containers])
        return y.sum() 
    def norm(self):
        return numpy.sqrt(self.squared_norm())    
    def copy(self):
        '''alias of clone'''    
        return self.clone()
    def clone(self):
        return type(self)(*[el.copy() for el in self.containers], shape=self.shape)
    
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
        '''Reverse addition
        
        to make sure that this method is called rather than the __mul__ of a numpy array
        the class constant __array_priority__ must be set > 0
        https://docs.scipy.org/doc/numpy-1.15.1/reference/arrays.classes.html#numpy.class.__array_priority__
        '''
        return self + other
    # __radd__
    
    def __rsub__(self, other):
        '''Reverse subtraction
        
        to make sure that this method is called rather than the __mul__ of a numpy array
        the class constant __array_priority__ must be set > 0
        https://docs.scipy.org/doc/numpy-1.15.1/reference/arrays.classes.html#numpy.class.__array_priority__
        '''
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
        '''Reverse division
        
        to make sure that this method is called rather than the __mul__ of a numpy array
        the class constant __array_priority__ must be set > 0
        https://docs.scipy.org/doc/numpy-1.15.1/reference/arrays.classes.html#numpy.class.__array_priority__
        '''
        return pow(self / other, -1)
    # __rdiv__
    def __rtruediv__(self, other):
        '''Reverse truedivision
        
        to make sure that this method is called rather than the __mul__ of a numpy array
        the class constant __array_priority__ must be set > 0
        https://docs.scipy.org/doc/numpy-1.15.1/reference/arrays.classes.html#numpy.class.__array_priority__
        '''
        return self.__rdiv__(other)
    
    def __rpow__(self, other):
        '''Reverse power
        
        to make sure that this method is called rather than the __mul__ of a numpy array
        the class constant __array_priority__ must be set > 0
        https://docs.scipy.org/doc/numpy-1.15.1/reference/arrays.classes.html#numpy.class.__array_priority__
        '''
        return other.power(self)
    
    def __iadd__(self, other):
        '''Inline addition'''
        if isinstance (other, BlockDataContainer):
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
    # __iadd__
    
    def __isub__(self, other):
        '''Inline subtraction'''
        if isinstance (other, BlockDataContainer):
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
    # __isub__
    
    def __imul__(self, other):
        '''Inline multiplication'''
        if isinstance (other, BlockDataContainer):
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
        '''Inline division'''
        if isinstance (other, BlockDataContainer):
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
        '''Inline truedivision'''
        return self.__idiv__(other)

