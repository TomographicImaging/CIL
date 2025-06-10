#  Copyright 2019 United Kingdom Research and Innovation
#  Copyright 2019 The University of Manchester
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Authors:
# CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt
import functools
import warnings
from numbers import Number

import numpy

from ..utilities.multiprocessing import NUM_THREADS
from .labels import FillType


class BlockGeometry(object):
    @property
    def RANDOM(self):
        warnings.warn("use FillType.RANDOM instead", DeprecationWarning, stacklevel=2)
        return FillType.RANDOM

    @property
    def RANDOM_INT(self):
        warnings.warn("use FillType.RANDOM_INT instead", DeprecationWarning, stacklevel=2)
        return FillType.RANDOM_INT

    @property
    def dtype(self):
        return tuple(i.dtype for i in self.geometries)

    '''Class to hold Geometry as column vector'''
    #__array_priority__ = 1
    def __init__(self, *args, **kwargs):
        ''''''
        self.geometries = args
        self.index = 0
        shape = (len(args),1)
        self.shape = shape

        n_elements = functools.reduce(lambda x,y: x*y, shape, 1)
        if len(args) != n_elements:
            raise ValueError(
                    'Dimension and size do not match: expected {} got {}'
                    .format(n_elements, len(args)))

    def get_item(self, index):
        '''returns the Geometry in the BlockGeometry located at position index'''
        return self.geometries[index]

    def allocate(self, value=0, **kwargs):

        '''Allocates a BlockDataContainer according to geometries contained in the BlockGeometry'''

        symmetry = kwargs.get('symmetry',False)
        containers = [geom.allocate(value, **kwargs) for geom in self.geometries]

        if symmetry == True:

            # for 2x2
            # [ ig11, ig12\
            #   ig21, ig22]

            # Row-wise Order

            if len(containers)==4:
                containers[1]=containers[2]

            # for 3x3
            # [ ig11, ig12, ig13\
            #   ig21, ig22, ig23\
            #   ig31, ig32, ig33]

            elif len(containers)==9:
                containers[1]=containers[3]
                containers[2]=containers[6]
                containers[5]=containers[7]

            # for 4x4
            # [ ig11, ig12, ig13, ig14\
            #   ig21, ig22, ig23, ig24\ c
            #   ig31, ig32, ig33, ig34
            #   ig41, ig42, ig43, ig44]

            elif len(containers) == 16:
                containers[1]=containers[4]
                containers[2]=containers[8]
                containers[3]=containers[12]
                containers[6]=containers[9]
                containers[7]=containers[10]
                containers[11]=containers[15]

        return BlockDataContainer(*containers)

    def __iter__(self):
        '''BlockGeometry is an iterable'''
        return self

    def __next__(self):
        '''BlockGeometry is an iterable'''
        if self.index < len(self.geometries):
            result = self.geometries[self.index]
            self.index += 1
            return result
        else:
            self.index = 0
            raise StopIteration

    def __eq__(self, value: object) -> bool:
        if len(self.geometries) != len(value.geometries):
            return False
        return functools.reduce(lambda x,y: x and y, \
                                [sel == vel for sel,vel in zip(self.geometries, value.geometries)], True)

class BlockDataContainer(object):
    '''Class to hold DataContainers as column vector

    Provides basic algebra between BlockDataContainer's, DataContainer's and
    subclasses and Numbers

    1) algebra between `BlockDataContainer`s will be element-wise, only if
       the shape of the 2 `BlockDataContainer`s is the same, otherwise it
       will fail
    2) algebra between `BlockDataContainer`s and `list` or `numpy array` will
       work as long as the number of `rows` and element of the arrays match,
       independently on the fact that the `BlockDataContainer` could be nested
    3) algebra between `BlockDataContainer` and one `DataContainer` is possible.
       It will require all the `DataContainers` in the block to be
       compatible with the `DataContainer` we want to operate with.
    4) algebra between `BlockDataContainer` and a `Number` is possible and it
       will be done with each element of the `BlockDataContainer` even if nested

    A = [ [B,C] , D]
    A * 3 = [ 3 * [B,C] , 3* D] = [ [ 3*B, 3*C]  , 3*D ]

    '''
    ADD       = 'add'
    SUBTRACT  = 'subtract'
    MULTIPLY  = 'multiply'
    DIVIDE    = 'divide'
    POWER     = 'power'
    SAPYB     = 'sapyb'
    MAXIMUM   = 'maximum'
    MINIMUM   = 'minimum'
    ABS       = 'abs'
    SIGN      = 'sign'
    SQRT      = 'sqrt'
    CONJUGATE = 'conjugate'
    __array_priority__ = 1
    __container_priority__ = 2

    @property
    def dtype(self):
        return tuple(i.dtype for i in self.containers)

    def __init__(self, *args, **kwargs):
        ''''''
        self.containers = args
        self.index = 0
        #if len(set([i.shape for i in self.containers])):
        #    self.geometry = self.containers[0].geometry

        shape = kwargs.get('shape', None)
        if shape is None:
           shape = (len(args),1)
#        shape = (len(args),1)
        self.shape = shape

        n_elements = functools.reduce(lambda x,y: x*y, shape, 1)
        if len(args) != n_elements:
            raise ValueError(
                    'Dimension and size do not match: expected {} got {}'
                    .format(n_elements, len(args)))


    def __iter__(self):
        '''BlockDataContainer is Iterable'''
        self.index=0
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
        elif isinstance(other, (list, tuple, numpy.ndarray)) :
            for ot in other:
                if not isinstance(ot, Number):
                    raise ValueError('List/ numpy array can only contain numbers {}'\
                                     .format(type(ot)))
            return len(self.containers) == len(other)
        elif isinstance(other, BlockDataContainer):
            return len(self.containers) == len(other.containers)
        else:
            # this should work for other as DataContainers and children
            ret = True
            for i, el in enumerate(self.containers):
                if isinstance(el, BlockDataContainer):
                    a = el.is_compatible(other)
                else:
                    a = el.shape == other.shape
                ret = ret and a
            # probably will raise
            return ret


    def get_item(self, row):
        if row > self.shape[0]:
            raise ValueError('Requested row {} > max {}'.format(row, self.shape[0]))
        return self.containers[row]

    def __getitem__(self, row):
        return self.get_item(row)

    def add(self, other, *args, **kwargs):
        '''Algebra: add method of BlockDataContainer with number/DataContainer or BlockDataContainer

        Parameters
        ----------
        other : number, DataContainer or subclasses or BlockDataContainer
        out : BlockDataContainer, optional
            Provides a placeholder for the result
        '''
        return self.binary_operations(BlockDataContainer.ADD, other, *args, **kwargs)
    def subtract(self, other, *args, **kwargs):
        '''Algebra: subtract method of BlockDataContainer with number/DataContainer or BlockDataContainer

        Parameters
        ----------
        other : number, DataContainer or subclasses or BlockDataContainer
        out : BlockDataContainer, optional
            Provides a placeholder for the result
        '''
        return self.binary_operations(BlockDataContainer.SUBTRACT, other, *args, **kwargs)
    def multiply(self, other, *args, **kwargs):
        '''Algebra: multiply method of BlockDataContainer with number/DataContainer or BlockDataContainer

        Parameters
        ----------
        other : number, DataContainer or subclasses or BlockDataContainer
        out : BlockDataContainer, optional
            Provides a placeholder for the result
        '''
        return self.binary_operations(BlockDataContainer.MULTIPLY, other, *args, **kwargs)
    def divide(self, other, *args, **kwargs):
        '''Algebra: divide method of BlockDataContainer with number/DataContainer or BlockDataContainer

        Parameters
        ----------
        other : number, DataContainer or subclasses or BlockDataContainer
        out : BlockDataContainer, optional
            Provides a placeholder for the result

        '''
        return self.binary_operations(BlockDataContainer.DIVIDE, other, *args, **kwargs)
    def power(self, other, *args, **kwargs):
        '''Algebra: power method of BlockDataContainer with number/DataContainer or BlockDataContainer

        Parameters
        ----------
        other : number, DataContainer or subclasses or BlockDataContainer
        out : BlockDataContainer, optional
            Provides a placeholder for the result
        '''
        return self.binary_operations(BlockDataContainer.POWER, other, *args, **kwargs)
    def maximum(self, other, *args, **kwargs):
        '''Algebra: maximum method of BlockDataContainer with number/DataContainer or BlockDataContainer

        Parameters
        ----------
        other : number, DataContainer or subclasses or BlockDataContainer
        out : BlockDataContainer, optional
            Provides a placeholder for the result
        '''
        return self.binary_operations(BlockDataContainer.MAXIMUM, other, *args, **kwargs)
    def minimum(self, other, *args, **kwargs):
        '''Algebra: minimum method of BlockDataContainer with number/DataContainer or BlockDataContainer

        Parameters
        ----------
        other : number, DataContainer or subclasses or BlockDataContainer
        out : BlockDataContainer, optional
            Provides a placeholder for the result
            
        '''
        return self.binary_operations(BlockDataContainer.MINIMUM, other, *args, **kwargs)

    def sapyb(self, a, y, b, out=None, num_threads = NUM_THREADS):
        r'''performs axpby element-wise on the BlockDataContainer containers

        Does the operation .. math:: a*x+b*y and stores the result in out, where x is self

        Parameters
        ----------
        a : scalar or BlockDataContainer
        b : scalar or BlockDataContainer
        y : compatible (Block)DataContainer
        out : BlockDataContainer, optional
            Provides a placeholder for the result

        Example
        -------
        >>> a = 2
        >>> b = 3
        >>> ig = ImageGeometry(10,11)
        >>> x = ig.allocate(1)
        >>> y = ig.allocate(2)
        >>> bdc1 = BlockDataContainer(2*x, y)
        >>> bdc2 = BlockDataContainer(x, 2*y)
        >>> out = bdc1.sapyb(a,bdc2,b)
        '''
        if out is None:
            out = self * 0
        kwargs = {'a':a, 'b':b, 'out':out, 'num_threads': NUM_THREADS}
        return self.binary_operations(BlockDataContainer.SAPYB, y, **kwargs)


    def binary_operations(self, operation, other, *args, **kwargs):
        '''Algebra: generic method of algebric operation with BlockDataContainer with number/DataContainer or BlockDataContainer

        Provides commutativity with DataContainer and subclasses, i.e. this
        class's reverse algebraic methods take precedence w.r.t. direct algebraic
        methods of DataContainer and subclasses.

        This method is not to be used directly
        '''
        if not self.is_compatible(other):
            raise ValueError('Incompatible for operation {}'.format(operation))
        out = kwargs.get('out', None)
        if isinstance(other, Number):
            # try to do algebra with one DataContainer. Will raise error if not compatible
            kw = kwargs.copy()
            res = []
            for i,el in enumerate(self.containers):
                if operation == BlockDataContainer.ADD:
                    op = el.add
                elif operation == BlockDataContainer.SUBTRACT:
                    op = el.subtract
                elif operation == BlockDataContainer.MULTIPLY:
                    op = el.multiply
                elif operation == BlockDataContainer.DIVIDE:
                    op = el.divide
                elif operation == BlockDataContainer.POWER:
                    op = el.power
                elif operation == BlockDataContainer.MAXIMUM:
                    op = el.maximum
                elif operation == BlockDataContainer.MINIMUM:
                    op = el.minimum
                else:
                    raise ValueError('Unsupported operation', operation)
                if out is not None:
                    kw['out'] = out.get_item(i)
                    op(other, *args, **kw)
                else:
                    res.append(op(other, *args, **kw))
            if out is not None:
                return out
            else:
                return type(self)(*res, shape=self.shape)
        elif isinstance(other, (list, tuple, numpy.ndarray, BlockDataContainer)):
            kw = kwargs.copy()
            res = []
            if isinstance(other, BlockDataContainer):
                the_other = other.containers
            else:
                the_other = other

            for i,zel in enumerate(zip ( self.containers, the_other) ):
                el = zel[0]
                ot = zel[1]
                if operation == BlockDataContainer.ADD:
                    op = el.add
                elif operation == BlockDataContainer.SUBTRACT:
                    op = el.subtract
                elif operation == BlockDataContainer.MULTIPLY:
                    op = el.multiply
                elif operation == BlockDataContainer.DIVIDE:
                    op = el.divide
                elif operation == BlockDataContainer.POWER:
                    op = el.power
                elif operation == BlockDataContainer.MAXIMUM:
                    op = el.maximum
                elif operation == BlockDataContainer.MINIMUM:
                    op = el.minimum
                elif operation == BlockDataContainer.SAPYB:
                    if not isinstance(other, BlockDataContainer):
                        raise ValueError("{} cannot handle {}".format(operation, type(other)))
                    op = el.sapyb
                else:
                    raise ValueError('Unsupported operation', operation)

                if out is not None:
                    if operation == BlockDataContainer.SAPYB:
                        if isinstance(kw['a'], BlockDataContainer):
                            a = kw['a'].get_item(i)
                        else:
                            a = kw['a']

                        if isinstance(kw['b'], BlockDataContainer):
                            b = kw['b'].get_item(i)
                        else:
                            b = kw['b']

                        el.sapyb(a, ot, b, out.get_item(i), num_threads=kw['num_threads'])
                    else:
                        kw['out'] = out.get_item(i)
                        op(ot, *args, **kw)
                else:
                    res.append(op(ot, *args, **kw))
            if out is not None:
                return out
            else:
                return type(self)(*res, shape=self.shape)
        else:
            # try to do algebra with one DataContainer. Will raise error if not compatible
            kw = kwargs.copy()
            if operation != BlockDataContainer.SAPYB:
                # remove keyworded argument related to SAPYB
                for k in ['a','b','y', 'num_threads', 'dtype']:
                    if k in kw.keys():
                        kw.pop(k)

            res = []
            for i,el in enumerate(self.containers):
                if operation == BlockDataContainer.ADD:
                    op = el.add
                elif operation == BlockDataContainer.SUBTRACT:
                    op = el.subtract
                elif operation == BlockDataContainer.MULTIPLY:
                    op = el.multiply
                elif operation == BlockDataContainer.DIVIDE:
                    op = el.divide
                elif operation == BlockDataContainer.POWER:
                    op = el.power
                elif operation == BlockDataContainer.MAXIMUM:
                    op = el.maximum
                elif operation == BlockDataContainer.MINIMUM:
                    op = el.minimum
                elif operation == BlockDataContainer.SAPYB:

                    if isinstance(kw['a'], BlockDataContainer):
                        a = kw['a'].get_item(i)
                    else:
                        a = kw['a']

                    if isinstance(kw['b'], BlockDataContainer):
                        b = kw['b'].get_item(i)
                    else:
                        b = kw['b']

                    el.sapyb(a, other, b, out.get_item(i), kw['num_threads'])

                    # As axpyb cannot return anything we `continue` to skip the rest of the code block
                    continue

                else:
                    raise ValueError('Unsupported operation', operation)
                if out is not None:
                    kw['out'] = out.get_item(i)
                    op(other, *args, **kw)
                else:
                    res.append(op(other, *args, **kw))

            if out is not None:
                return out
            else:
                return type(self)(*res, shape=self.shape)

    ## unary operations

    def unary_operations(self, operation, *args, **kwargs ):
        '''Unary operation on BlockDataContainer:

        generic method of unary operation with BlockDataContainer: abs, sign, sqrt and conjugate

        This method is not to be used directly
        '''
        out = kwargs.get('out', None)
        kw = kwargs.copy()
        if out is None:
            res = []
            for el in self.containers:
                if operation == BlockDataContainer.ABS:
                    op = el.abs
                elif operation == BlockDataContainer.SIGN:
                    op = el.sign
                elif operation == BlockDataContainer.SQRT:
                    op = el.sqrt
                elif operation == BlockDataContainer.CONJUGATE:
                    op = el.conjugate
                res.append(op(*args, **kw))
            return BlockDataContainer(*res)
        else:
            kw.pop('out')
            for el,elout in zip(self.containers, out.containers):
                if operation == BlockDataContainer.ABS:
                    op = el.abs
                elif operation == BlockDataContainer.SIGN:
                    op = el.sign
                elif operation == BlockDataContainer.SQRT:
                    op = el.sqrt
                elif operation == BlockDataContainer.CONJUGATE:
                    op = el.conjugate
                kw['out'] = elout
                op(*args, **kw)

    def abs(self, *args, **kwargs):
        return self.unary_operations(BlockDataContainer.ABS, *args, **kwargs)
    def sign(self, *args, **kwargs):
        return self.unary_operations(BlockDataContainer.SIGN, *args, **kwargs)
    def sqrt(self, *args, **kwargs):
        return self.unary_operations(BlockDataContainer.SQRT, *args, **kwargs)
    def conjugate(self, *args, **kwargs):
        return self.unary_operations(BlockDataContainer.CONJUGATE, *args, **kwargs)
    # def abs(self, *args,  **kwargs):
    #     return type(self)(*[ el.abs(*args, **kwargs) for el in self.containers], shape=self.shape)
    # def sign(self, *args,  **kwargs):
    #     return type(self)(*[ el.sign(*args, **kwargs) for el in self.containers], shape=self.shape)
    # def sqrt(self, *args,  **kwargs):
    #     return type(self)(*[ el.sqrt(*args, **kwargs) for el in self.containers], shape=self.shape)
    # def conjugate(self, out=None):
    #     return type(self)(*[el.conjugate() for el in self.containers], shape=self.shape)

    ## reductions

    def sum(self, *args, **kwargs):
        return numpy.sum([ el.sum(*args, **kwargs) for el in self.containers])

    def squared_norm(self):
        y = numpy.asarray([el.squared_norm() for el in self.containers])
        return y.sum()


    def norm(self):
        return numpy.sqrt(self.squared_norm())

    def pnorm(self, p=2):
        # See https://github.com/TomographicImaging/CIL/issues/1525#issuecomment-1757413803
        if not functools.reduce(lambda x,y: x and y, [el.shape == self.containers[0].shape for el in self.containers], True):
            raise ValueError('pnorm: Incompatible shapes - each container in the BlockDataContainer must have the same shape in order to calculate the pnorm')
        if p==1:
            return sum(self.abs())
        elif p==2:
            tmp = functools.reduce(lambda a,b: a + b.conjugate()*b, self.containers, self.get_item(0) * 0 ).sqrt()
            return tmp
        else:
            return ValueError('Not implemented')

    def copy(self):
        '''alias of clone'''
        return self.clone()
    def clone(self):
        return type(self)(*[el.copy() for el in self.containers], shape=self.shape)
    def fill(self, other):
        if isinstance (other, BlockDataContainer):
            if not self.is_compatible(other):
                raise ValueError('Incompatible containers')
            for el,ot in zip(self.containers, other.containers):
                el.fill(ot)
        else:
            return ValueError('Cannot fill with object provided {}'.format(type(other)))

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
            if not self.is_compatible(other):
                raise ValueError('Incompatible for __iadd__')
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
            if not self.is_compatible(other):
                raise ValueError('Incompatible for __isub__')
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
            if not self.is_compatible(other):
                raise ValueError('Incompatible for __imul__')
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
            if not self.is_compatible(other):
                raise ValueError('Incompatible for __idiv__')
            for el,ot in zip(self.containers, other):
                el /= ot
        return self
    # __rdiv__
    def __itruediv__(self, other):
        '''Inline truedivision'''
        return self.__idiv__(other)

    def __neg__(self):
        """ Return - self """
        return -1 * self

    def dot(self, other):
#
        tmp = [ self.containers[i].dot(other.containers[i]) for i in range(self.shape[0])]
        return sum(tmp)

    def __len__(self):

        return self.shape[0]

    @property
    def geometry(self):
        try:
            return BlockGeometry(*[el.geometry.copy() for el in self.containers])
        except AttributeError:
            return None
