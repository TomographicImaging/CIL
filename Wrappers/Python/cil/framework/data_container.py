#  Copyright 2018 United Kingdom Research and Innovation
#  Copyright 2018 The University of Manchester
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
import copy
import ctypes
import warnings
from functools import reduce
from numbers import Number

import numpy

from .cilacc import cilacc
from cil.utilities.multiprocessing import NUM_THREADS


class DataContainer(object):
    '''Generic class to hold data

    Data is currently held in a numpy arrays'''

    @property
    def geometry(self):
        return None

    @geometry.setter
    def geometry(self, val):
        if val is not None:
            raise TypeError("DataContainers cannot hold a geometry, use ImageData or AcquisitionData instead")

    @property
    def dimension_labels(self):

        if self._dimension_labels is None:
            default_labels = [0]*self.number_of_dimensions
            for i in range(self.number_of_dimensions):
                default_labels[i] = 'dimension_{0:02}'.format(i)
            return tuple(default_labels)
        else:
            return self._dimension_labels

    @dimension_labels.setter
    def dimension_labels(self, val):
        if val is None:
            self._dimension_labels = None
        elif len(val_tuple := tuple(val)) == self.number_of_dimensions:
            self._dimension_labels = val_tuple
        else:
            raise ValueError("dimension_labels expected a list containing {0} strings got {1}".format(self.number_of_dimensions, val))

    @property
    def shape(self):
        '''Returns the shape of the DataContainer'''
        return self.array.shape

    @property
    def ndim(self):
        '''Returns the ndim of the DataContainer'''
        return self.array.ndim

    @property
    def number_of_dimensions(self):
        '''Returns the shape of the  of the DataContainer'''
        return len(self.array.shape)

    @property
    def dtype(self):
        '''Returns the dtype of the data array.'''
        return self.array.dtype

    @property
    def size(self):
        '''Returns the number of elements of the DataContainer'''
        return self.array.size

    __container_priority__ = 1
    def __init__ (self, array, deep_copy=True, dimension_labels=None,
                  **kwargs):
        if type(array) == numpy.ndarray:
            if deep_copy:
                self.array = array.copy()
            else:
                self.array = array
        else:
            raise TypeError('Array must be NumpyArray, passed {0}'\
                            .format(type(array)))

        #Don't set for derived classes
        if type(self) is DataContainer:
            self.dimension_labels = dimension_labels

        # finally copy the geometry, and force dtype of the geometry of the data = the dype of the data
        if 'geometry' in kwargs.keys():
            try:
                self.geometry = kwargs['geometry'].copy()
                if self.geometry.dtype != self.dtype:
                    warnings.warn("Over-riding geometry.dtype with data.dtype", UserWarning)
                    self.geometry.dtype = self.dtype
            except:
                pass

    def get_dimension_size(self, dimension_label):

        if dimension_label in self.dimension_labels:
            i = self.dimension_labels.index(dimension_label)
            return self.shape[i]
        else:
            raise ValueError('Unknown dimension {0}. Should be one of {1}'.format(dimension_label,
                             self.dimension_labels))

    def get_dimension_axis(self, dimension_label):
        """
        Returns the axis index of the DataContainer array if the specified dimension_label(s) match
        any dimension_labels of the DataContainer or their indices

        Parameters
        ----------
        dimension_label: string or int or tuple of strings or ints
            Specify dimension_label(s) or index of the DataContainer from which to check and return the axis index

        Returns
        -------
        int or tuple of ints
            The axis index of the DataContainer matching the specified dimension_label
        """
        if isinstance(dimension_label,(tuple,list)):
            return tuple(self.get_dimension_axis(x) for x in dimension_label)

        if dimension_label in self.dimension_labels:
            return self.dimension_labels.index(dimension_label)
        elif isinstance(dimension_label, int) and dimension_label >= 0 and dimension_label < self.ndim:
            return dimension_label
        else:
            raise ValueError('Unknown dimension {0}. Should be one of {1}, or an integer in range {2} - {3}'.format(dimension_label,
                            self.dimension_labels, 0, self.ndim))


    def as_array(self):
        '''Returns the pointer to the array.
        '''
        return self.array


    def get_slice(self, **kw):
        '''
        Returns a new DataContainer containing a single slice in the requested direction. \
        Pass keyword arguments <dimension label>=index
        '''
        # Force is not relevant for a DataContainer:
        kw.pop('force', None)

        new_array = None

        #get ordered list of current dimensions
        dimension_labels_list = list(self.dimension_labels)

        #remove axes from array and labels
        for key, value in kw.items():
            if value is not None:
                axis = dimension_labels_list.index(key)
                dimension_labels_list.remove(key)
                if new_array is None:
                    new_array = self.as_array()
                new_array = new_array.take(indices=value, axis=axis)

        if new_array.ndim > 1:
            return DataContainer(new_array, False, dimension_labels_list, suppress_warning=True)
        from .vector_data import VectorData
        return VectorData(new_array, dimension_labels=dimension_labels_list)

    def reorder(self, order):
        '''
        reorders the data in memory as requested.

        :param order: ordered list of labels from self.dimension_labels
        :type order: list, sting
        '''
        try:
            if len(order) != len(self.shape):
                raise ValueError('The axes list for resorting must have {0} dimensions. Got {1}'.format(len(self.shape), len(order)))
        except TypeError as ae:
            raise ValueError('The order must be an iterable with __len__ implemented, like a list or a tuple. Got {}'.format(type(order)))

        correct = True
        for el in order:
            correct = correct and el in self.dimension_labels
        if not correct:
            raise ValueError('The axes list for resorting must contain the dimension_labels {0} got {1}'.format(self.dimension_labels, order))

        new_order = [0]*len(self.shape)
        dimension_labels_new = [0]*len(self.shape)

        for i, axis in enumerate(order):
            new_order[i] = self.dimension_labels.index(axis)
            dimension_labels_new[i] = axis

        self.array = numpy.ascontiguousarray(numpy.transpose(self.array, new_order))

        if self.geometry is None:
            self.dimension_labels = dimension_labels_new
        else:
            self.geometry.set_labels(dimension_labels_new)

    def fill(self, array, **dimension):
        '''fills the internal data array with the DataContainer, numpy array or number provided

        :param array: number, numpy array or DataContainer to copy into the DataContainer
        :type array: DataContainer or subclasses, numpy array or number
        :param dimension: dictionary, optional

        if the passed numpy array points to the same array that is contained in the DataContainer,
        it just returns

        In case a DataContainer or subclass is passed, there will be a check of the geometry,
        if present, and the array will be resorted if the data is not in the appropriate order.

        User may pass a named parameter to specify in which axis the fill should happen:

        dc.fill(some_data, vertical=1, horizontal_x=32)
        will copy the data in some_data into the data container.
        '''
        if id(array) == id(self.array):
            return
        if dimension == {}:
            if isinstance(array, numpy.ndarray):
                numpy.copyto(self.array, array)
            elif isinstance(array, Number):
                self.array.fill(array)
            elif issubclass(array.__class__ , DataContainer):

                try:
                    if self.dimension_labels != array.dimension_labels:
                        raise ValueError('Input array is not in the same order as destination array. Use "array.reorder()"')
                except AttributeError:
                    pass

                if self.array.shape == array.shape:
                    numpy.copyto(self.array, array.array)
                else:
                    raise ValueError('Cannot fill with the provided array.' + \
                                     'Expecting shape {0} got {1}'.format(
                                     self.shape,array.shape))
            else:
                raise TypeError('Can fill only with number, numpy array or DataContainer and subclasses. Got {}'.format(type(array)))
        else:

            axis = [':']* self.number_of_dimensions
            dimension_labels = tuple(self.dimension_labels)
            for k,v in dimension.items():
                i = dimension_labels.index(k)
                axis[i] = v

            command = 'self.array['
            i = 0
            for el in axis:
                if i > 0:
                    command += ','
                command += str(el)
                i+=1

            if isinstance(array, numpy.ndarray):
                command = command + "] = array[:]"
            elif issubclass(array.__class__, DataContainer):
                command = command + "] = array.as_array()[:]"
            elif isinstance (array, Number):
                command = command + "] = array"
            else:
                raise TypeError('Can fill only with number, numpy array or DataContainer and subclasses. Got {}'.format(type(array)))
            exec(command)


    def check_dimensions(self, other):
        return self.shape == other.shape

    ## algebra

    def __add__(self, other):
        return self.add(other)
    def __mul__(self, other):
        return self.multiply(other)
    def __sub__(self, other):
        return self.subtract(other)
    def __div__(self, other):
        return self.divide(other)
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
        return self * other
    # __rmul__

    def __rdiv__(self, other):
        tmp = self.power(-1)
        tmp *= other
        return tmp
    # __rdiv__
    def __rtruediv__(self, other):
        return self.__rdiv__(other)

    def __rpow__(self, other):
        if isinstance(other, Number) :
            fother = numpy.ones(numpy.shape(self.array)) * other
            return type(self)(fother ** self.array ,
                           dimension_labels=self.dimension_labels,
                           geometry=self.geometry)
    # __rpow__

    # in-place arithmetic operators:
    # (+=, -=, *=, /= , //=,
    # must return self

    def __iadd__(self, other):
        kw = {'out':self}
        return self.add(other, **kw)

    def __imul__(self, other):
        kw = {'out':self}
        return self.multiply(other, **kw)

    def __isub__(self, other):
        kw = {'out':self}
        return self.subtract(other, **kw)

    def __idiv__(self, other):
        kw = {'out':self}
        return self.divide(other, **kw)

    def __itruediv__(self, other):
        kw = {'out':self}
        return self.divide(other, **kw)

    def __neg__(self):
        '''negation operator'''
        return -1 * self

    def __str__ (self, representation=False):
        repres = ""
        repres += "Number of dimensions: {0}\n".format(self.number_of_dimensions)
        repres += "Shape: {0}\n".format(self.shape)
        repres += "Axis labels: {0}\n".format(self.dimension_labels)
        if representation:
            repres += "Representation: \n{0}\n".format(self.array)
        return repres

    def get_data_axes_order(self,new_order=None):
        '''returns the axes label of self as a list

        If new_order is None returns the labels of the axes as a sorted-by-key list.
        If new_order is a list of length number_of_dimensions, returns a list
        with the indices of the axes in new_order with respect to those in
        self.dimension_labels: i.e.
          >>> self.dimension_labels = {0:'horizontal',1:'vertical'}
          >>> new_order = ['vertical','horizontal']
          returns [1,0]
        '''
        if new_order is None:
            return self.dimension_labels
        else:
            if len(new_order) == self.number_of_dimensions:

                axes_order = [0]*len(self.shape)
                for i, axis in enumerate(new_order):
                    axes_order[i] = self.dimension_labels.index(axis)
                return axes_order
            else:
                raise ValueError(f"Expecting {len(self.shape)} axes, got {len(new_order)}")

    def clone(self):
        '''returns a copy of DataContainer'''
        return copy.deepcopy(self)

    def copy(self):
        '''alias of clone'''
        return self.clone()

    ## binary operations

    def pixel_wise_binary(self, pwop, x2, *args,  **kwargs):
        out = kwargs.get('out', None)

        if out is None:
            if isinstance(x2, Number):
                out = pwop(self.as_array() , x2 , *args, **kwargs )
            elif issubclass(x2.__class__ , DataContainer):
                out = pwop(self.as_array() , x2.as_array() , *args, **kwargs )
            elif isinstance(x2, numpy.ndarray):
                out = pwop(self.as_array() , x2 , *args, **kwargs )
            else:
                raise TypeError('Expected x2 type as number or DataContainer, got {}'.format(type(x2)))
            geom = self.geometry
            if geom is not None:
                geom = self.geometry.copy()
            return type(self)(out,
                   deep_copy=False,
                   dimension_labels=self.dimension_labels,
                   geometry= None if self.geometry is None else self.geometry.copy(),
                   suppress_warning=True)


        elif issubclass(type(out), DataContainer) and issubclass(type(x2), DataContainer):
            if self.check_dimensions(out) and self.check_dimensions(x2):
                kwargs['out'] = out.as_array()
                pwop(self.as_array(), x2.as_array(), *args, **kwargs )
                #return type(self)(out.as_array(),
                #       deep_copy=False,
                #       dimension_labels=self.dimension_labels,
                #       geometry=self.geometry)
                return out
            raise ValueError(f"Wrong size for data memory: out {out.shape} x2 {x2.shape} expected {self.shape}")
        elif issubclass(type(out), DataContainer) and \
             isinstance(x2, (Number, numpy.ndarray)):
            if self.check_dimensions(out):
                if isinstance(x2, numpy.ndarray) and\
                    not (x2.shape == self.shape and x2.dtype == self.dtype):
                    raise ValueError(f"Wrong size for data memory: out {out.shape} x2 {x2.shape} expected {self.shape}")
                kwargs['out']=out.as_array()
                pwop(self.as_array(), x2, *args, **kwargs )
                return out
            raise ValueError(f"Wrong size for data memory: {out.shape} {self.shape}")
        elif issubclass(type(out), numpy.ndarray):
            if self.array.shape == out.shape and self.array.dtype == out.dtype:
                kwargs['out'] = out
                pwop(self.as_array(), x2, *args, **kwargs)
                #return type(self)(out,
                #       deep_copy=False,
                #       dimension_labels=self.dimension_labels,
                #       geometry=self.geometry)
        else:
            raise ValueError(f"incompatible class: {pwop.__name__} {type(out)}")

    def add(self, other, *args, **kwargs):
        if hasattr(other, '__container_priority__') and \
           self.__class__.__container_priority__ < other.__class__.__container_priority__:
            return other.add(self, *args, **kwargs)
        return self.pixel_wise_binary(numpy.add, other, *args, **kwargs)

    def subtract(self, other, *args, **kwargs):
        if hasattr(other, '__container_priority__') and \
           self.__class__.__container_priority__ < other.__class__.__container_priority__:
            return other.sapyb(-1,self,1, out=kwargs.get('out', None))
        return self.pixel_wise_binary(numpy.subtract, other, *args, **kwargs)

    def multiply(self, other, *args, **kwargs):
        if hasattr(other, '__container_priority__') and \
           self.__class__.__container_priority__ < other.__class__.__container_priority__:
            return other.multiply(self, *args, **kwargs)
        return self.pixel_wise_binary(numpy.multiply, other, *args, **kwargs)

    def divide(self, other, *args, **kwargs):
        if hasattr(other, '__container_priority__') and \
           self.__class__.__container_priority__ < other.__class__.__container_priority__:
            _out = other.divide(self, *args, **kwargs)
            _out.power(-1, out=_out)
            return _out
        return self.pixel_wise_binary(numpy.divide, other, *args, **kwargs)

    def power(self, other, *args, **kwargs):
        return self.pixel_wise_binary(numpy.power, other, *args, **kwargs)

    def maximum(self, x2, *args, **kwargs):
        return self.pixel_wise_binary(numpy.maximum, x2, *args, **kwargs)

    def minimum(self,x2, out=None, *args, **kwargs):
        return self.pixel_wise_binary(numpy.minimum, x2=x2, out=out, *args, **kwargs)


    def sapyb(self, a, y, b, out=None, num_threads=NUM_THREADS):
        '''performs a*self + b * y. Can be done in-place

        Parameters
        ----------
        a : multiplier for self, can be a number or a numpy array or a DataContainer
        y : DataContainer
        b : multiplier for y, can be a number or a numpy array or a DataContainer
        out : return DataContainer, if None a new DataContainer is returned, default None.
            out can be self or y.
        num_threads : number of threads to use during the calculation, using the CIL C library
            It will try to use the CIL C library and default to numpy operations, in case the C library does not handle the types.


        Example
        -------

        >>> a = 2
        >>> b = 3
        >>> ig = ImageGeometry(10,11)
        >>> x = ig.allocate(1)
        >>> y = ig.allocate(2)
        >>> out = x.sapyb(a,y,b)
        '''

        if out is None:
            out = self * 0.

        if out.dtype in [ numpy.float32, numpy.float64 ]:
            # handle with C-lib _axpby
            try:
                self._axpby(a, b, y, out, out.dtype, num_threads)
                return out
            except RuntimeError as rte:
                warnings.warn("sapyb defaulting to Python due to: {}".format(rte))
            except TypeError as te:
                warnings.warn("sapyb defaulting to Python due to: {}".format(te))
            finally:
                pass


        # cannot be handled by _axpby
        ax = self * a
        y.multiply(b, out=out)
        out.add(ax, out=out)
        return out

    def _axpby(self, a, b, y, out, dtype=numpy.float32, num_threads=NUM_THREADS):
        '''performs axpby with cilacc C library, can be done in-place.

        Does the operation .. math:: a*x+b*y and stores the result in out, where x is self

        :param a: scalar
        :type a: float
        :param b: scalar
        :type b: float
        :param y: DataContainer
        :param out: DataContainer instance to store the result
        :param dtype: data type of the DataContainers
        :type dtype: numpy type, optional, default numpy.float32
        :param num_threads: number of threads to run on
        :type num_threads: int, optional, default 1/2 CPU of the system
        '''

        c_float_p = ctypes.POINTER(ctypes.c_float)
        c_double_p = ctypes.POINTER(ctypes.c_double)

        #convert a and b to numpy arrays and get the reference to the data (length = 1 or ndx.size)
        try:
            nda = a.as_array()
        except:
            nda = numpy.asarray(a)

        try:
            ndb = b.as_array()
        except:
            ndb = numpy.asarray(b)

        a_vec = 0
        if nda.size > 1:
            a_vec = 1

        b_vec = 0
        if ndb.size > 1:
            b_vec = 1

        # get the reference to the data
        ndx = self.as_array()
        ndy = y.as_array()
        ndout = out.as_array()

        if ndout.dtype != dtype:
            raise Warning("out array of type {0} does not match requested dtype {1}. Using {0}".format(ndout.dtype, dtype))
            dtype = ndout.dtype
        if ndx.dtype != dtype:
            ndx = ndx.astype(dtype, casting='safe')
        if ndy.dtype != dtype:
            ndy = ndy.astype(dtype, casting='safe')
        if nda.dtype != dtype:
            nda = nda.astype(dtype, casting='same_kind')
        if ndb.dtype != dtype:
            ndb = ndb.astype(dtype, casting='same_kind')

        if dtype == numpy.float32:
            x_p = ndx.ctypes.data_as(c_float_p)
            y_p = ndy.ctypes.data_as(c_float_p)
            out_p = ndout.ctypes.data_as(c_float_p)
            a_p = nda.ctypes.data_as(c_float_p)
            b_p = ndb.ctypes.data_as(c_float_p)
            f = cilacc.saxpby

        elif dtype == numpy.float64:
            x_p = ndx.ctypes.data_as(c_double_p)
            y_p = ndy.ctypes.data_as(c_double_p)
            out_p = ndout.ctypes.data_as(c_double_p)
            a_p = nda.ctypes.data_as(c_double_p)
            b_p = ndb.ctypes.data_as(c_double_p)
            f = cilacc.daxpby

        else:
            raise TypeError('Unsupported type {}. Expecting numpy.float32 or numpy.float64'.format(dtype))

        #out = numpy.empty_like(a)


        # int psaxpby(float * x, float * y, float * out, float a, float b, long size)
        cilacc.saxpby.argtypes = [ctypes.POINTER(ctypes.c_float),  # pointer to the first array
                                  ctypes.POINTER(ctypes.c_float),  # pointer to the second array
                                  ctypes.POINTER(ctypes.c_float),  # pointer to the third array
                                  ctypes.POINTER(ctypes.c_float),  # pointer to A
                                  ctypes.c_int,                    # type of type of A selector (int)
                                  ctypes.POINTER(ctypes.c_float),  # pointer to B
                                  ctypes.c_int,                    # type of type of B selector (int)
                                  ctypes.c_longlong,               # type of size of first array
                                  ctypes.c_int]                    # number of threads
        cilacc.daxpby.argtypes = [ctypes.POINTER(ctypes.c_double), # pointer to the first array
                                  ctypes.POINTER(ctypes.c_double), # pointer to the second array
                                  ctypes.POINTER(ctypes.c_double), # pointer to the third array
                                  ctypes.POINTER(ctypes.c_double), # type of A (c_double)
                                  ctypes.c_int,                    # type of type of A selector (int)
                                  ctypes.POINTER(ctypes.c_double), # type of B (c_double)
                                  ctypes.c_int,                    # type of type of B selector (int)
                                  ctypes.c_longlong,               # type of size of first array
                                  ctypes.c_int]                    # number of threads

        if f(x_p, y_p, out_p, a_p, a_vec, b_p, b_vec, ndx.size, num_threads) != 0:
            raise RuntimeError('axpby execution failed')


    ## unary operations
    def pixel_wise_unary(self, pwop, *args,  **kwargs):
        out = kwargs.get('out', None)
        if out is None:
            out = pwop(self.as_array() , *args, **kwargs )
            return type(self)(out,
                       deep_copy=False,
                       dimension_labels=self.dimension_labels,
                       geometry=self.geometry,
                       suppress_warning=True)
        elif issubclass(type(out), DataContainer):
            if self.check_dimensions(out):
                kwargs['out'] = out.as_array()
                pwop(self.as_array(), *args, **kwargs )
            else:
                raise ValueError(f"Wrong size for data memory: {out.shape} {self.shape}")
        elif issubclass(type(out), numpy.ndarray):
            if self.array.shape == out.shape and self.array.dtype == out.dtype:
                kwargs['out'] = out
                pwop(self.as_array(), *args, **kwargs)
        else:
            raise ValueError("incompatible class: {pwop.__name__} {type(out)}")

    def abs(self, *args,  **kwargs):
        return self.pixel_wise_unary(numpy.abs, *args,  **kwargs)

    def sign(self, *args,  **kwargs):
        return self.pixel_wise_unary(numpy.sign, *args,  **kwargs)

    def sqrt(self, *args,  **kwargs):
        return self.pixel_wise_unary(numpy.sqrt, *args,  **kwargs)

    def conjugate(self, *args,  **kwargs):
        return self.pixel_wise_unary(numpy.conjugate, *args,  **kwargs)

    def exp(self, *args, **kwargs):
        '''Applies exp pixel-wise to the DataContainer'''
        return self.pixel_wise_unary(numpy.exp, *args, **kwargs)

    def log(self, *args, **kwargs):
        '''Applies log pixel-wise to the DataContainer'''
        return self.pixel_wise_unary(numpy.log, *args, **kwargs)

    ## reductions
    def squared_norm(self, **kwargs):
        '''return the squared euclidean norm of the DataContainer viewed as a vector'''
        #shape = self.shape
        #size = reduce(lambda x,y:x*y, shape, 1)
        #y = numpy.reshape(self.as_array(), (size, ))
        return self.dot(self)
        #return self.dot(self)
    def norm(self, **kwargs):
        '''return the euclidean norm of the DataContainer viewed as a vector'''
        return numpy.sqrt(self.squared_norm(**kwargs))

    def dot(self, other, *args, **kwargs):
        '''returns the inner product of 2 DataContainers viewed as vectors. Suitable for real and complex data.
          For complex data,  the dot method returns a.dot(b.conjugate())
        '''
        method = kwargs.get('method', 'numpy')
        if method not in ['numpy','reduce']:
            raise ValueError('dot: specified method not valid. Expecting numpy or reduce got {} '.format(
                    method))

        if self.shape == other.shape:
            if method == 'numpy':
                return numpy.dot(self.as_array().ravel(), other.as_array().ravel().conjugate())
            elif method == 'reduce':
                # see https://github.com/vais-ral/CCPi-Framework/pull/273
                # notice that Python seems to be smart enough to use
                # the appropriate type to hold the result of the reduction
                sf = reduce(lambda x,y: x + y[0]*y[1],
                            zip(self.as_array().ravel(),
                                other.as_array().ravel().conjugate()),
                            0)
                return sf
        else:
            raise ValueError('Shapes are not aligned: {} != {}'.format(self.shape, other.shape))

    def _directional_reduction_unary(self, reduction_function, axis=None, out=None, *args, **kwargs):
        """
        Returns the result of a unary function, considering the direction from an axis argument to the function

        Parameters
        ----------
        reduction_function : function
            The unary function to be evaluated
        axis : string or tuple of strings or int or tuple of ints, optional
            Specify the axis or axes to calculate 'reduction_function' along. Can be specified as
            string(s) of dimension_labels or int(s) of indices
            Default None calculates the function over the whole array
        out: ndarray or DataContainer, optional
            Provide an object in which to place the result. The object must have the correct dimensions and
            (for DataContainers) the correct dimension_labels, but the type will be cast if necessary. See
            `Output type determination <https://numpy.org/doc/stable/user/basics.ufuncs.html#ufuncs-output-type>`_ for more details.
            Default is None

        Returns
        -------
        scalar or ndarray
            The result of the unary function
        """
        if axis is not None:
            axis = self.get_dimension_axis(axis)

        if out is None:
            result = reduction_function(self.as_array(), axis=axis, *args, **kwargs)
            if isinstance(result, numpy.ndarray):
                new_dimensions = numpy.array(self.dimension_labels)
                new_dimensions = numpy.delete(new_dimensions, axis)
                return DataContainer(result, dimension_labels=new_dimensions)
            else:
                return result
        else:
            if hasattr(out,'array'):
                out_arr = out.array
            else:
                out_arr = out

            reduction_function(self.as_array(), out=out_arr, axis=axis,  *args, **kwargs)

    def sum(self, axis=None, out=None, *args, **kwargs):
        """
        Returns the sum of values in the DataContainer

        Parameters
        ----------
        axis : string or tuple of strings or int or tuple of ints, optional
            Specify the axis or axes to calculate 'sum' along. Can be specified as
            string(s) of dimension_labels or int(s) of indices
            Default None calculates the function over the whole array
        out : ndarray or DataContainer, optional
            Provide an object in which to place the result. The object must have the correct dimensions and
            (for DataContainers) the correct dimension_labels, but the type will be cast if necessary. See
            `Output type determination <https://numpy.org/doc/stable/user/basics.ufuncs.html#ufuncs-output-type>`_ for more details.
            Default is None

        Returns
        -------
        scalar or DataContainer
            The sum as a scalar or inside a DataContainer with reduced dimension_labels
            Default is to accumulate and return data as float64 or complex128
        """
        if kwargs.get('dtype') is not None:
            warnings.warn("dtype is ignored (auto-using float64 or complex128)", DeprecationWarning, stacklevel=2)

        if numpy.isrealobj(self.array):
            kwargs['dtype'] = numpy.float64
        else:
            kwargs['dtype'] = numpy.complex128

        return self._directional_reduction_unary(numpy.sum, axis=axis, out=out, *args, **kwargs)

    def min(self, axis=None, out=None, *args, **kwargs):
        """
        Returns the minimum pixel value in the DataContainer

        Parameters
        ----------
        axis : string or tuple of strings or int or tuple of ints, optional
            Specify the axis or axes to calculate 'min' along. Can be specified as
            string(s) of dimension_labels or int(s) of indices
            Default None calculates the function over the whole array
        out : ndarray or DataContainer, optional
            Provide an object in which to place the result. The object must have the correct dimensions and
            (for DataContainers) the correct dimension_labels, but the type will be cast if necessary.  See
            `Output type determination <https://numpy.org/doc/stable/user/basics.ufuncs.html#ufuncs-output-type>`_ for more details.
            Default is None

        Returns
        -------
        scalar or DataContainer
            The min as a scalar or inside a DataContainer with reduced dimension_labels
        """
        return self._directional_reduction_unary(numpy.min, axis=axis, out=out, *args, **kwargs)

    def max(self, axis=None, out=None, *args, **kwargs):
        """
        Returns the maximum pixel value in the DataContainer

        Parameters
        ----------
        axis : string or tuple of strings or int or tuple of ints, optional
            Specify the axis or axes to calculate 'max' along. Can be specified as
            string(s) of dimension_labels or int(s) of indices
            Default None calculates the function over the whole array
        out : ndarray or DataContainer, optional
            Provide an object in which to place the result. The object must have the correct dimensions and
            (for DataContainers) the correct dimension_labels, but the type will be cast if necessary. See
            `Output type determination <https://numpy.org/doc/stable/user/basics.ufuncs.html#ufuncs-output-type>`_ for more details.
            Default is None

        Returns
        -------
        scalar or DataContainer
            The max as a scalar or inside a DataContainer with reduced dimension_labels
        """
        return self._directional_reduction_unary(numpy.max, axis=axis, out=out, *args, **kwargs)

    def mean(self, axis=None, out=None, *args, **kwargs):
        """
        Returns the mean pixel value of the DataContainer

        Parameters
        ----------
        axis : string or tuple of strings or int or tuple of ints, optional
            Specify the axis or axes to calculate 'mean' along. Can be specified as
            string(s) of dimension_labels or int(s) of indices
            Default None calculates the function over the whole array
        out : ndarray or DataContainer, optional
            Provide an object in which to place the result. The object must have the correct dimensions and
            (for DataContainers) the correct dimension_labels, but the type will be cast if necessary. See
            `Output type determination <https://numpy.org/doc/stable/user/basics.ufuncs.html#ufuncs-output-type>`_ for more details.
            Default is None

        Returns
        -------
        scalar or DataContainer
            The mean as a scalar or inside a DataContainer with reduced dimension_labels
            Default is to accumulate and return data as float64 or complex128
        """

        if kwargs.get('dtype') is not None:
            warnings.warn("dtype is ignored (auto-using float64 or complex128)", DeprecationWarning, stacklevel=2)

        if numpy.isrealobj(self.array):
            kwargs['dtype'] = numpy.float64
        else:
            kwargs['dtype'] = numpy.complex128

        return self._directional_reduction_unary(numpy.mean, axis=axis, out=out, *args, **kwargs)

    # Logic operators between DataContainers and floats
    def __le__(self, other):
        '''Returns boolean array of DataContainer less or equal than DataContainer/float'''
        if isinstance(other, DataContainer):
            return self.as_array()<=other.as_array()
        return self.as_array()<=other

    def __lt__(self, other):
        '''Returns boolean array of DataContainer less than DataContainer/float'''
        if isinstance(other, DataContainer):
            return self.as_array()<other.as_array()
        return self.as_array()<other

    def __ge__(self, other):
        '''Returns boolean array of DataContainer greater or equal than DataContainer/float'''
        if isinstance(other, DataContainer):
            return self.as_array()>=other.as_array()
        return self.as_array()>=other

    def __gt__(self, other):
        '''Returns boolean array of DataContainer greater than DataContainer/float'''
        if isinstance(other, DataContainer):
            return self.as_array()>other.as_array()
        return self.as_array()>other

    def __eq__(self, other):
        '''Returns boolean array of DataContainer equal to DataContainer/float'''
        if isinstance(other, DataContainer):
            return self.as_array()==other.as_array()
        return self.as_array()==other

    def __ne__(self, other):
        '''Returns boolean array of DataContainer negative to DataContainer/float'''
        if isinstance(other, DataContainer):
            return self.as_array()!=other.as_array()
        return self.as_array()!=other
