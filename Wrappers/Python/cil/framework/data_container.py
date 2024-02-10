import copy
import ctypes
import logging
import warnings
from functools import reduce
from numbers import Number

import numpy

from .label import image_labels, data_order, get_order_for_engine
from .cilacc import cilacc
from cil.utilities.multiprocessing import NUM_THREADS


def message(cls, msg, *args):
    msg = "{0}: " + msg
    for i in range(len(args)):
        msg += " {%d}" %(i+1)
    args = list(args)
    args.insert(0, cls.__name__ )

    return msg.format(*args )


class ImageGeometry(object):

    @property
    def shape(self):

        shape_dict = {image_labels["CHANNEL"]: self.channels,
                      image_labels["VERTICAL"]: self.voxel_num_z,
                      image_labels["HORIZONTAL_Y"]: self.voxel_num_y,
                      image_labels["HORIZONTAL_X"]: self.voxel_num_x}

        shape = []
        for label in self.dimension_labels:
            shape.append(shape_dict[label])

        return tuple(shape)

    @shape.setter
    def shape(self, val):
        print("Deprecated - shape will be set automatically")

    @property
    def spacing(self):

        spacing_dict = {image_labels["CHANNEL"]: self.channel_spacing,
                        image_labels["VERTICAL"]: self.voxel_size_z,
                        image_labels["HORIZONTAL_Y"]: self.voxel_size_y,
                        image_labels["HORIZONTAL_X"]: self.voxel_size_x}

        spacing = []
        for label in self.dimension_labels:
            spacing.append(spacing_dict[label])

        return tuple(spacing)

    @property
    def length(self):
        return len(self.dimension_labels)

    @property
    def ndim(self):
        return len(self.dimension_labels)

    @property
    def dimension_labels(self):

        labels_default = data_order["CIL_IG_LABELS"]

        shape_default = [   self.channels,
                            self.voxel_num_z,
                            self.voxel_num_y,
                            self.voxel_num_x]

        try:
            labels = list(self._dimension_labels)
        except AttributeError:
            labels = labels_default.copy()

        for i, x in enumerate(shape_default):
            if x == 0 or x==1:
                try:
                    labels.remove(labels_default[i])
                except ValueError:
                    pass #if not in custom list carry on
        return tuple(labels)

    @dimension_labels.setter
    def dimension_labels(self, val):
        self.set_labels(val)

    def set_labels(self, labels):
        labels_default = data_order["CIL_IG_LABELS"]

        #check input and store. This value is not used directly
        if labels is not None:
            for x in labels:
                if x not in labels_default:
                    raise ValueError('Requested axis are not possible. Accepted label names {},\ngot {}'\
                        .format(labels_default,labels))

            self._dimension_labels = tuple(labels)

    def __eq__(self, other):

        if not isinstance(other, self.__class__):
            return False

        if self.voxel_num_x == other.voxel_num_x \
            and self.voxel_num_y == other.voxel_num_y \
            and self.voxel_num_z == other.voxel_num_z \
            and self.voxel_size_x == other.voxel_size_x \
            and self.voxel_size_y == other.voxel_size_y \
            and self.voxel_size_z == other.voxel_size_z \
            and self.center_x == other.center_x \
            and self.center_y == other.center_y \
            and self.center_z == other.center_z \
            and self.channels == other.channels \
            and self.channel_spacing == other.channel_spacing \
            and self.dimension_labels == other.dimension_labels:

            return True

        return False

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, val):
        self._dtype = val

    def __init__(self,
                 voxel_num_x=0,
                 voxel_num_y=0,
                 voxel_num_z=0,
                 voxel_size_x=1,
                 voxel_size_y=1,
                 voxel_size_z=1,
                 center_x=0,
                 center_y=0,
                 center_z=0,
                 channels=1,
                 **kwargs):

        self.voxel_num_x = int(voxel_num_x)
        self.voxel_num_y = int(voxel_num_y)
        self.voxel_num_z = int(voxel_num_z)
        self.voxel_size_x = float(voxel_size_x)
        self.voxel_size_y = float(voxel_size_y)
        self.voxel_size_z = float(voxel_size_z)
        self.center_x = center_x
        self.center_y = center_y
        self.center_z = center_z
        self.channels = channels
        self.channel_labels = None
        self.channel_spacing = 1.0
        self.dimension_labels = kwargs.get('dimension_labels', None)
        self.dtype = kwargs.get('dtype', numpy.float32)


    def get_slice(self,channel=None, vertical=None, horizontal_x=None, horizontal_y=None):
        '''
        Returns a new ImageGeometry of a single slice of in the requested direction.
        '''
        geometry_new = self.copy()
        if channel is not None:
            geometry_new.channels = 1

            try:
                geometry_new.channel_labels = [self.channel_labels[channel]]
            except:
                geometry_new.channel_labels = None

        if vertical is not None:
            geometry_new.voxel_num_z = 0

        if horizontal_y is not None:
            geometry_new.voxel_num_y = 0

        if horizontal_x is not None:
            geometry_new.voxel_num_x = 0

        return geometry_new

    def get_order_by_label(self, dimension_labels, default_dimension_labels):
        order = []
        for i, el in enumerate(default_dimension_labels):
            for j, ek in enumerate(dimension_labels):
                if el == ek:
                    order.append(j)
                    break
        return order

    def get_min_x(self):
        return self.center_x - 0.5*self.voxel_num_x*self.voxel_size_x

    def get_max_x(self):
        return self.center_x + 0.5*self.voxel_num_x*self.voxel_size_x

    def get_min_y(self):
        return self.center_y - 0.5*self.voxel_num_y*self.voxel_size_y

    def get_max_y(self):
        return self.center_y + 0.5*self.voxel_num_y*self.voxel_size_y

    def get_min_z(self):
        if not self.voxel_num_z == 0:
            return self.center_z - 0.5*self.voxel_num_z*self.voxel_size_z
        else:
            return 0

    def get_max_z(self):
        if not self.voxel_num_z == 0:
            return self.center_z + 0.5*self.voxel_num_z*self.voxel_size_z
        else:
            return 0

    def clone(self):
        '''returns a copy of the ImageGeometry'''
        return copy.deepcopy(self)

    def copy(self):
        '''alias of clone'''
        return self.clone()

    def __str__ (self):
        repres = ""
        repres += "Number of channels: {0}\n".format(self.channels)
        repres += "channel_spacing: {0}\n".format(self.channel_spacing)

        if self.voxel_num_z > 0:
            repres += "voxel_num : x{0},y{1},z{2}\n".format(self.voxel_num_x, self.voxel_num_y, self.voxel_num_z)
            repres += "voxel_size : x{0},y{1},z{2}\n".format(self.voxel_size_x, self.voxel_size_y, self.voxel_size_z)
            repres += "center : x{0},y{1},z{2}\n".format(self.center_x, self.center_y, self.center_z)
        else:
            repres += "voxel_num : x{0},y{1}\n".format(self.voxel_num_x, self.voxel_num_y)
            repres += "voxel_size : x{0},y{1}\n".format(self.voxel_size_x, self.voxel_size_y)
            repres += "center : x{0},y{1}\n".format(self.center_x, self.center_y)

        return repres
    def allocate(self, value=0, **kwargs):
        '''allocates an ImageData according to the size expressed in the instance

        :param value: accepts numbers to allocate an uniform array, or a string as 'random' or 'random_int' to create a random array or None.
        :type value: number or string, default None allocates empty memory block, default 0
        :param dtype: numerical type to allocate
        :type dtype: numpy type, default numpy.float32
        '''

        dtype = kwargs.get('dtype', self.dtype)

        if kwargs.get('dimension_labels', None) is not None:
            raise ValueError("Deprecated: 'dimension_labels' cannot be set with 'allocate()'. Use 'geometry.set_labels()' to modify the geometry before using allocate.")

        out = ImageData(geometry=self.copy(),
                            dtype=dtype,
                            suppress_warning=True)

        if isinstance(value, Number):
            # it's created empty, so we make it 0
            out.array.fill(value)
        else:
            if value == image_labels["RANDOM"]:
                seed = kwargs.get('seed', None)
                if seed is not None:
                    numpy.random.seed(seed)
                if numpy.iscomplexobj(out.array):
                    r = numpy.random.random_sample(self.shape) + 1j * numpy.random.random_sample(self.shape)
                    out.fill(r)
                else:
                    out.fill(numpy.random.random_sample(self.shape))
            elif value == image_labels["RANDOM_INT"]:
                seed = kwargs.get('seed', None)
                if seed is not None:
                    numpy.random.seed(seed)
                max_value = kwargs.get('max_value', 100)
                r = numpy.random.randint(max_value,size=self.shape, dtype=numpy.int32)
                out.fill(numpy.asarray(r, dtype=self.dtype))
            elif value is None:
                pass
            else:
                raise ValueError('Value {} unknown'.format(value))

        return out


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
        elif len(list(val))==self.number_of_dimensions:
            self._dimension_labels = tuple(val)
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

    @shape.setter
    def shape(self, val):
        print("Deprecated - shape will be set automatically")

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
        '''Holds the data'''

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
            self.geometry = kwargs['geometry']
            try:
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
                    new_array = self.as_array().take(indices=value, axis=axis)
                else:
                    new_array = new_array.take(indices=value, axis=axis)

        if new_array.ndim > 1:
            return DataContainer(new_array, False, dimension_labels_list, suppress_warning=True)
        else:
            return VectorData(new_array, dimension_labels=dimension_labels_list)

    def reorder(self, order=None):
        '''
        reorders the data in memory as requested.

        :param order: ordered list of labels from self.dimension_labels, or order for engine 'astra' or 'tigre'
        :type order: list, sting
        '''

        if order in data_order["ENGINES"]:
            order = get_order_for_engine(order, self.geometry)

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
                if array.shape != self.shape:
                    raise ValueError('Cannot fill with the provided array.' + \
                                     'Expecting shape {0} got {1}'.format(
                                     self.shape,array.shape))
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
            dimension_labels = list(self.dimension_labels)
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
                raise ValueError('Expecting {0} axes, got {2}'\
                                 .format(len(self.shape),len(new_order)))

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
            else:
                raise ValueError(message(type(self),"Wrong size for data memory: out {} x2 {} expected {}".format( out.shape,x2.shape ,self.shape)))
        elif issubclass(type(out), DataContainer) and \
             isinstance(x2, (Number, numpy.ndarray)):
            if self.check_dimensions(out):
                if isinstance(x2, numpy.ndarray) and\
                    not (x2.shape == self.shape and x2.dtype == self.dtype):
                    raise ValueError(message(type(self),
                        "Wrong size for data memory: out {} x2 {} expected {}"\
                            .format( out.shape,x2.shape ,self.shape)))
                kwargs['out']=out.as_array()
                pwop(self.as_array(), x2, *args, **kwargs )
                return out
            else:
                raise ValueError(message(type(self),"Wrong size for data memory: ", out.shape,self.shape))
        elif issubclass(type(out), numpy.ndarray):
            if self.array.shape == out.shape and self.array.dtype == out.dtype:
                kwargs['out'] = out
                pwop(self.as_array(), x2, *args, **kwargs)
                #return type(self)(out,
                #       deep_copy=False,
                #       dimension_labels=self.dimension_labels,
                #       geometry=self.geometry)
        else:
            raise ValueError (message(type(self),  "incompatible class:" , pwop.__name__, type(out)))

    def add(self, other, *args, **kwargs):
        if hasattr(other, '__container_priority__') and \
           self.__class__.__container_priority__ < other.__class__.__container_priority__:
            return other.add(self, *args, **kwargs)
        return self.pixel_wise_binary(numpy.add, other, *args, **kwargs)

    def subtract(self, other, *args, **kwargs):
        if hasattr(other, '__container_priority__') and \
           self.__class__.__container_priority__ < other.__class__.__container_priority__:
            return other.subtract(self, *args, **kwargs)
        return self.pixel_wise_binary(numpy.subtract, other, *args, **kwargs)

    def multiply(self, other, *args, **kwargs):
        if hasattr(other, '__container_priority__') and \
           self.__class__.__container_priority__ < other.__class__.__container_priority__:
            return other.multiply(self, *args, **kwargs)
        return self.pixel_wise_binary(numpy.multiply, other, *args, **kwargs)

    def divide(self, other, *args, **kwargs):
        if hasattr(other, '__container_priority__') and \
           self.__class__.__container_priority__ < other.__class__.__container_priority__:
            return other.divide(self, *args, **kwargs)
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
        ret_out = False

        if out is None:
            out = self * 0.
            ret_out = True

        if out.dtype in [ numpy.float32, numpy.float64 ]:
            # handle with C-lib _axpby
            try:
                self._axpby(a, b, y, out, out.dtype, num_threads)
                if ret_out:
                    return out
                return
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

        if ret_out:
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
                                  ctypes.c_int,  # type of type of A selector (int)
                                  ctypes.POINTER(ctypes.c_float),  # pointer to B
                                  ctypes.c_int,  # type of type of B selector (int)
                                  ctypes.c_longlong,  # type of size of first array
                                  ctypes.c_int]                    # number of threads
        cilacc.daxpby.argtypes = [ctypes.POINTER(ctypes.c_double),  # pointer to the first array
                                  ctypes.POINTER(ctypes.c_double),  # pointer to the second array
                                  ctypes.POINTER(ctypes.c_double),  # pointer to the third array
                                  ctypes.POINTER(ctypes.c_double),  # type of A (c_double)
                                  ctypes.c_int,  # type of type of A selector (int)
                                  ctypes.POINTER(ctypes.c_double),  # type of B (c_double)
                                  ctypes.c_int,  # type of type of B selector (int)
                                  ctypes.c_longlong,  # type of size of first array
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
                raise ValueError(message(type(self),"Wrong size for data memory: ", out.shape,self.shape))
        elif issubclass(type(out), numpy.ndarray):
            if self.array.shape == out.shape and self.array.dtype == out.dtype:
                kwargs['out'] = out
                pwop(self.as_array(), *args, **kwargs)
        else:
            raise ValueError (message(type(self),  "incompatible class:" , pwop.__name__, type(out)))

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
            logging.WARNING("dtype argument is ignored, using float64 or complex128")

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

        if kwargs.get('dtype', None) is not None:
            logging.WARNING("dtype argument is ignored, using float64 or complex128")

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


class ImageData(DataContainer):
    '''DataContainer for holding 2D or 3D DataContainer'''
    __container_priority__ = 1

    @property
    def geometry(self):
        return self._geometry

    @geometry.setter
    def geometry(self, val):
        self._geometry = val

    @property
    def dimension_labels(self):
        return self.geometry.dimension_labels

    @dimension_labels.setter
    def dimension_labels(self, val):
        if val is not None:
            raise ValueError("Unable to set the dimension_labels directly. Use geometry.set_labels() instead")

    def __init__(self,
                 array = None,
                 deep_copy=False,
                 geometry=None,
                 **kwargs):

        dtype = kwargs.get('dtype', numpy.float32)


        if geometry is None:
            raise AttributeError("ImageData requires a geometry")


        labels = kwargs.get('dimension_labels', None)
        if labels is not None and labels != geometry.dimension_labels:
                raise ValueError("Deprecated: 'dimension_labels' cannot be set with 'allocate()'. Use 'geometry.set_labels()' to modify the geometry before using allocate.")

        if array is None:
            array = numpy.empty(geometry.shape, dtype=dtype)
        elif issubclass(type(array) , DataContainer):
            array = array.as_array()
        elif issubclass(type(array) , numpy.ndarray):
            pass
        else:
            raise TypeError('array must be a CIL type DataContainer or numpy.ndarray got {}'.format(type(array)))

        if array.shape != geometry.shape:
            raise ValueError('Shape mismatch {} {}'.format(array.shape, geometry.shape))

        if array.ndim not in [2,3,4]:
            raise ValueError('Number of dimensions are not 2 or 3 or 4 : {0}'.format(array.ndim))

        super(ImageData, self).__init__(array, deep_copy, geometry=geometry, **kwargs)


    def get_slice(self,channel=None, vertical=None, horizontal_x=None, horizontal_y=None, force=False):
        '''
        Returns a new ImageData of a single slice of in the requested direction.
        '''
        try:
            geometry_new = self.geometry.get_slice(channel=channel, vertical=vertical, horizontal_x=horizontal_x, horizontal_y=horizontal_y)
        except ValueError:
            if force:
                geometry_new = None
            else:
                raise ValueError ("Unable to return slice of requested ImageData. Use 'force=True' to return DataContainer instead.")

        #if vertical = 'centre' slice convert to index and subset, this will interpolate 2 rows to get the center slice value
        if vertical == 'centre':
            dim = self.geometry.dimension_labels.index('vertical')
            centre_slice_pos = (self.geometry.shape[dim]-1) / 2.
            ind0 = int(numpy.floor(centre_slice_pos))

            w2 = centre_slice_pos - ind0
            out = DataContainer.get_slice(self, channel=channel, vertical=ind0, horizontal_x=horizontal_x, horizontal_y=horizontal_y)

            if w2 > 0:
                out2 = DataContainer.get_slice(self, channel=channel, vertical=ind0 + 1, horizontal_x=horizontal_x, horizontal_y=horizontal_y)
                out = out * (1 - w2) + out2 * w2
        else:
            out = DataContainer.get_slice(self, channel=channel, vertical=vertical, horizontal_x=horizontal_x, horizontal_y=horizontal_y)

        if len(out.shape) == 1 or geometry_new is None:
            return out
        else:
            return ImageData(out.array, deep_copy=False, geometry=geometry_new, suppress_warning=True)


    def apply_circular_mask(self, radius=0.99, in_place=True):
        """

        Apply a circular mask to the horizontal_x and horizontal_y slices. Values outside this mask will be set to zero.

        This will most commonly be used to mask edge artefacts from standard CT reconstructions with FBP.

        Parameters
        ----------
        radius : float, default 0.99
            radius of mask by percentage of size of horizontal_x or horizontal_y, whichever is greater

        in_place : boolean, default True
            If `True` masks the current data, if `False` returns a new `ImageData` object.


        Returns
        -------
        ImageData
            If `in_place = False` returns a new ImageData object with the masked data

        """
        ig = self.geometry

        # grid
        y_range = (ig.voxel_num_y-1)/2
        x_range = (ig.voxel_num_x-1)/2

        Y, X = numpy.ogrid[-y_range:y_range+1,-x_range:x_range+1]

        # use centre from geometry in units distance to account for aspect ratio of pixels
        dist_from_center = numpy.sqrt((X*ig.voxel_size_x+ ig.center_x)**2 + (Y*ig.voxel_size_y+ig.center_y)**2)

        size_x = ig.voxel_num_x * ig.voxel_size_x
        size_y = ig.voxel_num_y * ig.voxel_size_y

        if size_x > size_y:
            radius_applied =radius * size_x/2
        else:
            radius_applied =radius * size_y/2

        # approximate the voxel as a circle and get the radius
        # ie voxel area = 1, circle of area=1 has r = 0.56
        r=((ig.voxel_size_x * ig.voxel_size_y )/numpy.pi)**(1/2)

        # we have the voxel centre distance to mask. voxels with distance greater than |r| are fully inside or outside.
        # values on the border region between -r and r are preserved
        mask =(radius_applied-dist_from_center).clip(-r,r)

        #  rescale to -pi/2->+pi/2
        mask *= (0.5*numpy.pi)/r

        # the sin of the linear distance gives us an approximation of area of the circle to include in the mask
        numpy.sin(mask, out = mask)

        # rescale the data 0 - 1
        mask = 0.5 + mask * 0.5

        # reorder dataset so 'horizontal_y' and 'horizontal_x' are the final dimensions
        labels_orig = self.dimension_labels
        labels = list(labels_orig)

        labels.remove('horizontal_y')
        labels.remove('horizontal_x')
        labels.append('horizontal_y')
        labels.append('horizontal_x')


        if in_place == True:
            self.reorder(labels)
            numpy.multiply(self.array, mask, out=self.array)
            self.reorder(labels_orig)

        else:
            image_data_out = self.copy()
            image_data_out.reorder(labels)
            numpy.multiply(image_data_out.array, mask, out=image_data_out.array)
            image_data_out.reorder(labels_orig)

            return image_data_out


class VectorData(DataContainer):
    '''DataContainer to contain 1D array'''

    @property
    def geometry(self):
        return self._geometry

    @geometry.setter
    def geometry(self, val):
        self._geometry = val

    @property
    def dimension_labels(self):
        if hasattr(self,'geometry'):
            return self.geometry.dimension_labels
        else:
            return self._dimension_labels

    @dimension_labels.setter
    def dimension_labels(self, val):
        if hasattr(self,'geometry'):
            self.geometry.dimension_labels = val

        self._dimension_labels = val

    def __init__(self, array=None, **kwargs):
        self.geometry = kwargs.get('geometry', None)

        dtype = kwargs.get('dtype', numpy.float32)

        if self.geometry is None:
            if array is None:
                raise ValueError('Please specify either a geometry or an array')
            else:
                if len(array.shape) > 1:
                    raise ValueError('Incompatible size: expected 1D got {}'.format(array.shape))
                out = array
                self.geometry = VectorGeometry(array.shape[0], **kwargs)
                self.length = self.geometry.length
        else:
            self.length = self.geometry.length

            if array is None:
                out = numpy.zeros((self.length,), dtype=dtype)
            else:
                if self.length == array.shape[0]:
                    out = array
                else:
                    raise ValueError('Incompatible size: expecting {} got {}'.format((self.length,), array.shape))
        deep_copy = True
        # need to pass the geometry, othewise None
        super(VectorData, self).__init__(out, deep_copy, self.geometry.dimension_labels, geometry = self.geometry)


class VectorGeometry(object):
    '''Geometry describing VectorData to contain 1D array'''
    RANDOM = 'random'
    RANDOM_INT = 'random_int'

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, val):
        self._dtype = val

    def __init__(self,
                 length, **kwargs):

        self.length = int(length)
        self.shape = (length, )
        self.dtype = kwargs.get('dtype', numpy.float32)
        self.dimension_labels = kwargs.get('dimension_labels', None)

    def clone(self):
        '''returns a copy of VectorGeometry'''
        return copy.deepcopy(self)

    def copy(self):
        '''alias of clone'''
        return self.clone()

    def __eq__(self, other):

        if not isinstance(other, self.__class__):
            return False

        if self.length == other.length \
            and self.shape == other.shape \
            and self.dimension_labels == other.dimension_labels:
            return True
        return False

    def __str__ (self):
        repres = ""
        repres += "Length: {0}\n".format(self.length)
        repres += "Shape: {0}\n".format(self.shape)
        repres += "Dimension_labels: {0}\n".format(self.dimension_labels)

        return repres

    def allocate(self, value=0, **kwargs):
        '''allocates an VectorData according to the size expressed in the instance

        :param value: accepts numbers to allocate an uniform array, or a string as 'random' or 'random_int' to create a random array or None.
        :type value: number or string, default None allocates empty memory block
        :param dtype: numerical type to allocate
        :type dtype: numpy type, default numpy.float32
        :param seed: seed for the random number generator
        :type seed: int, default None
        :param max_value: max value of the random int array
        :type max_value: int, default 100'''

        dtype = kwargs.get('dtype', self.dtype)
        # self.dtype = kwargs.get('dtype', numpy.float32)
        out = VectorData(geometry=self.copy(), dtype=dtype)
        if isinstance(value, Number):
            if value != 0:
                out += value
        else:
            if value == VectorGeometry.RANDOM:
                seed = kwargs.get('seed', None)
                if seed is not None:
                    numpy.random.seed(seed)
                out.fill(numpy.random.random_sample(self.shape))
            elif value == VectorGeometry.RANDOM_INT:
                seed = kwargs.get('seed', None)
                if seed is not None:
                    numpy.random.seed(seed)
                max_value = kwargs.get('max_value', 100)
                r = numpy.random.randint(max_value,size=self.shape, dtype=numpy.int32)
                out.fill(numpy.asarray(r, dtype=self.dtype))
            elif value is None:
                pass
            else:
                raise ValueError('Value {} unknown'.format(value))
        return out
