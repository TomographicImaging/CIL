# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library (CIL) developed by CCPi 
#   (Collaborative Computational Project in Tomographic Imaging), with 
#   substantial contributions by UKRI-STFC and University of Manchester.

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software 
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
import copy
import numpy
import warnings
from functools import reduce
from numbers import Number
import ctypes, platform
from ctypes import util
import math
from cil.utilities.multiprocessing import NUM_THREADS
# check for the extension

if platform.system() == 'Linux':
    dll = 'libcilacc.so'
elif platform.system() == 'Windows':
    dll_file = 'cilacc.dll'
    dll = util.find_library(dll_file)
elif platform.system() == 'Darwin':
    dll = 'libcilacc.dylib'
else:
    raise ValueError('Not supported platform, ', platform.system())

cilacc = ctypes.cdll.LoadLibrary(dll)

def find_key(dic, val):
    """return the key of dictionary dic given the value"""
    return [k for k, v in dic.items() if v == val][0]

def message(cls, msg, *args):
    msg = "{0}: " + msg
    for i in range(len(args)):
        msg += " {%d}" %(i+1)
    args = list(args)
    args.insert(0, cls.__name__ )
    
    return msg.format(*args )

class ImageGeometry(object):
    RANDOM = 'random'
    RANDOM_INT = 'random_int'
    CHANNEL = 'channel'
    VERTICAL = 'vertical'
    HORIZONTAL_X = 'horizontal_x'
    HORIZONTAL_Y = 'horizontal_y'
    
    @property
    def shape(self):

        shape_dict = {ImageGeometry.CHANNEL: self.channels,
                     ImageGeometry.VERTICAL: self.voxel_num_z,
                     ImageGeometry.HORIZONTAL_Y: self.voxel_num_y,        
                     ImageGeometry.HORIZONTAL_X: self.voxel_num_x}

        shape = []
        for label in self.dimension_labels:
            shape.append(shape_dict[label])

        return tuple(shape)

    @shape.setter
    def shape(self, val):
        print("Deprecated - shape will be set automatically")

    @property
    def spacing(self):

        spacing_dict = {ImageGeometry.CHANNEL: self.channel_spacing,
                        ImageGeometry.VERTICAL: self.voxel_size_z,
                        ImageGeometry.HORIZONTAL_Y: self.voxel_size_y,        
                        ImageGeometry.HORIZONTAL_X: self.voxel_size_x}

        spacing = []
        for label in self.dimension_labels:
            spacing.append(spacing_dict[label])

        return tuple(spacing)

    @property
    def length(self):
        return len(self.dimension_labels)

    @property
    def dimension_labels(self):
        
        labels_default = DataOrder.CIL_IG_LABELS

        shape_default = [   self.channels - 1, #channels default is 1
                            self.voxel_num_z,
                            self.voxel_num_y,
                            self.voxel_num_x]

        try:
            labels = list(self.__dimension_labels)
        except AttributeError:
            labels = labels_default.copy()

        for i, x in enumerate(shape_default):
            if x == 0:
                try:
                    labels.remove(labels_default[i])
                except ValueError:
                    pass #if not in custom list carry on
        return tuple(labels)
      
    @dimension_labels.setter
    def dimension_labels(self, val):
        self.set_labels(val)
    
    def set_labels(self, labels):
        labels_default = DataOrder.CIL_IG_LABELS

        #check input and store. This value is not used directly
        if labels is not None:
            for x in labels:
                if x not in labels_default:
                    raise ValueError('Requested axis are not possible. Accepted label names {},\ngot {}'\
                        .format(labels_default,labels))
                    
            self.__dimension_labels = tuple(labels)
        
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
        return self.__dtype

    @dtype.setter
    def dtype(self, val):
        self.__dtype = val           

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

    def subset(self, dimensions=None, **kw):
        '''Returns a new sliced and/or reshaped ImageGeometry'''

        if not kw.get('suppress_warning', False):
            warnings.warn('Subset has been deprecated and will be removed in following version. Use reorder() and get_slice() instead',
              DeprecationWarning)

        if dimensions is None:
            return self.get_slice(**kw)
        else:
            if len(dimensions) != len(self.dimension_labels):
                raise ValueError('The axes list for subset must contain the dimension_labels {0} got {1}'.format(self.dimension_labels, dimensions))
            
            temp = self.copy()
            temp.set_labels(dimensions)
            return temp      

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
            if value == ImageGeometry.RANDOM:
                seed = kwargs.get('seed', None)
                if seed is not None:
                    numpy.random.seed(seed)
                if dtype in [ numpy.complex , numpy.complex64 , numpy.complex128 ] :
                    r = numpy.random.random_sample(self.shape) + 1j * numpy.random.random_sample(self.shape)
                    out.fill(r)
                else: 
                    out.fill(numpy.random.random_sample(self.shape))
            elif value == ImageGeometry.RANDOM_INT:
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
    
class ComponentDescription(object):
    r'''This class enables the creation of vectors and unit vectors used to describe the components of a tomography system
     '''
    def __init__ (self, dof):
        self.__dof = dof

    @staticmethod  
    def CreateVector(val):
        try:
            vec = numpy.asarray(val, dtype=numpy.float32).reshape(len(val))
        except:
            raise ValueError("Can't convert to numpy array")
   
        return vec

    @staticmethod   
    def CreateUnitVector(val):
        vec = ComponentDescription.CreateVector(val)
        dot_product = vec.dot(vec)
        if abs(dot_product)>1e-8:
            vec = (vec/numpy.sqrt(dot_product))
        else:
            raise ValueError("Can't return a unit vector of a zero magnitude vector")
        return vec

    def length_check(self, val):
        try:
            val_length = len(val)
        except:
            raise ValueError("Vectors for {0}D geometries must have length = {0}. Got {1}".format(self.__dof,val))
        
        if val_length != self.__dof:
            raise ValueError("Vectors for {0}D geometries must have length = {0}. Got {1}".format(self.__dof,val))

class PositionVector(ComponentDescription):
    r'''This class creates a component of a tomography system with a position attribute
     '''
    @property
    def position(self):
        try:
            return self.__position
        except:
            raise AttributeError

    @position.setter
    def position(self, val):  
        self.length_check(val)
        self.__position = ComponentDescription.CreateVector(val)


class DirectionVector(ComponentDescription):
    r'''This class creates a component of a tomography system with a direction attribute
     '''
    @property
    def direction(self):      
        try:
            return self.__direction
        except:
            raise AttributeError

    @direction.setter
    def direction(self, val):
        self.length_check(val)    
        self.__direction = ComponentDescription.CreateUnitVector(val)

 
class PositionDirectionVector(PositionVector, DirectionVector):
    r'''This class creates a component of a tomography system with position and direction attributes
     '''
    pass

class Detector1D(PositionVector):
    r'''This class creates a component of a tomography system with position and direction_x attributes used for 1D panels
     '''
    @property
    def direction_x(self):
        try:
            return self.__direction_x
        except:
            raise AttributeError

    @direction_x.setter
    def direction_x(self, val):
        self.length_check(val)
        self.__direction_x = ComponentDescription.CreateUnitVector(val)

class Detector2D(PositionVector):
    r'''This class creates a component of a tomography system with position, direction_x and direction_y attributes used for 2D panels
     '''
    @property
    def direction_x(self):
        try:
            return self.__direction_x
        except:
            raise AttributeError

    @property
    def direction_y(self):
        try:
            return self.__direction_y
        except:
            raise AttributeError

    def set_direction(self, x, y):
        self.length_check(x)
        x = ComponentDescription.CreateUnitVector(x)

        self.length_check(y)
        y = ComponentDescription.CreateUnitVector(y)

        dot_product = x.dot(y)
        if not numpy.isclose(dot_product, 0):
            raise ValueError("vectors detector.direction_x and detector.direction_y must be orthogonal")

        self.__direction_y = y        
        self.__direction_x = x

class SystemConfiguration(object):
    r'''This is a generic class to hold the description of a tomography system
     '''

    SYSTEM_SIMPLE = 'simple' 
    SYSTEM_OFFSET = 'offset' 
    SYSTEM_ADVANCED = 'advanced' 

    @property
    def dimension(self):
        if self._dimension == 2:
            return '2D'
        else:
            return '3D'    

    @dimension.setter
    def dimension(self,val):
        if val != 2 and val != 3:
            raise ValueError('Can set up 2D and 3D systems only. got {0}D'.format(val))
        else:
            self._dimension = val

    @property
    def geometry(self):
        return self.__geometry

    @geometry.setter
    def geometry(self,val):
        if val != AcquisitionGeometry.CONE and val != AcquisitionGeometry.PARALLEL:
            raise ValueError('geom_type = {} not recognised please specify \'cone\' or \'parallel\''.format(val))
        else:
            self.__geometry = val

    def __init__(self, dof, geometry): 
        """Initialises the system component attributes for the acquisition type
        """                
        self.dimension = dof
        self.geometry = geometry
        
        if geometry == AcquisitionGeometry.PARALLEL:
            self.ray = DirectionVector(dof)
        else:
            self.source = PositionVector(dof)

        if dof == 2:
            self.detector = Detector1D(dof)
            self.rotation_axis = PositionVector(dof)
        else:
            self.detector = Detector2D(dof)
            self.rotation_axis = PositionDirectionVector(dof)
    
    def __str__(self):
        """Implements the string representation of the system configuration
        """   
        raise NotImplementedError

    def __eq__(self, other):
        """Implements the equality check of the system configuration
        """   
        raise NotImplementedError

    def update_reference_frame(self):
        """Returns the components of the system in the reference frame of the rotation axis at position 0
        """      
        raise NotImplementedError

    def get_centre_slice(self):
        """Returns the 2D system configuration corersponding to the centre slice
        """        
        raise NotImplementedError

    def calculate_magnification(self):
        r'''Calculates the magnification of the system using the source to rotate axis,
        and source to detector distance along the direction.

        :return: returns [dist_source_center, dist_center_detector, magnification],  [0] distance from the source to the rotate axis, [1] distance from the rotate axis to the detector, [2] magnification of the system
        :rtype: list
        '''
        raise NotImplementedError
  
    def system_description(self):
        r'''Returns `simple` if the the geometry matches the default definitions with no offsets or rotations,
            \nReturns `offset` if the the geometry matches the default definitions with centre-of-rotation or detector offsets
            \nReturns `advanced` if the the geometry has rotated or tilted rotation axis or detector, can also have offsets
        '''         
        raise NotImplementedError

    def copy(self):
        '''returns a copy of SystemConfiguration'''
        return copy.deepcopy(self)

class Parallel2D(SystemConfiguration):
    r'''This class creates the SystemConfiguration of a parallel beam 2D tomographic system
                       
    :param ray_direction: A 2D vector describing the x-ray direction (x,y)
    :type ray_direction: list, tuple, ndarray
    :param detector_pos: A 2D vector describing the position of the centre of the detector (x,y)
    :type detector_pos: list, tuple, ndarray
    :param detector_direction_x: A 2D vector describing the direction of the detector_x (x,y)
    :type detector_direction_x: list, tuple, ndarray
    :param rotation_axis_pos: A 2D vector describing the position of the axis of rotation (x,y)
    :type rotation_axis_pos: list, tuple, ndarray 
     '''
    def __init__ (self, ray_direction, detector_pos, detector_direction_x, rotation_axis_pos):
        """Constructor method
        """
        super(Parallel2D, self).__init__(dof=2, geometry = 'parallel')

        #source
        self.ray.direction = ray_direction

        #detector
        self.detector.position = detector_pos
        self.detector.direction_x = detector_direction_x
        
        #rotate axis
        self.rotation_axis.position = rotation_axis_pos

    def update_reference_frame(self):
        r'''Transforms the system origin to the rotate axis
        '''         
        self.detector.position -= self.rotation_axis.position
        self.rotation_axis.position = [0,0]


    def align_reference_frame(self):
        r'''Transforms the system origin to the rotate axis and aligns the ray along the positive Y direction
        '''          
        self.update_reference_frame()

        ray_vec = -self.ray.direction

        axis_rotation = numpy.eye(2)
        if numpy.allclose(ray_vec,[0,-1]):
            pass
        elif numpy.allclose(ray_vec,[0,1]):
            axis_rotation[0][0] = -1
            axis_rotation[1][1] = -1
        else:
            theta = math.atan2(ray_vec[0],-ray_vec[1])
            axis_rotation[0][0] = axis_rotation[1][1] = math.cos(theta)
            axis_rotation[0][1] = math.sin(theta)
            axis_rotation[1][0] = -math.sin(theta)

        rotation_matrix = numpy.matrix(axis_rotation)
        
        self.ray.direction = rotation_matrix.dot(self.ray.direction.reshape(2,1))
        self.detector.position = rotation_matrix.dot(self.detector.position.reshape(2,1))
        self.detector.direction_x = rotation_matrix.dot(self.detector.direction_x.reshape(2,1))

    def system_description(self):
        r'''Returns `simple` if the the geometry matches the default definitions with no offsets or rotations,
            \nReturns `offset` if the the geometry matches the default definitions with centre-of-rotation or detector offsets
            \nReturns `advanced` if the the geometry has rotated or tilted rotation axis or detector, can also have offsets
        '''       
        new = self.copy()
        new.align_reference_frame()

        try:
            det_unit = ComponentDescription.CreateUnitVector(new.detector.position)
        except ValueError: #pass test if detector is on origin
            det_unit = [0,1]

        if not numpy.allclose(new.ray.direction,[0,1]) or\
            not numpy.allclose(new.detector.direction_x,[1,0]):
            return SystemConfiguration.SYSTEM_ADVANCED
        elif not numpy.allclose(det_unit,[0,1]):
            return SystemConfiguration.SYSTEM_OFFSET
        else:
            return SystemConfiguration.SYSTEM_SIMPLE


    def __str__(self):
        def csv(val):
            return numpy.array2string(val, separator=', ')
        
        repres = "2D Parallel-beam tomography\n"
        repres += "System configuration:\n"
        repres += "\tRay direction: {0}\n".format(csv(self.ray.direction))
        repres += "\tRotation axis position: {0}\n".format(csv(self.rotation_axis.position))
        repres += "\tDetector position: {0}\n".format(csv(self.detector.position))
        repres += "\tDetector direction x: {0}\n".format(csv(self.detector.direction_x))
        return repres

    def __eq__(self, other):

        if not isinstance(other, self.__class__):
            return False
        
        if numpy.allclose(self.ray.direction, other.ray.direction) \
        and numpy.allclose(self.detector.position, other.detector.position)\
        and numpy.allclose(self.detector.direction_x, other.detector.direction_x)\
        and numpy.allclose(self.rotation_axis.position, other.rotation_axis.position):
            return True
        
        return False

    def get_centre_slice(self):
        return self

    def calculate_magnification(self):
        return [None, None, 1.0]

class Parallel3D(SystemConfiguration):
    r'''This class creates the SystemConfiguration of a parallel beam 3D tomographic system
                       
    :param ray_direction: A 3D vector describing the x-ray direction (x,y,z)
    :type ray_direction: list, tuple, ndarray
    :param detector_pos: A 3D vector describing the position of the centre of the detector (x,y,z)
    :type detector_pos: list, tuple, ndarray
    :param detector_direction_x: A 3D vector describing the direction of the detector_x (x,y,z)
    :type detector_direction_x: list, tuple, ndarray
    :param detector_direction_y: A 3D vector describing the direction of the detector_y (x,y,z)
    :type detector_direction_y: list, tuple, ndarray
    :param rotation_axis_pos: A 3D vector describing the position of the axis of rotation (x,y,z)
    :type rotation_axis_pos: list, tuple, ndarray
    :param rotation_axis_direction: A 3D vector describing the direction of the axis of rotation (x,y,z)
    :type rotation_axis_direction: list, tuple, ndarray       
     '''
    def __init__ (self,  ray_direction, detector_pos, detector_direction_x, detector_direction_y, rotation_axis_pos, rotation_axis_direction):
        """Constructor method
        """
        super(Parallel3D, self).__init__(dof=3, geometry = 'parallel')
                    
        #source
        self.ray.direction = ray_direction

        #detector
        self.detector.position = detector_pos
        self.detector.set_direction(detector_direction_x, detector_direction_y)

        #rotate axis
        self.rotation_axis.position = rotation_axis_pos
        self.rotation_axis.direction = rotation_axis_direction

    def update_reference_frame(self):
        r'''Transforms the system origin to the rotate axis with z direction aligned to the rotate axis direction
        '''          
        #shift detector
        self.detector.position = (self.detector.position - self.rotation_axis.position)
        self.rotation_axis.position = [0,0,0]

        #calculate rotation matrix to align rotation axis direction with z
        a = self.rotation_axis.direction

        if numpy.allclose(a,[0,0,1]):
            return
        elif numpy.allclose(a,[0,0,-1]):
            axis_rotation = numpy.eye(3)
            axis_rotation[1][1] = -1
            axis_rotation[2][2] = -1
        else:
            vx = numpy.array([[0, 0, -a[0]], [0, 0, -a[1]], [a[0], a[1], 0]])
            axis_rotation = numpy.eye(3) + vx + vx.dot(vx) *  1 / (1 + a[2])
        
        rotation_matrix = numpy.matrix(axis_rotation)

        #sanity check
        new_rotation_axis_direction = rotation_matrix.dot(self.rotation_axis.direction.reshape(3,1))

        if not numpy.allclose(new_rotation_axis_direction.flatten(), [0,0,1], atol=1e-7):
            raise ValueError("Failed to align reference frame")

        #apply transform
        self.rotation_axis.direction = [0,0,1]
        self.ray.direction = rotation_matrix.dot(self.ray.direction.reshape(3,1))
        self.detector.position = rotation_matrix.dot(self.detector.position.reshape(3,1))
        new_x = rotation_matrix.dot(self.detector.direction_x.reshape(3,1))
        new_y = rotation_matrix.dot(self.detector.direction_y.reshape(3,1))
        self.detector.set_direction(new_x, new_y)

    def align_reference_frame(self):
        r'''Transforms the system origin to the rotate axis with z direction aligned to the rotate axis direction, and aligns the ray direction along the positive Y direction
        '''          
        self.update_reference_frame()

        ray_vec = -self.ray.direction
        axis_rotation = numpy.eye(3)

        if numpy.allclose(ray_vec,[0,-1,0]):
            pass
        elif numpy.allclose(ray_vec,[0,1,0]):
            axis_rotation[0][0] = -1
            axis_rotation[1][1] = -1
        else:
            theta = math.atan2(ray_vec[0],ray_vec[1])
            axis_rotation[0][0] = axis_rotation[1][1] = math.cos(theta)
            axis_rotation[0][1] = -math.sin(theta)
            axis_rotation[1][0] = math.sin(theta)

        rotation_matrix = numpy.matrix(axis_rotation)
        
        self.ray.direction = rotation_matrix.dot(self.ray.direction.reshape(3,1))
        self.detector.position = rotation_matrix.dot(self.detector.position.reshape(3,1))

        new_direction_x = rotation_matrix.dot(self.detector.direction_x.reshape(3,1))
        new_direction_y = rotation_matrix.dot(self.detector.direction_y.reshape(3,1))

        self.detector.set_direction(new_direction_x, new_direction_y)

    def system_description(self):
        r'''Returns `simple` if the the geometry matches the default definitions with no offsets or rotations,
            \nReturns `offset` if the the geometry matches the default definitions with centre-of-rotation or detector offsets
            \nReturns `advanced` if the the geometry has rotated or tilted rotation axis or detector, can also have offsets
        '''              
        new = self.copy()
        new.align_reference_frame()

        try:
            det_unit = ComponentDescription.CreateUnitVector(new.detector.position)
        except ValueError: #pass test if detector is on origin
            det_unit = [0,1,0]

        if not numpy.allclose(new.ray.direction,[0,1,0]) or\
            not numpy.allclose(new.detector.direction_x,[1,0,0]) or\
            not numpy.allclose(new.detector.direction_y,[0,0,1]) or\
            not numpy.allclose(new.rotation_axis.direction,[0,0,1]):
            return SystemConfiguration.SYSTEM_ADVANCED
        elif not numpy.allclose(det_unit,[0,1,0]):
            return SystemConfiguration.SYSTEM_OFFSET
        else:
            return SystemConfiguration.SYSTEM_SIMPLE
        
        return False


    def __str__(self):
        def csv(val):
            return numpy.array2string(val, separator=', ')

        repres = "3D Parallel-beam tomography\n"
        repres += "System configuration:\n"
        repres += "\tRay direction: {0}\n".format(csv(self.ray.direction))
        repres += "\tRotation axis position: {0}\n".format(csv(self.rotation_axis.position))
        repres += "\tRotation axis direction: {0}\n".format(csv(self.rotation_axis.direction))
        repres += "\tDetector position: {0}\n".format(csv(self.detector.position))
        repres += "\tDetector direction x: {0}\n".format(csv(self.detector.direction_x))
        repres += "\tDetector direction y: {0}\n".format(csv(self.detector.direction_y))    
        return repres

    def __eq__(self, other):

        if not isinstance(other, self.__class__):
            return False
        
        if numpy.allclose(self.ray.direction, other.ray.direction) \
        and numpy.allclose(self.detector.position, other.detector.position)\
        and numpy.allclose(self.detector.direction_x, other.detector.direction_x)\
        and numpy.allclose(self.detector.direction_y, other.detector.direction_y)\
        and numpy.allclose(self.rotation_axis.position, other.rotation_axis.position)\
        and numpy.allclose(self.rotation_axis.direction, other.rotation_axis.direction):
            
            return True
        
        return False

    def calculate_magnification(self):
        return [None, None, 1.0]

    def get_centre_slice(self):
        """Returns the 2D system configuration corersponding to the centre slice
        """  
        dp1 = self.rotation_axis.direction.dot(self.ray.direction)
        dp2 = self.rotation_axis.direction.dot(self.detector.direction_x)

        if numpy.isclose(dp1, 0) and numpy.isclose(dp2, 0):
            temp = self.copy()

            #convert to rotation axis reference frame
            temp.update_reference_frame()

            ray_direction = temp.ray.direction[0:2]
            detector_position = temp.detector.position[0:2]
            detector_direction_x = temp.detector.direction_x[0:2]
            rotation_axis_position = temp.rotation_axis.position[0:2]

            return Parallel2D(ray_direction, detector_position, detector_direction_x, rotation_axis_position)

        else:
            raise ValueError('Cannot convert geometry to 2D. Requires axis of rotation to be perpenidular to ray direction and the detector direction x.')


class Cone2D(SystemConfiguration):
    r'''This class creates the SystemConfiguration of a cone beam 2D tomographic system
                       
    :param source_pos: A 2D vector describing the position of the source (x,y)
    :type source_pos: list, tuple, ndarray
    :param detector_pos: A 2D vector describing the position of the centre of the detector (x,y)
    :type detector_pos: list, tuple, ndarray
    :param detector_direction_x: A 2D vector describing the direction of the detector_x (x,y)
    :type detector_direction_x: list, tuple, ndarray
    :param rotation_axis_pos: A 2D vector describing the position of the axis of rotation (x,y)
    :type rotation_axis_pos: list, tuple, ndarray    
     '''

    def __init__ (self, source_pos, detector_pos, detector_direction_x, rotation_axis_pos):
        """Constructor method
        """
        super(Cone2D, self).__init__(dof=2, geometry = 'cone')

        #source
        self.source.position = source_pos

        #detector
        self.detector.position = detector_pos
        self.detector.direction_x = detector_direction_x

        #rotate axis
        self.rotation_axis.position = rotation_axis_pos

    def update_reference_frame(self):
        r'''Transforms the system origin to the rotate axis
        '''                  
        self.source.position -= self.rotation_axis.position
        self.detector.position -= self.rotation_axis.position
        self.rotation_axis.position = [0,0]

    def align_reference_frame(self):
        r'''Transforms the system origin to the rotate axis and aligns the source position along the negative Y direction
        '''          
        self.update_reference_frame()

        src_dir = ComponentDescription.CreateUnitVector(self.source.position)

        axis_rotation = numpy.eye(2)
        if numpy.allclose(src_dir,[0,-1]):
            pass
        elif numpy.allclose(src_dir,[0,1]):
            axis_rotation[0][0] = -1
            axis_rotation[1][1] = -1
        else:
            theta = math.atan2(src_dir[0],src_dir[1])
            axis_rotation[0][0] = axis_rotation[1][1] = math.cos(theta)
            axis_rotation[0][1] = -math.sin(theta)
            axis_rotation[1][0] = math.sin(theta)

        rotation_matrix = numpy.matrix(axis_rotation)
        
        self.source.position = rotation_matrix.dot(self.source.position.reshape(2,1))
        self.detector.position = rotation_matrix.dot(self.detector.position.reshape(2,1))
        self.detector.direction_x = rotation_matrix.dot(self.detector.direction_x.reshape(2,1))


    def system_description(self):
        r'''Returns `simple` if the the geometry matches the default definitions with no offsets or rotations,
            \nReturns `offset` if the the geometry matches the default definitions with centre-of-rotation or detector offsets
            \nReturns `advanced` if the the geometry has rotated or tilted rotation axis or detector, can also have offsets
        '''           
        new = self.copy()
        new.align_reference_frame()
        dot_prod = (new.detector.position - new.source.position).dot(new.detector.direction_x)

        if abs(dot_prod)>1e-6:
            return SystemConfiguration.SYSTEM_ADVANCED
        elif abs(new.source.position[0])>1e-6 or\
            abs(new.detector.position[0])>1e-6:
            return SystemConfiguration.SYSTEM_OFFSET 
        else:
            return SystemConfiguration.SYSTEM_SIMPLE

    def __str__(self):
        def csv(val):
            return numpy.array2string(val, separator=', ')

        repres = "2D Cone-beam tomography\n"
        repres += "System configuration:\n"
        repres += "\tSource position: {0}\n".format(csv(self.source.position))
        repres += "\tRotation axis position: {0}\n".format(csv(self.rotation_axis.position))
        repres += "\tDetector position: {0}\n".format(csv(self.detector.position))
        repres += "\tDetector direction x: {0}\n".format(csv(self.detector.direction_x)) 
        return repres    

    def __eq__(self, other):

        if not isinstance(other, self.__class__):
            return False
        
        if numpy.allclose(self.source.position, other.source.position) \
        and numpy.allclose(self.detector.position, other.detector.position)\
        and numpy.allclose(self.detector.direction_x, other.detector.direction_x)\
        and numpy.allclose(self.rotation_axis.position, other.rotation_axis.position):
            return True
        
        return False

    def get_centre_slice(self):
        return self

    def calculate_magnification(self):
        #64bit for maths
        rotation_axis_position = self.rotation_axis.position.astype(numpy.float64)
        source_position = self.source.position.astype(numpy.float64)
        detector_position = self.detector.position.astype(numpy.float64)
        direction_x = self.detector.direction_x.astype(numpy.float64)

        ab = (rotation_axis_position - source_position)
        dist_source_center = float(numpy.sqrt(ab.dot(ab)))

        ab_unit = ab / numpy.sqrt(ab.dot(ab))

        n = ComponentDescription.CreateVector([direction_x[1], -direction_x[0]]).astype(numpy.float64)

        #perpendicular distance between source and detector centre
        sd = float((detector_position - source_position).dot(n))
        ratio = float(ab_unit.dot(n))

        source_to_detector = sd / ratio
        dist_center_detector = source_to_detector - dist_source_center
        magnification = (dist_center_detector + dist_source_center) / dist_source_center

        return [dist_source_center, dist_center_detector, magnification]

class Cone3D(SystemConfiguration):
    r'''This class creates the SystemConfiguration of a cone beam 3D tomographic system
                       
    :param source_pos: A 3D vector describing the position of the source (x,y,z)
    :type source_pos: list, tuple, ndarray
    :param detector_pos: A 3D vector describing the position of the centre of the detector (x,y,z)
    :type detector_pos: list, tuple, ndarray
    :param detector_direction_x: A 3D vector describing the direction of the detector_x (x,y,z)
    :type detector_direction_x: list, tuple, ndarray
    :param detector_direction_y: A 3D vector describing the direction of the detector_y (x,y,z)
    :type detector_direction_y: list, tuple, ndarray   
    :param rotation_axis_pos: A 3D vector describing the position of the axis of rotation (x,y,z)
    :type rotation_axis_pos: list, tuple, ndarray
    :param rotation_axis_direction: A 3D vector describing the direction of the axis of rotation (x,y,z)
    :type rotation_axis_direction: list, tuple, ndarray   
    '''

    def __init__ (self, source_pos, detector_pos, detector_direction_x, detector_direction_y, rotation_axis_pos, rotation_axis_direction):
        """Constructor method
        """
        super(Cone3D, self).__init__(dof=3, geometry = 'cone')

        #source
        self.source.position = source_pos

        #detector
        self.detector.position = detector_pos
        self.detector.set_direction(detector_direction_x, detector_direction_y)

        #rotate axis
        self.rotation_axis.position = rotation_axis_pos
        self.rotation_axis.direction = rotation_axis_direction

    def update_reference_frame(self):
        r'''Transforms the system origin to the rotate axis with z direction aligned to the rotate axis direction
        '''                  
        #shift 
        self.detector.position = (self.detector.position - self.rotation_axis.position)
        self.source.position = (self.source.position - self.rotation_axis.position)
        self.rotation_axis.position = [0,0,0]

        #calculate rotation matrix to align rotation axis direction with z
        a = self.rotation_axis.direction
        if numpy.allclose(a,[0,0,1]):
            return
        elif numpy.allclose(a,[0,0,-1]):
            axis_rotation = numpy.eye(3)
            axis_rotation[1][1] = -1
            axis_rotation[2][2] = -1
        else:
            vx = numpy.array([[0, 0, -a[0]], [0, 0, -a[1]], [a[0], a[1], 0]])
            axis_rotation = numpy.eye(3) + vx + vx.dot(vx) *  1 / (1 + a[2])
    
        rotation_matrix = numpy.matrix(axis_rotation)

        #sanity check
        new_rotation_axis_direction = rotation_matrix.dot(self.rotation_axis.direction.reshape(3,1))

        if not numpy.allclose(new_rotation_axis_direction.flatten(), [0,0,1], atol=1e-7):
            raise ValueError("Failed to align reference frame")
    
        #apply transform
        self.rotation_axis.direction = [0,0,1]
        self.source.position = rotation_matrix.dot(self.source.position.reshape(3,1))
        self.detector.position = rotation_matrix.dot(self.detector.position.reshape(3,1))
        new_x = rotation_matrix.dot(self.detector.direction_x.reshape(3,1)) 
        new_y = rotation_matrix.dot(self.detector.direction_y.reshape(3,1))
        self.detector.set_direction(new_x, new_y)

    def align_reference_frame(self):
        r'''Transforms the system origin to the rotate axis with z direction aligned to the rotate axis direction, and aligns the source direction along the negative Y direction
        '''          
        self.update_reference_frame()

        src_dir = ComponentDescription.CreateUnitVector(self.source.position)

        axis_rotation = numpy.eye(3)
        if numpy.allclose(src_dir,[0,-1,0]):
            pass
        elif numpy.allclose(src_dir,[0,1,0]):
            axis_rotation[0][0] = -1
            axis_rotation[1][1] = -1
        else:
            theta = math.atan2(src_dir[0],-src_dir[1])
            axis_rotation[0][0] = axis_rotation[1][1] = math.cos(theta)
            axis_rotation[0][1] = math.sin(theta)
            axis_rotation[1][0] = -math.sin(theta)

        rotation_matrix = numpy.matrix(axis_rotation)
        
        self.source.position = rotation_matrix.dot(self.source.position.reshape(3,1))
        self.detector.position = rotation_matrix.dot(self.detector.position.reshape(3,1))

        new_direction_x = rotation_matrix.dot(self.detector.direction_x.reshape(3,1))
        new_direction_y = rotation_matrix.dot(self.detector.direction_y.reshape(3,1))

        self.detector.set_direction(new_direction_x, new_direction_y)

    def system_description(self):
        r'''Returns `simple` if the the geometry matches the default definitions with no offsets or rotations,
            \nReturns `offset` if the the geometry matches the default definitions with centre-of-rotation or detector offsets
            \nReturns `advanced` if the the geometry has rotated or tilted rotation axis or detector, can also have offsets
        '''       
        new = self.copy()
        new.align_reference_frame()

        dot_prod_a = (new.detector.position - new.source.position).dot(new.detector.direction_x)
        dot_prod_b = (new.detector.position - new.source.position).dot(new.detector.direction_y)
        dot_prod_c = (new.detector.direction_x).dot(new.rotation_axis.direction)
        dot_prod_d = (new.detector.position - new.source.position).dot(new.rotation_axis.direction)

        if abs(dot_prod_a)>1e-6 or\
            abs(dot_prod_b)>1e-6 or\
            abs(dot_prod_c)>1e-6 or\
            abs(dot_prod_d)>1e-6: 
            return SystemConfiguration.SYSTEM_ADVANCED

        elif abs(new.source.position[0])>1e-6 or\
            abs(new.source.position[2])>1e-6 or\
            abs(new.detector.position[0])>1e-6 or\
            abs(new.detector.position[2])>1e-6:
            return SystemConfiguration.SYSTEM_OFFSET
        else:
            return SystemConfiguration.SYSTEM_SIMPLE


    def get_centre_slice(self):
        """Returns the 2D system configuration corersponding to the centre slice
        """ 
        #requires the rotate axis to be perpendicular to the normal of the detector, and perpendicular to detector_direction_x
        vec1= numpy.cross(self.detector.direction_x, self.detector.direction_y)  
        dp1 = self.rotation_axis.direction.dot(vec1)
        dp2 = self.rotation_axis.direction.dot(self.detector.direction_x)
        
        if numpy.isclose(dp1, 0) and numpy.isclose(dp2, 0):
            temp = self.copy()
            temp.update_reference_frame()
            source_position = temp.source.position[0:2]
            detector_position = temp.detector.position[0:2]
            detector_direction_x = temp.detector.direction_x[0:2]
            rotation_axis_position = temp.rotation_axis.position[0:2]

            return Cone2D(source_position, detector_position, detector_direction_x, rotation_axis_position)
        else:
            raise ValueError('Cannot convert geometry to 2D. Requires axis of rotation to be perpendicular to the detector.')
        
    def __str__(self):
        def csv(val):
            return numpy.array2string(val, separator=', ')

        repres = "3D Cone-beam tomography\n"
        repres += "System configuration:\n"
        repres += "\tSource position: {0}\n".format(csv(self.source.position))
        repres += "\tRotation axis position: {0}\n".format(csv(self.rotation_axis.position))
        repres += "\tRotation axis direction: {0}\n".format(csv(self.rotation_axis.direction))
        repres += "\tDetector position: {0}\n".format(csv(self.detector.position))
        repres += "\tDetector direction x: {0}\n".format(csv(self.detector.direction_x))
        repres += "\tDetector direction y: {0}\n".format(csv(self.detector.direction_y))
        return repres   

    def __eq__(self, other):

        if not isinstance(other, self.__class__):
            return False
        
        if numpy.allclose(self.source.position, other.source.position) \
        and numpy.allclose(self.detector.position, other.detector.position)\
        and numpy.allclose(self.detector.direction_x, other.detector.direction_x)\
        and numpy.allclose(self.detector.direction_y, other.detector.direction_y)\
        and numpy.allclose(self.rotation_axis.position, other.rotation_axis.position)\
        and numpy.allclose(self.rotation_axis.direction, other.rotation_axis.direction):
            
            return True
        
        return False

    def calculate_magnification(self):
    
        #64bit for maths
        rotation_axis_position = self.rotation_axis.position.astype(numpy.float64)
        source_position = self.source.position.astype(numpy.float64)
        detector_position = self.detector.position.astype(numpy.float64)
        direction_x = self.detector.direction_x.astype(numpy.float64)

        ab = (rotation_axis_position - source_position)
        dist_source_center = float(numpy.sqrt(ab.dot(ab)))

        ab_unit = ab / numpy.sqrt(ab.dot(ab))

        #dey y and det x are perpendicular unit vectors so n is a unit vector
        #unit vector orthogonal to the detector
        direction_y = self.detector.direction_y.astype(numpy.float64)
        n = numpy.cross(direction_x,direction_y)

        #perpendicular distance between source and detector centre
        sd = float((detector_position - source_position).dot(n))
        ratio = float(ab_unit.dot(n))

        source_to_detector = sd / ratio
        dist_center_detector = source_to_detector - dist_source_center
        magnification = (dist_center_detector + dist_source_center) / dist_source_center

        return [dist_source_center, dist_center_detector, magnification]

class Panel(object):
    r'''This is a class describing the panel of the system. 
                 
    :param num_pixels: num_pixels_h or (num_pixels_h, num_pixels_v) containing the number of pixels of the panel
    :type num_pixels: int, list, tuple
    :param pixel_size: pixel_size_h or (pixel_size_h, pixel_size_v) containing the size of the pixels of the panel
    :type pixel_size: int, lust, tuple
    :param origin: the position of pixel 0 (the data origin) of the panel `top-left`, `top-right`, `bottom-left`, `bottom-right`
    :type origin: string 
     '''

    @property
    def num_pixels(self):
        return self.__num_pixels

    @num_pixels.setter
    def num_pixels(self, val):

        if isinstance(val,int):
            num_pixels_temp = [val, 1]
        else:
            try:
                length_val = len(val)
            except:
                raise TypeError('num_pixels expected int x or [int x, int y]. Got {}'.format(type(val)))


            if length_val == 2:
                try:
                    val0 = int(val[0])
                    val1 = int(val[1])
                except:
                    raise TypeError('num_pixels expected int x or [int x, int y]. Got {0},{1}'.format(type(val[0]), type(val[1])))

                num_pixels_temp = [val0, val1]
            else:
                raise ValueError('num_pixels expected int x or [int x, int y]. Got {}'.format(val))
   
        if num_pixels_temp[1] > 1 and self._dimension == 2:
            raise ValueError('2D acquisitions expects a 1D panel. Expected num_pixels[1] = 1. Got {}'.format(num_pixels_temp[1]))
        if num_pixels_temp[0] < 1 or num_pixels_temp[1] < 1:
            raise ValueError('num_pixels (x,y) must be >= (1,1). Got {}'.format(num_pixels_temp))
        else:
            self.__num_pixels = numpy.array(num_pixels_temp, dtype=numpy.int16)

    @property
    def pixel_size(self):
        return self.__pixel_size

    @pixel_size.setter
    def pixel_size(self, val):

        if val is None:
            pixel_size_temp = [1.0,1.0] 
        else:
            try:
                length_val = len(val)
            except:
                try:
                    temp = float(val)
                    pixel_size_temp = [temp, temp]

                except:
                    raise TypeError('pixel_size expected float xy or [float x, float y]. Got {}'.format(val))    
            else:
                if length_val == 2:
                    try:
                        temp0 = float(val[0]) 
                        temp1 = float(val[1]) 
                        pixel_size_temp = [temp0, temp1]
                    except:
                        raise ValueError('pixel_size expected float xy or [float x, float y]. Got {}'.format(val))
                else:
                    raise ValueError('pixel_size expected float xy or [float x, float y]. Got {}'.format(val))
    
            if pixel_size_temp[0] <= 0 or pixel_size_temp[1] <= 0:
                raise ValueError('pixel_size (x,y) at must be > (0.,0.). Got {}'.format(pixel_size_temp)) 

        self.__pixel_size = numpy.array(pixel_size_temp)

    @property
    def origin(self):
        return self.__origin

    @origin.setter
    def origin(self, val):
        allowed = ['top-left', 'top-right','bottom-left','bottom-right']
        if val in allowed:
            self.__origin=val
        else:
            raise ValueError('origin expected one of {0}. Got {1}'.format(allowed, val))

    def __str__(self):
        repres = "Panel configuration:\n"             
        repres += "\tNumber of pixels: {0}\n".format(self.num_pixels)
        repres += "\tPixel size: {0}\n".format(self.pixel_size)
        repres += "\tPixel origin: {0}\n".format(self.origin)
        return repres   

    def __eq__(self, other):

        if not isinstance(other, self.__class__):
            return False
        
        if numpy.array_equal(self.num_pixels, other.num_pixels) \
            and numpy.allclose(self.pixel_size, other.pixel_size) \
            and self.origin == other.origin:   
            return True
        
        return False

    def __init__ (self, num_pixels, pixel_size, origin, dimension):  
        """Constructor method
        """
        self._dimension = dimension
        self.num_pixels = num_pixels
        self.pixel_size = pixel_size
        self.origin = origin

class Channels(object):
    r'''This is a class describing the channels of the data. 
    This will be created on initialisation of AcquisitionGeometry.
                       
    :param num_channels: The number of channels of data
    :type num_channels: int
    :param channel_labels: A list of channel labels
    :type channel_labels: list, optional
     '''

    @property
    def num_channels(self):
        return self.__num_channels

    @num_channels.setter
    def num_channels(self, val):      
        try:
            val = int(val)
        except TypeError:
            raise ValueError('num_channels expected a positive integer. Got {}'.format(type(val)))

        if val > 0:
            self.__num_channels = val
        else:
            raise ValueError('num_channels expected a positive integer. Got {}'.format(val))

    @property
    def channel_labels(self):
        return self.__channel_labels

    @channel_labels.setter
    def channel_labels(self, val):      
        if val is None or len(val) == self.__num_channels:
            self.__channel_labels = val  
        else:
            raise ValueError('labels expected to have length {0}. Got {1}'.format(self.__num_channels, len(val)))

    def __str__(self):
        repres = "Channel configuration:\n"             
        repres += "\tNumber of channels: {0}\n".format(self.num_channels)
        
        num_print=min(10,self.num_channels)                     
        if  hasattr(self, 'channel_labels'):
            repres += "\tChannel labels 0-{0}: {1}\n".format(num_print, self.channel_labels[0:num_print])
        
        return repres

    def __eq__(self, other):

        if not isinstance(other, self.__class__):
            return False
        
        if self.num_channels != other.num_channels:
            return False

        if hasattr(self,'channel_labels'):
            if self.channel_labels != other.channel_labels:
                return False
         
        return True

    def __init__ (self, num_channels, channel_labels):  
        """Constructor method
        """
        self.num_channels = num_channels
        if channel_labels is not None:
            self.channel_labels = channel_labels

class Angles(object):
    r'''This is a class describing the angles of the data. 

    :param angles: The angular positions of the acquisition data
    :type angles: list, ndarray
    :param initial_angle: The angular offset of the object from the reference frame
    :type initial_angle: float, optional
    :param angle_unit: The units of the stored angles 'degree' or 'radian'
    :type angle_unit: string
     '''

    @property
    def angle_data(self):
        return self.__angle_data

    @angle_data.setter
    def angle_data(self, val):
        if val is None:
            raise ValueError('angle_data expected to be a list of floats') 
        else:
            try:
                self.num_positions = len(val)

            except TypeError:
                self.num_positions = 1
                val = [val]

            finally:
                try:
                    self.__angle_data = numpy.asarray(val, dtype=numpy.float32)
                except:
                    raise ValueError('angle_data expected to be a list of floats') 

    @property
    def initial_angle(self):
        return self.__initial_angle

    @initial_angle.setter
    def initial_angle(self, val):
        try:
            val = float(val)
        except:
            raise TypeError('initial_angle expected a float. Got {0}'.format(type(val)))

        self.__initial_angle = val

    @property
    def angle_unit(self):
        return self.__angle_unit

    @angle_unit.setter
    def angle_unit(self,val):
        if val != AcquisitionGeometry.DEGREE and val != AcquisitionGeometry.RADIAN:
            raise ValueError('angle_unit = {} not recognised please specify \'degree\' or \'radian\''.format(val))
        else:
            self.__angle_unit = val

    def __str__(self):
        repres = "Acquisition description:\n"
        repres += "\tNumber of positions: {0}\n".format(self.num_positions)
        num_print=min(20,self.num_positions)    
        repres += "\tAngles 0-{0} in {1}s:\n{2}\n".format(num_print, self.angle_unit, numpy.array2string(self.angle_data[0:num_print], separator=', '))
        return repres   

    def __eq__(self, other):

        if not isinstance(other, self.__class__):
            return False
        
        if self.angle_unit != other.angle_unit:
            return False

        if self.initial_angle != other.initial_angle:
            return False

        if not numpy.allclose(self.angle_data, other.angle_data):
            return False
         
        return True

    def __init__ (self, angles, initial_angle, angle_unit):  
        """Constructor method
        """
        self.angle_data = angles
        self.initial_angle = initial_angle
        self.angle_unit = angle_unit

class Configuration(object):
    r'''This is a class holds the description of the system components. 
     '''

    def __init__(self):
        self.system = None
        self.angles = None
        self.panel = None
        self.channels = Channels(1, None)

    @property
    def configured(self):
        if self.system is None:
            print("Please configure AcquisitionGeometry using one of the following methods:\
                    \n\tAcquisitionGeometry.create_Parallel2D()\
                    \n\tAcquisitionGeometry.create_Cone3D()\
                    \n\tAcquisitionGeometry.create_Parallel2D()\
                    \n\tAcquisitionGeometry.create_Cone3D()")
            return False

        configured = True
        if self.angles is None:
            print("Please configure angular data using the set_angles() method")
            configured = False
        if self.panel is None:
            print("Please configure the panel using the set_panel() method")
            configured = False
        return configured

    def __str__(self):
        repres = ""
        if self.configured:
            repres += str(self.system)
            repres += str(self.panel)
            repres += str(self.channels)
            repres += str(self.angles)
        
        return repres

    def __eq__(self, other):
        
        if not isinstance(other, self.__class__):
            return False

        if self.system == other.system\
        and self.panel == other.panel\
        and self.channels == other.channels\
        and self.angles == other.angles:
            return True

        return False


class AcquisitionGeometry(object):
    r'''This class holds the AcquisitionGeometry of the system.
    
    Please initialise using factory:
    AcquisitionGeometry.create_Parallel2D
    AcquisitionGeometry.create_Cone3D
    AcquisitionGeometry.create_Parallel2D
    AcquisitionGeometry.create_Cone3D


    These initialisation parameters will be deprecated in a future release.    
    :param geom_type: A description of the system type 'cone' or 'parallel'
    :type geom_type: string
    :param pixel_num_h: Number of pixels in the horizontal direction
    :type pixel_num_h: int, optional
    :param pixel_num_v: Number of pixels in the vertical direction
    :type pixel_num_v: int, optional    
    :param pixel_size_h: Size of pixels in the horizontal direction
    :type pixel_size_h: float, optional    
    :param pixel_size_v: Size of pixels in the vertical direction
    :type pixel_size_v: float, optional       
    :param chanels: Number of channels
    :type chanels: int, optional       
    :param dist_source_center: Distance from the source to the origin
    :type dist_source_center: float, optional
    :param dist_center_detector: Distance from the origin to the centre of the detector
    :type dist_center_detector: float, optional

    '''

    RANDOM = 'random'
    RANDOM_INT = 'random_int'
    ANGLE_UNIT = 'angle_unit'
    DEGREE = 'degree'
    RADIAN = 'radian'
    CHANNEL = 'channel'
    ANGLE = 'angle'
    VERTICAL = 'vertical'
    HORIZONTAL = 'horizontal'
    PARALLEL = 'parallel'
    CONE = 'cone'
    DIM2 = '2D'
    DIM3 = '3D'

    #for backwards compatibility
    @property
    def geom_type(self):
        return self.config.system.geometry

    @property
    def num_projections(self):
        return len(self.angles)       

    @property
    def pixel_num_h(self):
        return self.config.panel.num_pixels[0]

    @pixel_num_h.setter
    def pixel_num_h(self, val):
        self.config.panel.num_pixels[0] = val

    @property
    def pixel_num_v(self):
        return self.config.panel.num_pixels[1]

    @pixel_num_v.setter
    def pixel_num_v(self, val):
        self.config.panel.num_pixels[1] = val

    @property
    def pixel_size_h(self):
        return self.config.panel.pixel_size[0]

    @pixel_size_h.setter
    def pixel_size_h(self, val):
        self.config.panel.pixel_size[0] = val

    @property
    def pixel_size_v(self):
        return self.config.panel.pixel_size[1]

    @pixel_size_v.setter
    def pixel_size_v(self, val):
        self.config.panel.pixel_size[1] = val

    @property
    def channels(self):
        return self.config.channels.num_channels

    @property
    def angles(self):
        return self.config.angles.angle_data

    @property
    def dist_source_center(self):
        out = self.config.system.calculate_magnification()
        return out[0]

    @property
    def dist_center_detector(self):
        out = self.config.system.calculate_magnification()
        return out[1]

    @property
    def magnification(self):
        out = self.config.system.calculate_magnification()
        return out[2]

    @property
    def dimension(self):
        return self.config.system.dimension

    @property
    def shape(self):

        shape_dict = {AcquisitionGeometry.CHANNEL: self.config.channels.num_channels,
                     AcquisitionGeometry.ANGLE: self.config.angles.num_positions,
                     AcquisitionGeometry.VERTICAL: self.config.panel.num_pixels[1],        
                     AcquisitionGeometry.HORIZONTAL: self.config.panel.num_pixels[0]}

        shape = []
        for label in self.dimension_labels:
            shape.append(shape_dict[label])

        return tuple(shape)

    @shape.setter
    def shape(self, val):
        print("Deprecated - shape will be set automatically")

    @property
    def dimension_labels(self):
        labels_default = DataOrder.CIL_AG_LABELS

        shape_default = [self.config.channels.num_channels,
                            self.config.angles.num_positions,
                            self.config.panel.num_pixels[1],
                            self.config.panel.num_pixels[0]
                            ]

        try:
            labels = list(self.__dimension_labels)
        except AttributeError:
            labels = labels_default.copy()

        #remove from list labels where len == 1
        #
        for i, x in enumerate(shape_default):
            if x == 1:
                try:
                    labels.remove(labels_default[i])
                except ValueError:
                    pass #if not in custom list carry on

        return tuple(labels)
      
    @dimension_labels.setter
    def dimension_labels(self, val):

        labels_default = DataOrder.CIL_AG_LABELS

        #check input and store. This value is not used directly
        if val is not None:
            for x in val:
                if x not in labels_default:
                    raise ValueError('Requested axis are not possible. Accepted label names {},\ngot {}'.format(labels_default,val))
                    
            self.__dimension_labels = tuple(val)


    @property
    def system_description(self):
        return self.config.system.system_description()

    @property
    def dtype(self):
        return self.__dtype

    @dtype.setter
    def dtype(self, val):
        self.__dtype = val       


    def __init__(self,
                geom_type, 
                dimension=None,
                angles=None, 
                pixel_num_h=1, 	
                pixel_size_h=1,
                pixel_num_v=1,
                pixel_size_v=1,
                dist_source_center=None,
                dist_center_detector=None,
                channels=1,
                ** kwargs):

        """Constructor method
        """

        # default dtype for the acquisition geometry
        self.dtype = kwargs.get('dtype', numpy.float32)

        #backward compatibility
        new_setup = kwargs.get('new_setup', False)

        #set up old geometry        
        if new_setup is False:
            self.config = Configuration()
            
            if angles is None:
                raise ValueError("AcquisitionGeometry not configured. Parameter 'angles' is required")

            if geom_type == AcquisitionGeometry.CONE:
                if dist_source_center is None:
                    raise ValueError("AcquisitionGeometry not configured. Parameter 'dist_source_center' is required")
                if dist_center_detector is None:
                    raise ValueError("AcquisitionGeometry not configured. Parameter 'dist_center_detector' is required")

            if pixel_num_v > 1:
                dimension = 3
                num_pixels = [pixel_num_h, pixel_num_v]
                pixel_size = [pixel_size_h, pixel_size_v]
                if geom_type == AcquisitionGeometry.CONE:
                    self.config.system = Cone3D(source_pos=[0,-dist_source_center,0], detector_pos=[0,dist_center_detector,0], detector_direction_x=[1,0,0], detector_direction_y=[0,0,1], rotation_axis_pos=[0,0,0], rotation_axis_direction=[0,0,1])
                else:
                    self.config.system = Parallel3D(ray_direction=[0,1,0], detector_pos=[0,0,0], detector_direction_x=[1,0,0], detector_direction_y=[0,0,1], rotation_axis_pos=[0,0,0], rotation_axis_direction=[0,0,1])
            else:
                dimension = 2
                num_pixels = [pixel_num_h, 1]
                pixel_size = [pixel_size_h, pixel_size_h]                
                if geom_type == AcquisitionGeometry.CONE:
                    self.config.system = Cone2D(source_pos=[0,-dist_source_center], detector_pos=[0,dist_center_detector], detector_direction_x=[1,0], rotation_axis_pos=[0,0])
                else:
                    self.config.system = Parallel2D(ray_direction=[0,1], detector_pos=[0,0], detector_direction_x=[1,0], rotation_axis_pos=[0,0])


            self.config.panel = Panel(num_pixels, pixel_size, 'bottom-left', dimension)  
            self.config.channels = Channels(channels, channel_labels=None)  
            self.config.angles = Angles(angles, 0, kwargs.get(AcquisitionGeometry.ANGLE_UNIT, AcquisitionGeometry.DEGREE))

            self.dimension_labels = kwargs.get('dimension_labels', None)
            if self.config.configured:
                print("AcquisitionGeometry configured using deprecated method")
            else:
                raise ValueError("AcquisitionGeometry not configured")

    def set_angles(self, angles, initial_angle=0, angle_unit='degree'):
        r'''This method configures the angular information of an AcquisitionGeometry object. 

        :param angles: The angular positions of the acquisition data
        :type angles: list, ndarray
        :param initial_angle: The angular offset of the object from the reference frame
        :type initial_angle: float, optional
        :param angle_unit: The units of the stored angles 'degree' or 'radian'
        :type angle_unit: string
        :return: returns a configured AcquisitionGeometry object
        :rtype: AcquisitionGeometry        
        '''
        self.config.angles = Angles(angles, initial_angle, angle_unit)
        return self

    def set_panel(self, num_pixels, pixel_size=(1,1), origin='bottom-left'):

        r'''This method configures the panel information of an AcquisitionGeometry object. 
                    
        :param num_pixels: num_pixels_h or (num_pixels_h, num_pixels_v) containing the number of pixels of the panel
        :type num_pixels: int, list, tuple
        :param pixel_size: pixel_size_h or (pixel_size_h, pixel_size_v) containing the size of the pixels of the panel
        :type pixel_size: int, list, tuple, optional
        :param origin: the position of pixel 0 (the data origin) of the panel 'top-left', 'top-right', 'bottom-left', 'bottom-right'
        :type origin: string, default 'bottom-left'
        :return: returns a configured AcquisitionGeometry object
        :rtype: AcquisitionGeometry       
        '''        
        self.config.panel = Panel(num_pixels, pixel_size, origin, self.config.system._dimension)
        return self

    def set_channels(self, num_channels=1, channel_labels=None):
        r'''This method configures the channel information of an AcquisitionGeometry object. 
                        
        :param num_channels: The number of channels of data
        :type num_channels: int, optional
        :param channel_labels: A list of channel labels
        :type channel_labels: list, optional
        :return: returns a configured AcquisitionGeometry object
        :rtype: AcquisitionGeometry        
        '''        
        self.config.channels = Channels(num_channels, channel_labels)
        return self

    def set_labels(self, labels=None):
        r'''This method configures the dimension labels of an AcquisitionGeometry object. 
                        
        :param labels:  The order of the dimensions describing the data.\
                        Expects a list containing at least one of the unique labels: 'channel' 'angle' 'vertical' 'horizontal'
                        default = ['channel','angle','vertical','horizontal']
        :type labels: list, optional
        :return: returns a configured AcquisitionGeometry object
        :rtype: AcquisitionGeometry        
        '''           
        self.dimension_labels = labels
        return self
 
    @staticmethod
    def create_Parallel2D(ray_direction=[0, 1], detector_position=[0, 0], detector_direction_x=[1, 0], rotation_axis_position=[0, 0]):
        r'''This creates the AcquisitionGeometry for a parallel beam 2D tomographic system

        :param ray_direction: A 2D vector describing the x-ray direction (x,y)
        :type ray_direction: list, tuple, ndarray, optional
        :param detector_position: A 2D vector describing the position of the centre of the detector (x,y)
        :type detector_position: list, tuple, ndarray, optional
        :param detector_direction_x: A 2D vector describing the direction of the detector_x (x,y)
        :type detector_direction_x: list, tuple, ndarray
        :param rotation_axis_position: A 2D vector describing the position of the axis of rotation (x,y)
        :type rotation_axis_position: list, tuple, ndarray, optional
        :return: returns a configured AcquisitionGeometry object
        :rtype: AcquisitionGeometry
        '''
        AG = AcquisitionGeometry('', new_setup=True)
        AG.config = Configuration()
        AG.config.system = Parallel2D(ray_direction, detector_position, detector_direction_x, rotation_axis_position)
        return AG    

    @staticmethod
    def create_Cone2D(source_position, detector_position, detector_direction_x=[1,0], rotation_axis_position=[0,0]):
        r'''This creates the AcquisitionGeometry for a cone beam 2D tomographic system          

        :param source_position: A 2D vector describing the position of the source (x,y)
        :type source_position: list, tuple, ndarray
        :param detector_position: A 2D vector describing the position of the centre of the detector (x,y)
        :type detector_position: list, tuple, ndarray
        :param detector_direction_x: A 2D vector describing the direction of the detector_x (x,y)
        :type detector_direction_x: list, tuple, ndarray
        :param rotation_axis_position: A 2D vector describing the position of the axis of rotation (x,y)
        :type rotation_axis_position: list, tuple, ndarray, optional
        :return: returns a configured AcquisitionGeometry object
        :rtype: AcquisitionGeometry        
     '''    
        AG = AcquisitionGeometry('', new_setup=True)
        AG.config = Configuration()
        AG.config.system = Cone2D(source_position, detector_position, detector_direction_x, rotation_axis_position)
        return AG   

    @staticmethod
    def create_Parallel3D(ray_direction=[0,1,0], detector_position=[0,0,0], detector_direction_x=[1,0,0], detector_direction_y=[0,0,1], rotation_axis_position=[0,0,0], rotation_axis_direction=[0,0,1]):
        r'''This creates the AcquisitionGeometry for a parallel beam 3D tomographic system
                       
        :param ray_direction: A 3D vector describing the x-ray direction (x,y,z)
        :type ray_direction: list, tuple, ndarray, optional
        :param detector_position: A 3D vector describing the position of the centre of the detector (x,y,z)
        :type detector_position: list, tuple, ndarray, optional
        :param detector_direction_x: A 3D vector describing the direction of the detector_x (x,y,z)
        :type detector_direction_x: list, tuple, ndarray
        :param detector_direction_y: A 3D vector describing the direction of the detector_y (x,y,z)
        :type detector_direction_y: list, tuple, ndarray
        :param rotation_axis_position: A 3D vector describing the position of the axis of rotation (x,y,z)
        :type rotation_axis_position: list, tuple, ndarray, optional
        :param rotation_axis_direction: A 3D vector describing the direction of the axis of rotation (x,y,z)
        :type rotation_axis_direction: list, tuple, ndarray, optional     
        :return: returns a configured AcquisitionGeometry object
        :rtype: AcquisitionGeometry       
     '''
        AG = AcquisitionGeometry('', new_setup=True)
        AG.config = Configuration()
        AG.config.system = Parallel3D(ray_direction, detector_position, detector_direction_x, detector_direction_y, rotation_axis_position, rotation_axis_direction)
        return AG            

    @staticmethod
    def create_Cone3D(source_position, detector_position, detector_direction_x=[1,0,0], detector_direction_y=[0,0,1], rotation_axis_position=[0,0,0], rotation_axis_direction=[0,0,1]):
        r'''This creates the AcquisitionGeometry for a cone beam 3D tomographic system
                        
        :param source_position: A 3D vector describing the position of the source (x,y,z)
        :type source_position: list, tuple, ndarray, optional
        :param detector_position: A 3D vector describing the position of the centre of the detector (x,y,z)
        :type detector_position: list, tuple, ndarray, optional
        :param detector_direction_x: A 3D vector describing the direction of the detector_x (x,y,z)
        :type detector_direction_x: list, tuple, ndarray
        :param detector_direction_y: A 3D vector describing the direction of the detector_y (x,y,z)
        :type detector_direction_y: list, tuple, ndarray
        :param rotation_axis_position: A 3D vector describing the position of the axis of rotation (x,y,z)
        :type rotation_axis_position: list, tuple, ndarray, optional
        :param rotation_axis_direction: A 3D vector describing the direction of the axis of rotation (x,y,z)
        :type rotation_axis_direction: list, tuple, ndarray, optional
        :return: returns a configured AcquisitionGeometry object
        :rtype: AcquisitionGeometry           
        '''
        AG = AcquisitionGeometry('',  new_setup=True)
        AG.config = Configuration()
        AG.config.system = Cone3D(source_position, detector_position, detector_direction_x, detector_direction_y, rotation_axis_position, rotation_axis_direction)
        return AG          

    def get_order_by_label(self, dimension_labels, default_dimension_labels):
        order = []
        for i, el in enumerate(default_dimension_labels):
            for j, ek in enumerate(dimension_labels):
                if el == ek:
                    order.append(j)
                    break
        return order

    def __eq__(self, other):

        if isinstance(other, self.__class__) and self.config == other.config :
            return True
        return False

    def clone(self):
        '''returns a copy of the AcquisitionGeometry'''
        return copy.deepcopy(self)

    def copy(self):
        '''alias of clone'''
        return self.clone()

    def get_centre_slice(self):
        '''returns a 2D AcquisitionGeometry that corresponds to the centre slice of the input'''

        if self.dimension == '2D':
            return self
              
        AG_2D = copy.deepcopy(self)
        AG_2D.config.system = self.config.system.get_centre_slice()
        AG_2D.config.panel.num_pixels[1] = 1
        AG_2D.config.panel.pixel_size[1] = abs(self.config.system.detector.direction_y[2]) * self.config.panel.pixel_size[1]
        return AG_2D

    def get_ImageGeometry(self, resolution=1.0):
        '''returns a default configured ImageGeometry object based on the AcquisitionGeomerty'''

        num_voxel_xy = int(numpy.ceil(self.config.panel.num_pixels[0] * resolution))
        voxel_size_xy = self.config.panel.pixel_size[0] / (resolution * self.magnification)

        if self.dimension == '3D':
            num_voxel_z = int(numpy.ceil(self.config.panel.num_pixels[1] * resolution))
            voxel_size_z = self.config.panel.pixel_size[1] / (resolution * self.magnification)
        else:
            num_voxel_z = 0
            voxel_size_z = 1
            
        return ImageGeometry(num_voxel_xy, num_voxel_xy, num_voxel_z, voxel_size_xy, voxel_size_xy, voxel_size_z, channels=self.channels)

    def __str__ (self):
        return str(self.config)

    def subset(self, dimensions=None, **kw):
        '''Returns a new sliced and/or reshaped AcquisitionGeometry'''
        
        if not kw.get('suppress_warning', False):
            warnings.warn('Subset has been deprecated and will be removed in following version. Use reorder() and get_slice() instead',
              DeprecationWarning)
 
        if dimensions is None:
            return self.get_slice(**kw)
        else:
            if len(dimensions) != len(self.dimension_labels):
                raise ValueError('The axes list for subset must contain the dimension_labels {0} got {1}'.format(self.dimension_labels, dimensions))

            temp = self.copy()
            temp.set_labels(dimensions)
            return temp        

    def get_slice(self, channel=None, angle=None, vertical=None, horizontal=None):
        '''
        Returns a new AcquisitionGeometry of a single slice of in the requested direction. Will only return reconstructable geometries.
        '''
        geometry_new = self.copy()

        if channel is not None:
            geometry_new.config.channels.num_channels = 1
            if hasattr(geometry_new.config.channels,'channel_labels'):
                geometry_new.config.panel.channel_labels = geometry_new.config.panel.channel_labels[channel]

        if angle is not None:
            geometry_new.config.angles.angle_data = geometry_new.config.angles.angle_data[angle]
        
        if vertical is not None:
            if geometry_new.geom_type == AcquisitionGeometry.PARALLEL or vertical == 'centre' or vertical == geometry_new.pixel_num_v//2:
                geometry_new = geometry_new.get_centre_slice()
            else:
                raise ValueError("Can only subset centre slice geometry on cone-beam data. Expected vertical = 'centre'. Got vertical = {0}".format(vertical))
        
        if horizontal is not None:
            raise ValueError("Cannot calculate system geometry for a horizontal slice")

        return geometry_new

    def allocate(self, value=0, **kwargs):
        '''allocates an AcquisitionData according to the size expressed in the instance
        
        :param value: accepts numbers to allocate an uniform array, or a string as 'random' or 'random_int' to create a random array or None.
        :type value: number or string, default None allocates empty memory block
        :param dtype: numerical type to allocate
        :type dtype: numpy type, default numpy.float32
        '''
        dtype = kwargs.get('dtype', self.dtype)

        if kwargs.get('dimension_labels', None) is not None:
            raise ValueError("Deprecated: 'dimension_labels' cannot be set with 'allocate()'. Use 'geometry.set_labels()' to modify the geometry before using allocate.")

        out = AcquisitionData(geometry=self.copy(), 
                                dtype=dtype,
                                suppress_warning=True)

        if isinstance(value, Number):
            # it's created empty, so we make it 0
            out.array.fill(value)
        else:
            if value == AcquisitionGeometry.RANDOM:
                seed = kwargs.get('seed', None)
                if seed is not None:
                    numpy.random.seed(seed)
                if dtype in [ numpy.complex , numpy.complex64 , numpy.complex128 ] :
                    r = numpy.random.random_sample(self.shape) + 1j * numpy.random.random_sample(self.shape)
                    out.fill(r)
                else:
                    out.fill(numpy.random.random_sample(self.shape))
            elif value == AcquisitionGeometry.RANDOM_INT:
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

        if self.__dimension_labels is None:
            default_labels = [0]*self.number_of_dimensions
            for i in range(self.number_of_dimensions):
                default_labels[i] = 'dimension_{0:02}'.format(i)
            return tuple(default_labels)
        else:
            return self.__dimension_labels
      
    @dimension_labels.setter
    def dimension_labels(self, val):
        if val is None:
            self.__dimension_labels = None
        elif len(list(val))==self.number_of_dimensions:
            self.__dimension_labels = tuple(val)
        else:
            raise ValueError("dimension_labels expected a list containing {0} strings got {1}".format(self.number_of_dimensions, val))

    @property
    def shape(self):
        '''Returns the shape of the  of the DataContainer'''
        return self.array.shape

    @shape.setter
    def shape(self, val):
        print("Deprecated - shape will be set automatically")

    @property
    def number_of_dimensions(self):
        '''Returns the shape of the  of the DataContainer'''
        return len(self.array.shape)

    @property
    def dtype(self):
        '''Returns the dtype of the data array. 
           If geometry exists, the dtype of the geometry = dtype of the array'''                          
        self.geometry.dtype = self.array.dtype       
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

        if dimension_label in self.dimension_labels:
            return self.dimension_labels.index(dimension_label)
        else:
            raise ValueError('Unknown dimension {0}. Should be one of {1}'.format(dimension_label,
                             self.dimension_labels))
                        
    def as_array(self):
        '''Returns the pointer to the array.
        '''
        return self.array

    def subset(self, dimensions=None, **kw):
        '''Creates a DataContainer containing a subset of self according to the 
        labels in dimensions'''
        
        if not kw.get('suppress_warning', False):
            warnings.warn('Subset has been deprecated and will be removed in following version. Use reorder() and get_slice() instead',
              DeprecationWarning)

        if dimensions is None:
            return self.get_slice(**kw)
        else:
            temp = self.copy()
            temp.reorder(dimensions)
            return temp

    def get_slice(self,**kw):
        '''
        Returns a new DataContainer containing a single slice of in the requested direction. \
        Pass keyword arguments <dimension label>=index
        '''
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

        if order == 'astra' or order == 'tigre':
            order = DataOrder.get_order_for_engine(order, self.geometry)  

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
                                     'Expecting {0} got {1}'.format(
                                     self.shape,array.shape))
                numpy.copyto(self.array, array)
            elif isinstance(array, Number):
                self.array.fill(array) 
            elif issubclass(array.__class__ , DataContainer):
                if hasattr(self, 'geometry') and hasattr(array, 'geometry'):
                    if self.geometry != array.geometry:
                        numpy.copyto(self.array, array.subset(dimensions=array.dimension_labels).as_array())
                        return
                numpy.copyto(self.array, array.as_array())
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
        
        if new_order is None returns the labels of the axes as a sorted-by-key list
        if new_order is a list of length number_of_dimensions, returns a list
        with the indices of the axes in new_order with respect to those in 
        self.dimension_labels: i.e.
          self.dimension_labels = {0:'horizontal',1:'vertical'}
          new_order = ['vertical','horizontal']
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
            if isinstance(x2, (int, float, complex, \
                                 numpy.int, numpy.int8, numpy.int16, numpy.int32, numpy.int64,\
                                 numpy.float, numpy.float16, numpy.float32, numpy.float64, \
                                 numpy.complex)):
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
             isinstance(x2, (int,float,complex, numpy.int, numpy.int8, \
                             numpy.int16, numpy.int32, numpy.int64,\
                             numpy.float, numpy.float16, numpy.float32,\
                             numpy.float64, numpy.complex)):
            if self.check_dimensions(out):
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
        
        It will try to use the CIL C library and default to numpy operations, in case the C library does
        not handle the types.
        
        Example:
        -------

        a = 2
        b = 3
        ig = ImageGeometry(10,11)
        x = ig.allocate(1)
        y = ig.allocate(2)
        out = x.sapyb(a,y,b)
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


    def axpby(self, a, b, y, out, dtype=numpy.float32, num_threads=NUM_THREADS):
        '''Deprecated. Alias of _axpby'''
        warnings.warn('The use of axpby is deprecated and will be removed in following version. Use sapyb instead',
              DeprecationWarning)
        self._axpby(a,b,y,out, dtype, num_threads)


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
    def sum(self, *args, **kwargs):
        return self.as_array().sum(*args, **kwargs)
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
        '''return the inner product of 2 DataContainers viewed as vectors
        
        applies to real and complex data. In such case the dot method returns

        a.dot(b.conjugate())
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
    
    def min(self, *args, **kwargs):
        '''Returns the min pixel value in the DataContainer'''
        return numpy.min(self.as_array(), *args, **kwargs)
    
    def max(self, *args, **kwargs):
        '''Returns the max pixel value in the DataContainer'''
        return numpy.max(self.as_array(), *args, **kwargs)
    
    def mean(self, *args, **kwargs):
        '''Returns the mean pixel value of the DataContainer'''
        if kwargs.get('dtype', None) is None:
            kwargs['dtype'] = numpy.float64
        return numpy.mean(self.as_array(), *args, **kwargs)


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
        return self.__geometry

    @geometry.setter
    def geometry(self, val):
        self.__geometry = val

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

        if not kwargs.get('suppress_warning', False):
            warnings.warn('Direct invocation is deprecated and will be removed in following version. Use allocate from ImageGeometry instead',
              DeprecationWarning)

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
                               
    def subset(self, dimensions=None, **kw):
        '''returns a subset of ImageData and regenerates the geometry'''
        
        if not kw.get('suppress_warning', False):
            warnings.warn('Subset has been deprecated and will be removed in following version. Use reorder() and get_slice() instead',
              DeprecationWarning)

        if dimensions is None:
            return self.get_slice(**kw)
        else:
            temp = self.copy()
            temp.reorder(dimensions)
            return temp

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

class AcquisitionData(DataContainer):
    '''DataContainer for holding 2D or 3D sinogram'''
    __container_priority__ = 1

    @property
    def geometry(self):
        return self.__geometry

    @geometry.setter
    def geometry(self, val):
        self.__geometry = val

    @property
    def dimension_labels(self):
        return self.geometry.dimension_labels

    @dimension_labels.setter
    def dimension_labels(self, val):
        if val is not None:
            raise ValueError("Unable to set the dimension_labels directly. Use geometry.set_labels() instead")

    def __init__(self, 
                 array = None, 
                 deep_copy=True, 
                 geometry = None,
                 **kwargs):
        if not kwargs.get('suppress_warning', False):
            warnings.warn('Direct invocation is deprecated and will be removed in following version. Use allocate from AcquisitionGeometry instead',
              DeprecationWarning)

        dtype = kwargs.get('dtype', numpy.float32)

        if geometry is None:
            raise AttributeError("AcquisitionData requires a geometry")
        
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
            raise ValueError('Shape mismatch got {} expected {}'.format(array.shape, geometry.shape))
    
        super(AcquisitionData, self).__init__(array, deep_copy, geometry=geometry,**kwargs)
  
    def subset(self, dimensions=None, **kw):
        '''returns a subset of the AcquisitionData and regenerates the geometry'''
        
        if not kw.get('suppress_warning', False):
            warnings.warn('Subset has been deprecated and will be removed in following version. Use reorder() and get_slice() instead',
              DeprecationWarning)

        if dimensions is None:
            return self.get_slice(**kw)
        else:
            temp = self.copy()
            temp.reorder(dimensions)
            return temp

    def get_slice(self,channel=None, angle=None, vertical=None, horizontal=None, force=False):
        '''
        Returns a new dataset of a single slice of in the requested direction. \
        '''
        try:
            geometry_new = self.geometry.get_slice(channel=channel, angle=angle, vertical=vertical, horizontal=horizontal)
        except ValueError:
            if force:
                geometry_new = None
            else:
                raise ValueError ("Unable to return slice of requested AcquisitionData. Use 'force=True' to return DataContainer instead.")

        #get new data
        #if vertical = 'centre' slice convert to index and subset, this will interpolate 2 rows to get the center slice value
        if vertical == 'centre':
            dim = self.geometry.dimension_labels.index('vertical')
            
            centre_slice_pos = (self.geometry.shape[dim]-1) / 2.
            ind0 = int(numpy.floor(centre_slice_pos))
            w2 = centre_slice_pos - ind0
            out = DataContainer.get_slice(self, channel=channel, angle=angle, vertical=ind0, horizontal=horizontal)
            
            if w2 > 0:
                out2 = DataContainer.get_slice(self, channel=channel, angle=angle, vertical=ind0 + 1, horizontal=horizontal)
                out = out * (1 - w2) + out2 * w2
        else:
            out = DataContainer.get_slice(self, channel=channel, angle=angle, vertical=vertical, horizontal=horizontal)

        if len(out.shape) == 1 or geometry_new is None:
            return out
        else:
            return AcquisitionData(out.array, deep_copy=False, geometry=geometry_new, suppress_warning=True)

class Processor(object):

    '''Defines a generic DataContainer processor
                       
    accepts a DataContainer as input
    returns a DataContainer
    `__setattr__` allows additional attributes to be defined

    `store_output` boolian defining whether a copy of the output is stored. Default is False.
    If no attributes are modified get_output will return this stored copy bypassing `process`
    '''
    def __init__(self, **attributes):
        if not 'store_output' in attributes.keys():
            attributes['store_output'] = False

        attributes['output'] = None
        attributes['shouldRun'] = True
        attributes['input'] = None

        for key, value in attributes.items():
            self.__dict__[key] = value
        
    def __setattr__(self, name, value):
        if name == 'input':
            self.set_input(value)
        elif name in self.__dict__.keys():

            self.__dict__[name] = value

            if name == 'shouldRun':
                pass
            elif name == 'output':
                self.__dict__['shouldRun'] = False
            else:            
                self.__dict__['shouldRun'] = True
        else:
            raise KeyError('Attribute {0} not found'.format(name))
    
    def set_input(self, dataset):
        """
        Set the input data to the processor

        Parameters
        ----------
        input : DataContainer
            The input DataContainer
        """

        if issubclass(type(dataset), DataContainer):
            if self.check_input(dataset):
                self.__dict__['input'] = dataset
                self.__dict__['shouldRun'] = True
            else:
                raise ValueError('Input data not compatible')
        else:
            raise TypeError("Input type mismatch: got {0} expecting {1}"\
                            .format(type(dataset), DataContainer))
    
    def clear_input(self):
        self.__dict__['input']= None

    def check_input(self, dataset):
        '''Checks parameters of the input DataContainer
        
        Should raise an Error if the DataContainer does not match expectation, e.g.
        if the expected input DataContainer is 3D and the Processor expects 2D.
        '''
        raise NotImplementedError('Implement basic checks for input DataContainer')
        
    def get_output(self, out=None):
        """
        Runs the configured processor and returns the processed data

        Parameters
        ----------
        out : DataContainer, optional
           Fills the referenced DataContainer with the processed data and suppresses the return
        
        Returns
        -------
        DataContainer
            The processed data. Suppressed if `out` is passed
        """
        if self.output is None or self.shouldRun:
            if out is None:
                out = self.process()
            else:
                self.process(out=out)

            if self.store_output: 
                self.output = out.copy()
            
            return out

        else:
            return self.output.copy()
            
    
    def set_input_processor(self, processor):
        if issubclass(type(processor), DataProcessor):
            self.__dict__['input'] = processor
        else:
            raise TypeError("Input type mismatch: got {0} expecting {1}"\
                            .format(type(processor), DataProcessor))
        
    def get_input(self):
        '''returns the input DataContainer
        
        It is useful in the case the user has provided a DataProcessor as
        input
        '''
        if issubclass(type(self.input), DataProcessor):
            dsi = self.input.get_output()
        else:
            dsi = self.input
        return dsi
        
    def process(self, out=None):
        raise NotImplementedError('process must be implemented')
    
    def __call__(self, x, out=None):
        
        self.set_input(x)    

        if out is None:
            out = self.get_output()      
        else:
            self.get_output(out=out)

        self.clear_input()
        
        return out


class DataProcessor(Processor):
    '''Basically an alias of Processor Class'''
    pass

class DataProcessor23D(DataProcessor):
    '''Regularizers DataProcessor
    '''
            
    def check_input(self, dataset):
        '''Checks number of dimensions input DataContainer
        
        Expected input is 2D or 3D
        '''
        if dataset.number_of_dimensions == 2 or \
           dataset.number_of_dimensions == 3:
               return True
        else:
            raise ValueError("Expected input dimensions is 2 or 3, got {0}"\
                             .format(dataset.number_of_dimensions))
    
###### Example of DataProcessors

class AX(DataProcessor):
    '''Example DataProcessor
    The AXPY routines perform a vector multiplication operation defined as

    y := a*x
    where:

    a is a scalar

    x a DataContainer.
    '''
    
    def __init__(self):
        kwargs = {'scalar':None, 
                  'input':None, 
                  }
        
        #DataProcessor.__init__(self, **kwargs)
        super(AX, self).__init__(**kwargs)
    
    def check_input(self, dataset):
        return True
        
    def process(self, out=None):
        
        dsi = self.get_input()
        a = self.scalar
        if out is None:
            y = DataContainer( a * dsi.as_array() , True, 
                        dimension_labels=dsi.dimension_labels )
            #self.setParameter(output_dataset=y)
            return y
        else:
            out.fill(a * dsi.as_array())
    

###### Example of DataProcessors

class CastDataContainer(DataProcessor):
    '''Example DataProcessor
    Cast a DataContainer array to a different type.

    y := a*x
    where:

    a is a scalar

    x a DataContainer.
    '''
    
    def __init__(self, dtype=None):
        kwargs = {'dtype':dtype, 
                  'input':None, 
                  }
        
        #DataProcessor.__init__(self, **kwargs)
        super(CastDataContainer, self).__init__(**kwargs)
    
    def check_input(self, dataset):
        return True
        
    def process(self, out=None):
        
        dsi = self.get_input()
        dtype = self.dtype
        if out is None:
            y = numpy.asarray(dsi.as_array(), dtype=dtype)
            
            return type(dsi)(numpy.asarray(dsi.as_array(), dtype=dtype),
                                dimension_labels=dsi.dimension_labels )
        else:
            out.fill(numpy.asarray(dsi.as_array(), dtype=dtype))
    
class PixelByPixelDataProcessor(DataProcessor):
    '''Example DataProcessor
    
    This processor applies a python function to each pixel of the DataContainer
    
    f is a python function

    x a DataSet.
    '''
    
    def __init__(self):
        kwargs = {'pyfunc':None, 
                  'input':None, 
                  }
        #DataProcessor.__init__(self, **kwargs)
        super(PixelByPixelDataProcessor, self).__init__(**kwargs)
        
    def check_input(self, dataset):
        return True
    
    def process(self, out=None):
        
        pyfunc = self.pyfunc
        dsi = self.get_input()
        
        eval_func = numpy.frompyfunc(pyfunc,1,1)

        
        y = DataContainer( eval_func( dsi.as_array() ) , True, 
                    dimension_labels=dsi.dimension_labels )
        return y
    

class VectorData(DataContainer):
    '''DataContainer to contain 1D array'''

    @property
    def geometry(self):
        return self.__geometry

    @geometry.setter
    def geometry(self, val):
        self.__geometry = val

    @property
    def dimension_labels(self):
        if hasattr(self,'geometry'):
            return self.geometry.dimension_labels
        else:
            return self.__dimension_labels

    @dimension_labels.setter
    def dimension_labels(self, val):
        if hasattr(self,'geometry'):
            self.geometry.dimension_labels = val
        
        self.__dimension_labels = val

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
        return self.__dtype

    @dtype.setter
    def dtype(self, val):
        self.__dtype = val      
        
    def __init__(self, 
                 length, **kwargs):
        
        self.length = length
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

class DataOrder():
    ASTRA_IG_LABELS = [ImageGeometry.CHANNEL, ImageGeometry.VERTICAL, ImageGeometry.HORIZONTAL_Y, ImageGeometry.HORIZONTAL_X]
    TIGRE_IG_LABELS = [ImageGeometry.CHANNEL, ImageGeometry.VERTICAL, ImageGeometry.HORIZONTAL_Y, ImageGeometry.HORIZONTAL_X]
    ASTRA_AG_LABELS = [AcquisitionGeometry.CHANNEL, AcquisitionGeometry.VERTICAL, AcquisitionGeometry.ANGLE, AcquisitionGeometry.HORIZONTAL]
    TIGRE_AG_LABELS = [AcquisitionGeometry.CHANNEL, AcquisitionGeometry.ANGLE, AcquisitionGeometry.VERTICAL, AcquisitionGeometry.HORIZONTAL]
    CIL_IG_LABELS = [ImageGeometry.CHANNEL, ImageGeometry.VERTICAL, ImageGeometry.HORIZONTAL_Y, ImageGeometry.HORIZONTAL_X]
    CIL_AG_LABELS = [AcquisitionGeometry.CHANNEL, AcquisitionGeometry.ANGLE, AcquisitionGeometry.VERTICAL, AcquisitionGeometry.HORIZONTAL] 
    TOMOPHANTOM_IG_LABELS = [ImageGeometry.CHANNEL, ImageGeometry.VERTICAL, ImageGeometry.HORIZONTAL_Y, ImageGeometry.HORIZONTAL_X]
    
    @staticmethod
    def get_order_for_engine(engine, geometry):
        if engine == 'astra':
            if isinstance(geometry, AcquisitionGeometry):
                dim_order = DataOrder.ASTRA_AG_LABELS
            else:
                dim_order = DataOrder.ASTRA_IG_LABELS
        elif engine == 'tigre':
            if isinstance(geometry, AcquisitionGeometry):
                dim_order = DataOrder.TIGRE_AG_LABELS
            else:
                dim_order = DataOrder.TIGRE_IG_LABELS
        elif engine == 'cil':
            if isinstance(geometry, AcquisitionGeometry):
                dim_order = DataOrder.CIL_AG_LABELS
            else:
                dim_order = DataOrder.CIL_IG_LABELS
        else:
            raise ValueError("Unknown engine expected 'tigre' or 'astra' got {}".format(engine))
        
        dimensions = []
        for label in dim_order:
            if label in geometry.dimension_labels:
                dimensions.append(label)

        return dimensions

    @staticmethod
    def check_order_for_engine(engine, geometry):
        order_requested = DataOrder.get_order_for_engine(engine, geometry)

        if order_requested == list(geometry.dimension_labels):
            return True
        else:
            raise ValueError("Expected dimension_label order {0}, got {1}.\nTry using `data.reorder('{2}')` to permute for {2}"
                 .format(order_requested, list(geometry.dimension_labels), engine))
