# -*- coding: utf-8 -*-
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
import numpy
from numbers import Number
import ctypes, platform
from ctypes import util
import math
import weakref
import logging

from Wrappers.Python.cil.framework import DataContainer, Partitioner

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
    def ndim(self):
        return len(self.dimension_labels)

    @property
    def dimension_labels(self):
        
        labels_default = DataOrder.CIL_IG_LABELS

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
        labels_default = DataOrder.CIL_IG_LABELS

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
            if value == ImageGeometry.RANDOM:
                seed = kwargs.get('seed', None)
                if seed is not None:
                    numpy.random.seed(seed)
                if numpy.iscomplexobj(out.array):
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
        self._dof = dof

    @staticmethod  
    def create_vector(val):
        try:
            vec = numpy.array(val, dtype=numpy.float64).reshape(len(val))
        except:
            raise ValueError("Can't convert to numpy array")
   
        return vec

    @staticmethod   
    def create_unit_vector(val):
        vec = ComponentDescription.create_vector(val)
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
            raise ValueError("Vectors for {0}D geometries must have length = {0}. Got {1}".format(self._dof,val))
        
        if val_length != self._dof:
            raise ValueError("Vectors for {0}D geometries must have length = {0}. Got {1}".format(self._dof,val))

    @staticmethod   
    def test_perpendicular(vector1, vector2):
        dor_prod = vector1.dot(vector2)
        if abs(dor_prod) <1e-10:
            return True
        return False

    @staticmethod   
    def test_parallel(vector1, vector2):
        '''For unit vectors only. Returns true if directions are opposite'''
        dor_prod = vector1.dot(vector2)
        if 1- abs(dor_prod) <1e-10:
            return True
        return False

class PositionVector(ComponentDescription):
    r'''This class creates a component of a tomography system with a position attribute
     '''
    @property
    def position(self):
        try:
            return self._position
        except:
            raise AttributeError

    @position.setter
    def position(self, val):  
        self.length_check(val)
        self._position = ComponentDescription.create_vector(val)


class DirectionVector(ComponentDescription):
    r'''This class creates a component of a tomography system with a direction attribute
     '''
    @property
    def direction(self):      
        try:
            return self._direction
        except:
            raise AttributeError

    @direction.setter
    def direction(self, val):
        self.length_check(val)    
        self._direction = ComponentDescription.create_unit_vector(val)

 
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
            return self._direction_x
        except:
            raise AttributeError

    @direction_x.setter
    def direction_x(self, val):
        self.length_check(val)
        self._direction_x = ComponentDescription.create_unit_vector(val)

    @property
    def normal(self):
        try:
            return ComponentDescription.create_unit_vector([self._direction_x[1], -self._direction_x[0]])
        except:
            raise AttributeError


class Detector2D(PositionVector):
    r'''This class creates a component of a tomography system with position, direction_x and direction_y attributes used for 2D panels
     '''
    @property
    def direction_x(self):
        try:
            return self._direction_x
        except:
            raise AttributeError

    @property
    def direction_y(self):
        try:
            return self._direction_y
        except:
            raise AttributeError

    @property
    def normal(self):
        try:
            return numpy.cross(self._direction_x, self._direction_y)
        except:
            raise AttributeError

    def set_direction(self, x, y):
        self.length_check(x)
        x = ComponentDescription.create_unit_vector(x)

        self.length_check(y)
        y = ComponentDescription.create_unit_vector(y)

        dot_product = x.dot(y)
        if not numpy.isclose(dot_product, 0):
            raise ValueError("vectors detector.direction_x and detector.direction_y must be orthogonal")

        self._direction_y = y        
        self._direction_x = x

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
        return self._geometry

    @geometry.setter
    def geometry(self,val):
        if val != AcquisitionGeometry.CONE and val != AcquisitionGeometry.PARALLEL:
            raise ValueError('geom_type = {} not recognised please specify \'cone\' or \'parallel\''.format(val))
        else:
            self._geometry = val

    def __init__(self, dof, geometry, units='units'): 
        """Initialises the system component attributes for the acquisition type
        """                
        self.dimension = dof
        self.geometry = geometry
        self.units = units
        
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

    @staticmethod
    def rotation_vec_to_y(vec):
        ''' returns a rotation matrix that will rotate the projection of vec on the x-y plane to the +y direction [0,1, Z]'''
    
        vec = ComponentDescription.create_unit_vector(vec)

        axis_rotation = numpy.eye(len(vec))

        if numpy.allclose(vec[:2],[0,1]):
            pass
        elif numpy.allclose(vec[:2],[0,-1]):
            axis_rotation[0][0] = -1
            axis_rotation[1][1] = -1
        else:
            theta = math.atan2(vec[0],vec[1])
            axis_rotation[0][0] = axis_rotation[1][1] = math.cos(theta)
            axis_rotation[0][1] = -math.sin(theta)
            axis_rotation[1][0] = math.sin(theta)

        return axis_rotation

    @staticmethod
    def rotation_vec_to_z(vec):
        ''' returns a rotation matrix that will align vec with the z-direction [0,0,1]'''

        vec = ComponentDescription.create_unit_vector(vec)

        if len(vec) == 2:
            return numpy.array([[1, 0],[0, 1]])

        elif len(vec) == 3:
            axis_rotation = numpy.eye(3)

            if numpy.allclose(vec,[0,0,1]):
                pass
            elif numpy.allclose(vec,[0,0,-1]):
                axis_rotation = numpy.eye(3)
                axis_rotation[1][1] = -1
                axis_rotation[2][2] = -1
            else:
                vx = numpy.array([[0, 0, -vec[0]], [0, 0, -vec[1]], [vec[0], vec[1], 0]])
                axis_rotation = numpy.eye(3) + vx + vx.dot(vx) *  1 / (1 + vec[2])

        else:
            raise ValueError("Vec must have length 3, got {}".format(len(vec)))
    
        return axis_rotation

    def update_reference_frame(self):
        r'''Transforms the system origin to the rotation_axis position
        '''   
        self.set_origin(self.rotation_axis.position)


    def set_origin(self, origin):
        r'''Transforms the system origin to the input origin
        '''
        translation = origin.copy()
        if hasattr(self,'source'):              
            self.source.position -= translation

        self.detector.position -= translation
        self.rotation_axis.position -= translation


    def get_centre_slice(self):
        """Returns the 2D system configuration corresponding to the centre slice
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
    :param units: Label the units of distance used for the configuration
    :type units: string
    '''

    def __init__ (self, ray_direction, detector_pos, detector_direction_x, rotation_axis_pos, units='units'):
        """Constructor method
        """
        super(Parallel2D, self).__init__(dof=2, geometry = 'parallel', units=units)

        #source
        self.ray.direction = ray_direction

        #detector
        self.detector.position = detector_pos
        self.detector.direction_x = detector_direction_x
        
        #rotate axis
        self.rotation_axis.position = rotation_axis_pos


    def align_reference_frame(self, definition='cil'):
        r'''Transforms and rotates the system to backend definitions

        'cil' sets the origin to the rotation axis and aligns the y axis with the ray-direction
        'tigre' sets the origin to the rotation axis and aligns the y axis with the ray-direction
        '''
        #in this instance definitions are the same
        if definition not in ['cil','tigre']:
            raise ValueError("Geometry can be configured for definition = 'cil' or 'tigre'  only. Got {}".format(definition))

        self.set_origin(self.rotation_axis.position)

        rotation_matrix = SystemConfiguration.rotation_vec_to_y(self.ray.direction)

        self.ray.direction = rotation_matrix.dot(self.ray.direction.reshape(2,1))
        self.detector.position = rotation_matrix.dot(self.detector.position.reshape(2,1))
        self.detector.direction_x = rotation_matrix.dot(self.detector.direction_x.reshape(2,1))


    def system_description(self):
        r'''Returns `simple` if the the geometry matches the default definitions with no offsets or rotations,
            \nReturns `offset` if the the geometry matches the default definitions with centre-of-rotation or detector offsets
            \nReturns `advanced` if the the geometry has rotated or tilted rotation axis or detector, can also have offsets
        '''       


        rays_perpendicular_detector = ComponentDescription.test_parallel(self.ray.direction, self.detector.normal)

        #rotation axis position + ray direction hits detector position
        if numpy.allclose(self.rotation_axis.position, self.detector.position): #points are equal so on ray path
            rotation_axis_centred = True
        else:
            vec_a = ComponentDescription.create_unit_vector(self.detector.position - self.rotation_axis.position)
            rotation_axis_centred = ComponentDescription.test_parallel(self.ray.direction, vec_a)

        if not rays_perpendicular_detector: 
            config = SystemConfiguration.SYSTEM_ADVANCED
        elif not rotation_axis_centred:
            config =  SystemConfiguration.SYSTEM_OFFSET
        else:
            config =  SystemConfiguration.SYSTEM_SIMPLE

        return config


    def rotation_axis_on_detector(self):
        """
        Calculates the position, on the detector, of the projection of the rotation axis in the world coordinate system

        Returns
        -------
        PositionVector
            Position in the 3D system
        """
        Pv = self.rotation_axis.position
        ratio = (self.detector.position - Pv).dot(self.detector.normal) / self.ray.direction.dot(self.detector.normal)
        out = PositionVector(2)
        out.position = Pv + self.ray.direction * ratio
        return out

    def calculate_centre_of_rotation(self):
        """
        Calculates the position, on the detector, of the projection of the rotation axis in the detector coordinate system

        Note
        ----
         - Origin is in the centre of the detector
         - Axes directions are specified by detector.direction_x, detector.direction_y
         - Units are the units of distance used to specify the component's positions

        Returns
        -------
        Float
            Offset position along the detector x_axis at y=0
        Float
            Angle between the y_axis and the rotation axis projection, in radians
        """

        #convert to the detector coordinate system
        dp1 = self.rotation_axis_on_detector().position - self.detector.position
        offset = self.detector.direction_x.dot(dp1)

        return (offset, 0.0)

    def set_centre_of_rotation(self, offset):
        """ Configures the geometry to have the requested centre of rotation offset at the detector
        """
        offset_current = self.calculate_centre_of_rotation()[0]
        offset_new = offset - offset_current

        self.rotation_axis.position = self.rotation_axis.position + offset_new * self.detector.direction_x

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
    :param units: Label the units of distance used for the configuration
    :type units: string
    '''

    def __init__ (self,  ray_direction, detector_pos, detector_direction_x, detector_direction_y, rotation_axis_pos, rotation_axis_direction, units='units'):
        """Constructor method
        """
        super(Parallel3D, self).__init__(dof=3, geometry = 'parallel', units=units)
                    
        #source
        self.ray.direction = ray_direction

        #detector
        self.detector.position = detector_pos
        self.detector.set_direction(detector_direction_x, detector_direction_y)

        #rotate axis
        self.rotation_axis.position = rotation_axis_pos
        self.rotation_axis.direction = rotation_axis_direction

    def align_z(self):
        r'''Transforms the system origin to the rotate axis with z direction aligned to the rotate axis direction
        '''          
        self.set_origin(self.rotation_axis.position)

        #calculate rotation matrix to align rotation axis direction with z
        rotation_matrix = SystemConfiguration.rotation_vec_to_z(self.rotation_axis.direction)

        #apply transform
        self.rotation_axis.direction = [0,0,1]
        self.ray.direction = rotation_matrix.dot(self.ray.direction.reshape(3,1))
        self.detector.position = rotation_matrix.dot(self.detector.position.reshape(3,1))
        new_x = rotation_matrix.dot(self.detector.direction_x.reshape(3,1))
        new_y = rotation_matrix.dot(self.detector.direction_y.reshape(3,1))
        self.detector.set_direction(new_x, new_y)


    def align_reference_frame(self, definition='cil'):
        r'''Transforms and rotates the system to backend definitions
        '''
        #in this instance definitions are the same
        if definition not in ['cil','tigre']:
            raise ValueError("Geometry can be configured for definition = 'cil' or 'tigre'  only. Got {}".format(definition))

        self.align_z()
        rotation_matrix = SystemConfiguration.rotation_vec_to_y(self.ray.direction)
                
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


        '''
        simple
         - rays perpendicular to detector
         - rotation axis parallel to detector y
         - rotation axis position + ray direction hits detector with no x offset (y offsets allowed)
        offset
         - rays perpendicular to detector
         - rotation axis parallel to detector y
        rolled
         - rays perpendicular to detector
         - rays perpendicular to rotation axis
        advanced
         - not rays perpendicular to detector (for parallel just equates to an effective pixel size change?)
         or 
         - not rays perpendicular to rotation axis  (tilted, i.e. laminography)
        '''

        rays_perpendicular_detector = ComponentDescription.test_parallel(self.ray.direction, self.detector.normal)
        rays_perpendicular_rotation = ComponentDescription.test_perpendicular(self.ray.direction, self.rotation_axis.direction)
        rotation_parallel_detector_y = ComponentDescription.test_parallel(self.rotation_axis.direction, self.detector.direction_y)

        #rotation axis to detector is parallel with ray
        if numpy.allclose(self.rotation_axis.position, self.detector.position): #points are equal so on ray path
            rotation_axis_centred = True
        else:
            vec_a = ComponentDescription.create_unit_vector(self.detector.position - self.rotation_axis.position )
            rotation_axis_centred = ComponentDescription.test_parallel(self.ray.direction, vec_a)

        if not rays_perpendicular_detector or\
            not rays_perpendicular_rotation or\
            not rotation_parallel_detector_y: 
            config = SystemConfiguration.SYSTEM_ADVANCED
        elif not rotation_axis_centred:
            config =  SystemConfiguration.SYSTEM_OFFSET
        else:
            config =  SystemConfiguration.SYSTEM_SIMPLE

        return config
        

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
        """Returns the 2D system configuration corresponding to the centre slice
        """  
        dp1 = self.rotation_axis.direction.dot(self.ray.direction)
        dp2 = self.rotation_axis.direction.dot(self.detector.direction_x)

        if numpy.isclose(dp1, 0) and numpy.isclose(dp2, 0):
            temp = self.copy()

            #convert to rotation axis reference frame
            temp.align_reference_frame()

            ray_direction = temp.ray.direction[0:2]
            detector_position = temp.detector.position[0:2]
            detector_direction_x = temp.detector.direction_x[0:2]
            rotation_axis_position = temp.rotation_axis.position[0:2]

            return Parallel2D(ray_direction, detector_position, detector_direction_x, rotation_axis_position)

        else:
            raise ValueError('Cannot convert geometry to 2D. Requires axis of rotation to be perpendicular to ray direction and the detector direction x.')


    def rotation_axis_on_detector(self):
        """
        Calculates the position, on the detector, of the projection of the rotation axis in the world coordinate system

        Returns
        -------
        PositionDirectionVector
            Position and direction in the 3D system
        """
        #calculate the rotation axis line with the detector
        vec_a = self.ray.direction

        #calculate the intersection with the detector
        Pv = self.rotation_axis.position
        ratio = (self.detector.position - Pv).dot(self.detector.normal) / vec_a.dot(self.detector.normal)
        point1 = Pv + vec_a * ratio

        Pv = self.rotation_axis.position + self.rotation_axis.direction
        ratio = (self.detector.position - Pv).dot(self.detector.normal) / vec_a.dot(self.detector.normal)
        point2 = Pv + vec_a * ratio

        out = PositionDirectionVector(3)
        out.position = point1
        out.direction = point2 - point1
        return out


    def calculate_centre_of_rotation(self):
        """
        Calculates the position, on the detector, of the projection of the rotation axis in the detector coordinate system

        Note
        ----
         - Origin is in the centre of the detector
         - Axes directions are specified by detector.direction_x, detector.direction_y
         - Units are the units of distance used to specify the component's positions

        Returns
        -------
        Float
            Offset position along the detector x_axis at y=0
        Float
            Angle between the y_axis and the rotation axis projection, in radians
        """
        rotate_axis_projection = self.rotation_axis_on_detector()

        p1 = rotate_axis_projection.position
        p2 = p1 + rotate_axis_projection.direction

        #point1 and point2 are on the detector plane. need to return them in the detector coordinate system
        dp1 = p1 - self.detector.position
        x1 = self.detector.direction_x.dot(dp1)
        y1 = self.detector.direction_y.dot(dp1)
        dp2 = p2 - self.detector.position
        x2 = self.detector.direction_x.dot(dp2)
        y2 = self.detector.direction_y.dot(dp2)

        #y = m * x + c
        #c = y1 - m * x1
        #when y is 0
        #x=-c/m
        #x_y0 = -y1/m + x1 
        offset_x_y0 = x1 -y1 * (x2 - x1)/(y2-y1)

        angle = math.atan2(x2 - x1, y2 - y1)
        offset = offset_x_y0

        return (offset, angle)

    def set_centre_of_rotation(self, offset, angle):
        """ Configures the geometry to have the requested centre of rotation offset at the detector
        """

        #two points on the detector
        x1 = offset
        y1 = 0
        x2 = offset + math.tan(angle)
        y2 = 1

        #convert to 3d coordinates in system frame
        p1 = self.detector.position + x1 * self.detector.direction_x + y1 * self.detector.direction_y
        p2 = self.detector.position + x2 * self.detector.direction_x + y2 * self.detector.direction_y

        # find where vec p1 + t * ray dirn intersects plane defined by rotate axis (pos and dir) and det_x direction

        vector_pos=p1
        vec_dirn=self.ray.direction
        plane_pos=self.rotation_axis.position
        plane_normal = numpy.cross(self.detector.direction_x, self.rotation_axis.direction)


        ratio = (plane_pos - vector_pos).dot(plane_normal) / vec_dirn.dot(plane_normal)
        p1_on_plane = vector_pos + vec_dirn * ratio

        vector_pos=p2
        ratio = (plane_pos - vector_pos).dot(plane_normal) / vec_dirn.dot(plane_normal)
        p2_on_plane = vector_pos + vec_dirn * ratio

        self.rotation_axis.position = p1_on_plane
        self.rotation_axis.direction = p2_on_plane - p1_on_plane


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
    :param units: Label the units of distance used for the configuration
    :type units: string
    '''

    def __init__ (self, source_pos, detector_pos, detector_direction_x, rotation_axis_pos, units='units'):
        """Constructor method
        """
        super(Cone2D, self).__init__(dof=2, geometry = 'cone', units=units)

        #source
        self.source.position = source_pos

        #detector
        self.detector.position = detector_pos
        self.detector.direction_x = detector_direction_x

        #rotate axis
        self.rotation_axis.position = rotation_axis_pos


    def align_reference_frame(self, definition='cil'):
        r'''Transforms and rotates the system to backend definitions
        '''
        self.set_origin(self.rotation_axis.position)

        if definition=='cil':
            rotation_matrix = SystemConfiguration.rotation_vec_to_y(self.detector.position - self.source.position)
        elif definition=='tigre':
            rotation_matrix = SystemConfiguration.rotation_vec_to_y(self.rotation_axis.position - self.source.position)
        else:
            raise ValueError("Geometry can be configured for definition = 'cil' or 'tigre'  only. Got {}".format(definition))

        self.source.position = rotation_matrix.dot(self.source.position.reshape(2,1))
        self.detector.position = rotation_matrix.dot(self.detector.position.reshape(2,1))
        self.detector.direction_x = rotation_matrix.dot(self.detector.direction_x.reshape(2,1))


    def system_description(self):
        r'''Returns `simple` if the the geometry matches the default definitions with no offsets or rotations,
            \nReturns `offset` if the the geometry matches the default definitions with centre-of-rotation or detector offsets
            \nReturns `advanced` if the the geometry has rotated or tilted rotation axis or detector, can also have offsets
        '''           

        vec_src2det = ComponentDescription.create_unit_vector(self.detector.position - self.source.position)

        principal_ray_centred = ComponentDescription.test_parallel(vec_src2det, self.detector.normal)

        #rotation axis to detector is parallel with centre ray
        if numpy.allclose(self.rotation_axis.position, self.detector.position): #points are equal
            rotation_axis_centred = True
        else:
            vec_b = ComponentDescription.create_unit_vector(self.detector.position - self.rotation_axis.position )
            rotation_axis_centred = ComponentDescription.test_parallel(vec_src2det, vec_b)

        if not principal_ray_centred:
            config = SystemConfiguration.SYSTEM_ADVANCED
        elif not rotation_axis_centred:
            config =  SystemConfiguration.SYSTEM_OFFSET
        else:
            config =  SystemConfiguration.SYSTEM_SIMPLE

        return config

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

        ab = (self.rotation_axis.position - self.source.position)
        dist_source_center = float(numpy.sqrt(ab.dot(ab)))

        ab_unit = ab / numpy.sqrt(ab.dot(ab))

        n = self.detector.normal

        #perpendicular distance between source and detector centre
        sd = float((self.detector.position - self.source.position).dot(n))
        ratio = float(ab_unit.dot(n))

        source_to_detector = sd / ratio
        dist_center_detector = source_to_detector - dist_source_center
        magnification = (dist_center_detector + dist_source_center) / dist_source_center

        return [dist_source_center, dist_center_detector, magnification]

    def rotation_axis_on_detector(self):
        """
        Calculates the position, on the detector, of the projection of the rotation axis in the world coordinate system

        Returns
        -------
        PositionVector
            Position in the 3D system
        """
        #calculate the point the rotation axis intersects with the detector
        vec_a = self.rotation_axis.position - self.source.position

        Pv = self.rotation_axis.position
        ratio = (self.detector.position - Pv).dot(self.detector.normal) / vec_a.dot(self.detector.normal)

        out = PositionVector(2)
        out.position = Pv + vec_a * ratio

        return out


    def calculate_centre_of_rotation(self):
        """
        Calculates the position, on the detector, of the projection of the rotation axis in the detector coordinate system

        Note
        ----
         - Origin is in the centre of the detector
         - Axes directions are specified by detector.direction_x, detector.direction_y
         - Units are the units of distance used to specify the component's positions

        Returns
        -------
        Float
            Offset position along the detector x_axis at y=0
        Float
            Angle between the y_axis and the rotation axis projection, in radians
        """
        #convert to the detector coordinate system
        dp1 = self.rotation_axis_on_detector().position - self.detector.position
        offset = self.detector.direction_x.dot(dp1)

        return (offset, 0.0)

    def set_centre_of_rotation(self, offset):
        """ Configures the geometry to have the requested centre of rotation offset at the detector
        """
        offset_current = self.calculate_centre_of_rotation()[0]
        offset_new = offset - offset_current

        cofr_shift = offset_new * self.detector.direction_x /self.calculate_magnification()[2]
        self.rotation_axis.position =self.rotation_axis.position + cofr_shift

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
    :param units: Label the units of distance used for the configuration
    :type units: string
    '''

    def __init__ (self, source_pos, detector_pos, detector_direction_x, detector_direction_y, rotation_axis_pos, rotation_axis_direction, units='units'):
        """Constructor method
        """
        super(Cone3D, self).__init__(dof=3, geometry = 'cone', units=units)

        #source
        self.source.position = source_pos

        #detector
        self.detector.position = detector_pos
        self.detector.set_direction(detector_direction_x, detector_direction_y)

        #rotate axis
        self.rotation_axis.position = rotation_axis_pos
        self.rotation_axis.direction = rotation_axis_direction

    def align_z(self):
        r'''Transforms the system origin to the rotate axis with z direction aligned to the rotate axis direction
        '''
        self.set_origin(self.rotation_axis.position)   
        rotation_matrix = SystemConfiguration.rotation_vec_to_z(self.rotation_axis.direction)
    
        #apply transform
        self.rotation_axis.direction = [0,0,1]
        self.source.position = rotation_matrix.dot(self.source.position.reshape(3,1))
        self.detector.position = rotation_matrix.dot(self.detector.position.reshape(3,1))
        new_x = rotation_matrix.dot(self.detector.direction_x.reshape(3,1)) 
        new_y = rotation_matrix.dot(self.detector.direction_y.reshape(3,1))
        self.detector.set_direction(new_x, new_y)


    def align_reference_frame(self, definition='cil'):
        r'''Transforms and rotates the system to backend definitions
        '''            

        self.align_z()

        if definition=='cil':
            rotation_matrix = SystemConfiguration.rotation_vec_to_y(self.detector.position - self.source.position)
        elif definition=='tigre':
            rotation_matrix = SystemConfiguration.rotation_vec_to_y(self.rotation_axis.position - self.source.position)
        else:
            raise ValueError("Geometry can be configured for definition = 'cil' or 'tigre'  only. Got {}".format(definition))
                            
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

        vec_src2det = ComponentDescription.create_unit_vector(self.detector.position - self.source.position)

        principal_ray_centred = ComponentDescription.test_parallel(vec_src2det, self.detector.normal)
        centre_ray_perpendicular_rotation = ComponentDescription.test_perpendicular(vec_src2det, self.rotation_axis.direction)
        rotation_parallel_detector_y = ComponentDescription.test_parallel(self.rotation_axis.direction, self.detector.direction_y)

        #rotation axis to detector is parallel with centre ray
        if numpy.allclose(self.rotation_axis.position, self.detector.position): #points are equal
            rotation_axis_centred = True
        else:
            vec_b = ComponentDescription.create_unit_vector(self.detector.position - self.rotation_axis.position )
            rotation_axis_centred = ComponentDescription.test_parallel(vec_src2det, vec_b)

        if not principal_ray_centred or\
            not centre_ray_perpendicular_rotation or\
            not rotation_parallel_detector_y: 
            config = SystemConfiguration.SYSTEM_ADVANCED
        elif not rotation_axis_centred:
            config =  SystemConfiguration.SYSTEM_OFFSET
        else:
            config =  SystemConfiguration.SYSTEM_SIMPLE

        return config

    def get_centre_slice(self):
        """Returns the 2D system configuration corresponding to the centre slice
        """ 
        #requires the rotate axis to be perpendicular to the normal of the detector, and perpendicular to detector_direction_x
        dp1 = self.rotation_axis.direction.dot(self.detector.normal)
        dp2 = self.rotation_axis.direction.dot(self.detector.direction_x)
        
        if numpy.isclose(dp1, 0) and numpy.isclose(dp2, 0):
            temp = self.copy()
            temp.align_reference_frame()
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
    
        ab = (self.rotation_axis.position - self.source.position)
        dist_source_center = float(numpy.sqrt(ab.dot(ab)))

        ab_unit = ab / numpy.sqrt(ab.dot(ab))

        n = self.detector.normal

        #perpendicular distance between source and detector centre
        sd = float((self.detector.position - self.source.position).dot(n))
        ratio = float(ab_unit.dot(n))

        source_to_detector = sd / ratio
        dist_center_detector = source_to_detector - dist_source_center
        magnification = (dist_center_detector + dist_source_center) / dist_source_center

        return [dist_source_center, dist_center_detector, magnification]

    def rotation_axis_on_detector(self):
        """
        Calculates the position, on the detector, of the projection of the rotation axis in the world coordinate system

        Returns
        -------
        PositionDirectionVector
            Position and direction in the 3D system
        """
        #calculate the intersection with the detector, of source to pv
        Pv = self.rotation_axis.position
        vec_a = Pv - self.source.position
        ratio = (self.detector.position - Pv).dot(self.detector.normal) / vec_a.dot(self.detector.normal)
        point1 = Pv + vec_a * ratio

        #calculate the intersection with the detector, of source to pv
        Pv = self.rotation_axis.position + self.rotation_axis.direction
        vec_a = Pv - self.source.position
        ratio = (self.detector.position - Pv).dot(self.detector.normal) / vec_a.dot(self.detector.normal)
        point2 = Pv + vec_a * ratio

        out = PositionDirectionVector(3)
        out.position = point1
        out.direction = point2 - point1
        return out

    def calculate_centre_of_rotation(self):
        """
        Calculates the position, on the detector, of the projection of the rotation axis in the detector coordinate system

        Note
        ----
         - Origin is in the centre of the detector
         - Axes directions are specified by detector.direction_x, detector.direction_y
         - Units are the units of distance used to specify the component's positions

        Returns
        -------
        Float
            Offset position along the detector x_axis at y=0
        Float
            Angle between the y_axis and the rotation axis projection, in radians
        """
        rotate_axis_projection = self.rotation_axis_on_detector()

        p1 = rotate_axis_projection.position
        p2 = p1 + rotate_axis_projection.direction

        #point1 and point2 are on the detector plane. need to return them in the detector coordinate system
        dp1 = p1 - self.detector.position
        x1 = self.detector.direction_x.dot(dp1)
        y1 = self.detector.direction_y.dot(dp1)
        dp2 = p2 - self.detector.position
        x2 = self.detector.direction_x.dot(dp2)
        y2 = self.detector.direction_y.dot(dp2)

        #y = m * x + c
        #c = y1 - m * x1
        #when y is 0
        #x=-c/m
        #x_y0 = -y1/m + x1 
        offset_x_y0 = x1 -y1 * (x2 - x1)/(y2-y1)

        angle = math.atan2(x2 - x1, y2 - y1)
        offset = offset_x_y0

        return (offset, angle)


    def set_centre_of_rotation(self, offset, angle):
        """ Configures the geometry to have the requested centre of rotation offset at the detector
        """
        #two points on the detector
        x1 = offset
        y1 = 0
        x2 = offset + math.tan(angle)
        y2 = 1

        #convert to 3d coordinates in system frame
        p1 = self.detector.position + x1 * self.detector.direction_x + y1 * self.detector.direction_y
        p2 = self.detector.position + x2 * self.detector.direction_x + y2 * self.detector.direction_y

        # vectors from source define plane
        sp1 = p1 - self.source.position
        sp2 = p2 - self.source.position

        #find vector intersection with a plane defined by rotate axis (pos and dir) and det_x direction
        plane_normal = numpy.cross(self.rotation_axis.direction, self.detector.direction_x)

        ratio = (self.rotation_axis.position - self.source.position).dot(plane_normal) / sp1.dot(plane_normal)
        p1_on_plane = self.source.position + sp1 * ratio

        ratio = (self.rotation_axis.position - self.source.position).dot(plane_normal) / sp2.dot(plane_normal)
        p2_on_plane = self.source.position + sp2 * ratio

        self.rotation_axis.position = p1_on_plane
        self.rotation_axis.direction = p2_on_plane - p1_on_plane


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
        return self._num_pixels

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
            self._num_pixels = numpy.array(num_pixels_temp, dtype=numpy.int16)

    @property
    def pixel_size(self):
        return self._pixel_size

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

        self._pixel_size = numpy.array(pixel_size_temp)

    @property
    def origin(self):
        return self._origin

    @origin.setter
    def origin(self, val):
        allowed = ['top-left', 'top-right','bottom-left','bottom-right']
        if val in allowed:
            self._origin=val
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
        return self._num_channels

    @num_channels.setter
    def num_channels(self, val):      
        try:
            val = int(val)
        except TypeError:
            raise ValueError('num_channels expected a positive integer. Got {}'.format(type(val)))

        if val > 0:
            self._num_channels = val
        else:
            raise ValueError('num_channels expected a positive integer. Got {}'.format(val))

    @property
    def channel_labels(self):
        return self._channel_labels

    @channel_labels.setter
    def channel_labels(self, val):      
        if val is None or len(val) == self._num_channels:
            self._channel_labels = val  
        else:
            raise ValueError('labels expected to have length {0}. Got {1}'.format(self._num_channels, len(val)))

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
        return self._angle_data

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
                    self._angle_data = numpy.asarray(val, dtype=numpy.float32)
                except:
                    raise ValueError('angle_data expected to be a list of floats') 

    @property
    def initial_angle(self):
        return self._initial_angle

    @initial_angle.setter
    def initial_angle(self, val):
        try:
            val = float(val)
        except:
            raise TypeError('initial_angle expected a float. Got {0}'.format(type(val)))

        self._initial_angle = val

    @property
    def angle_unit(self):
        return self._angle_unit

    @angle_unit.setter
    def angle_unit(self,val):
        if val != AcquisitionGeometry.DEGREE and val != AcquisitionGeometry.RADIAN:
            raise ValueError('angle_unit = {} not recognised please specify \'degree\' or \'radian\''.format(val))
        else:
            self._angle_unit = val

    def __str__(self):
        repres = "Acquisition description:\n"
        repres += "\tNumber of positions: {0}\n".format(self.num_positions)
        # max_num_print = 30 
        if self.num_positions < 31:
            repres += "\tAngles 0-{0} in {1}s: {2}\n".format(self.num_positions-1, self.angle_unit, numpy.array2string(self.angle_data[0:self.num_positions], separator=', '))
        else:
            repres += "\tAngles 0-9 in {0}s: {1}\n".format(self.angle_unit, numpy.array2string(self.angle_data[0:10], separator=', '))
            repres += "\tAngles {0}-{1} in {2}s: {3}\n".format(self.num_positions-10, self.num_positions-1, self.angle_unit, numpy.array2string(self.angle_data[self.num_positions-10:self.num_positions], separator=', '))
            repres += "\tFull angular array can be accessed with acquisition_data.geometry.angles\n"
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
    r'''This class holds the description of the system components. 
     '''

    def __init__(self, units_distance='units distance'):
        self.system = None #has distances
        self.angles = None #has angles
        self.panel = None #has distances
        self.channels = Channels(1, None)
        self.units = units_distance

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

    def shift_detector_in_plane(self,
                                          pixel_offset,
                                          direction='horizontal'):
        """
        Adjusts the position of the detector in a specified direction within the imaging plane.

        Parameters:
        -----------
        pixel_offset : float
            The number of pixels to adjust the detector's position by.
        direction : {'horizontal', 'vertical'}, optional
            The direction in which to adjust the detector's position. Defaults to 'horizontal'.

        Notes:
        ------
        - If `direction` is 'horizontal':
            - If the panel's origin is 'left', positive offsets translate the detector to the right.
            - If the panel's origin is 'right', positive offsets translate the detector to the left.

        - If `direction` is 'vertical':
            - If the panel's origin is 'bottom', positive offsets translate the detector upward.
            - If the panel's origin is 'top', positive offsets translate the detector downward.

        Returns:
        --------
        None
        """

        if direction == 'horizontal':
            pixel_size = self.panel.pixel_size[0]
            pixel_direction = self.system.detector.direction_x

        elif direction == 'vertical':
            pixel_size = self.panel.pixel_size[1]
            pixel_direction = self.system.detector.direction_y

        if 'bottom' in self.panel.origin or 'left' in self.panel.origin:
            self.system.detector.position -= pixel_offset * pixel_direction * pixel_size
        else:
            self.system.detector.position += pixel_offset * pixel_direction * pixel_size


    def __str__(self):
        repres = ""
        if self.configured:
            repres += str(self.system)
            repres += str(self.panel)
            repres += str(self.channels)
            repres += str(self.angles)

            repres += "Distances in units: {}".format(self.units)
        
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
    """This class holds the AcquisitionGeometry of the system.
    
    Please initialise the AcquisitionGeometry using the using the static methods:

    `AcquisitionGeometry.create_Parallel2D()`

    `AcquisitionGeometry.create_Cone2D()`

    `AcquisitionGeometry.create_Parallel3D()`

    `AcquisitionGeometry.create_Cone3D()`
    """

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

    @property
    def dimension_labels(self):
        labels_default = DataOrder.CIL_AG_LABELS

        shape_default = [self.config.channels.num_channels,
                            self.config.angles.num_positions,
                            self.config.panel.num_pixels[1],
                            self.config.panel.num_pixels[0]
                            ]

        try:
            labels = list(self._dimension_labels)
        except AttributeError:
            labels = labels_default.copy()

        #remove from list labels where len == 1
        #
        for i, x in enumerate(shape_default):
            if x == 0 or x==1:
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
                    
            self._dimension_labels = tuple(val)

    @property
    def ndim(self):
        return len(self.dimension_labels)

    @property
    def system_description(self):
        return self.config.system.system_description()

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, val):
        self._dtype = val       


    def __init__(self):
        self._dtype = numpy.float32


    def get_centre_of_rotation(self, distance_units='default', angle_units='radian'):
        """
        Returns the system centre of rotation offset at the detector

        Note
        ----
         - Origin is in the centre of the detector
         - Axes directions are specified by detector.direction_x, detector.direction_y

        Parameters
        ----------
        distance_units : string, default='default'
            Units of distance used to calculate the return values.
            'default' uses the same units the system and panel were specified in.
            'pixels' uses pixels sizes in the horizontal and vertical directions as appropriate.
        angle_units : string
            Units to return the angle in. Can take 'radian' or 'degree'.

        Returns
        -------
        Dictionary
            {'offset': (offset, distance_units), 'angle': (angle, angle_units)}
            where,
            'offset' gives the position along the detector x_axis at y=0
            'angle' gives the angle between the y_axis and the projection of the rotation axis on the detector
        """

        if hasattr(self.config.system, 'calculate_centre_of_rotation'):
            offset_distance, angle_rad = self.config.system.calculate_centre_of_rotation()
        else:
            raise NotImplementedError

        if distance_units == 'default':
            offset = offset_distance
            offset_units = self.config.units
        elif distance_units == 'pixels':

            offset = offset_distance/ self.config.panel.pixel_size[0]
            offset_units = 'pixels'

            if self.dimension == '3D' and self.config.panel.pixel_size[0] != self.config.panel.pixel_size[1]:
                #if aspect ratio of pixels isn't 1:1 need to convert angle by new ratio
                y_pix = 1 /self.config.panel.pixel_size[1]
                x_pix = math.tan(angle_rad)/self.config.panel.pixel_size[0]
                angle_rad = math.atan2(x_pix,y_pix)
        else:
            raise ValueError("`distance_units` is not recognised. Must be 'default' or 'pixels'. Got {}".format(distance_units))

        if angle_units == 'radian':
            angle = angle_rad
            ang_units = 'radian'
        elif angle_units == 'degree':
            angle = numpy.degrees(angle_rad)
            ang_units = 'degree'
        else:
            raise ValueError("`angle_units` is not recognised. Must be 'radian' or 'degree'. Got {}".format(angle_units))

        return {'offset': (offset, offset_units), 'angle': (angle, ang_units)}


    def set_centre_of_rotation(self, offset=0.0, distance_units='default', angle=0.0, angle_units='radian'): 
        """
        Configures the system geometry to have the requested centre of rotation offset at the detector.

        Note
        ----
         - Origin is in the centre of the detector
         - Axes directions are specified by detector.direction_x, detector.direction_y

        Parameters
        ----------
        offset: float, default 0.0
            The position of the centre of rotation along the detector x_axis at y=0

        distance_units : string, default='default'
            Units the offset is specified in. Can be 'default'or 'pixels'.
            'default' interprets the input as same units the system and panel were specified in.
            'pixels' interprets the input in horizontal pixels.

        angle: float, default=0.0
            The angle between the detector y_axis and the rotation axis direction on the detector

            Notes
            -----
            If aspect ratio of pixels is not 1:1 ensure the angle is calculated from the x and y values in the correct units.

        angle_units : string, default='radian'
            Units the angle is specified in. Can take 'radian' or 'degree'.

        """

        if not hasattr(self.config.system, 'set_centre_of_rotation'):
            raise NotImplementedError()
        
        if angle_units == 'radian':
            angle_rad = angle
        elif angle_units == 'degree':
            angle_rad = numpy.radians(angle)
        else:
            raise ValueError("`angle_units` is not recognised. Must be 'radian' or 'degree'. Got {}".format(angle_units))

        if distance_units =='default':
            offset_distance = offset
        elif distance_units =='pixels':
            offset_distance = offset * self.config.panel.pixel_size[0]
        else:
            raise ValueError("`distance_units` is not recognised. Must be 'default' or 'pixels'. Got {}".format(distance_units))

        if self.dimension == '2D':
            self.config.system.set_centre_of_rotation(offset_distance)
        else:
            self.config.system.set_centre_of_rotation(offset_distance, angle_rad)


    def set_centre_of_rotation_by_slice(self, offset1, slice_index1=None, offset2=None, slice_index2=None): 
        """
        Configures the system geometry to have the requested centre of rotation offset at the detector.
        
        If two slices are passed the rotation axis will be rotated to pass through both points.

        Note
        ----
         - Offset is specified in pixels
         - Offset can be sub-pixels
         - Offset direction is specified by detector.direction_x

        Parameters
        ----------
        offset1: float
            The offset from the centre of the detector to the projected rotation position at slice_index_1
        
        slice_index1: int, optional
            The slice number of offset1

        offset2: float, optional
            The offset from the centre of the detector to the projected rotation position at slice_index_2
        
        slice_index2: int, optional
            The slice number of offset2
        """
        

        if not hasattr(self.config.system, 'set_centre_of_rotation'):
            raise NotImplementedError()
        
        if self.dimension == '2D':
            if offset2 is not None:
                logging.WARNING("Only offset1 is being used")
            self.set_centre_of_rotation(offset1)
        
        if offset2 is None or offset1 == offset2:
            offset_x_y0 = offset1
            angle = 0
        else:
            if slice_index1 is None or slice_index2 is None or slice_index1 == slice_index2:
                raise ValueError("Cannot calculate angle. Please specify `slice_index1` and `slice_index2` to define a rotated axis")

            offset_x_y0 = offset1 -slice_index1 * (offset2 - offset1)/(slice_index2-slice_index1)
            angle = math.atan2(offset2 - offset1, slice_index2 - slice_index1)

        self.set_centre_of_rotation(offset_x_y0, 'pixels', angle, 'radian')


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
    def create_Parallel2D(ray_direction=[0, 1], detector_position=[0, 0], detector_direction_x=[1, 0], rotation_axis_position=[0, 0], units='units distance'):
        r'''This creates the AcquisitionGeometry for a parallel beam 2D tomographic system

        :param ray_direction: A 2D vector describing the x-ray direction (x,y)
        :type ray_direction: list, tuple, ndarray, optional
        :param detector_position: A 2D vector describing the position of the centre of the detector (x,y)
        :type detector_position: list, tuple, ndarray, optional
        :param detector_direction_x: A 2D vector describing the direction of the detector_x (x,y)
        :type detector_direction_x: list, tuple, ndarray
        :param rotation_axis_position: A 2D vector describing the position of the axis of rotation (x,y)
        :type rotation_axis_position: list, tuple, ndarray, optional
        :param units: Label the units of distance used for the configuration, these should be consistent for the geometry and panel
        :type units: string
        :return: returns a configured AcquisitionGeometry object
        :rtype: AcquisitionGeometry
        '''
        AG = AcquisitionGeometry()
        AG.config = Configuration(units)
        AG.config.system = Parallel2D(ray_direction, detector_position, detector_direction_x, rotation_axis_position, units)
        return AG    

    @staticmethod
    def create_Cone2D(source_position, detector_position, detector_direction_x=[1,0], rotation_axis_position=[0,0], units='units distance'):
        r'''This creates the AcquisitionGeometry for a cone beam 2D tomographic system          

        :param source_position: A 2D vector describing the position of the source (x,y)
        :type source_position: list, tuple, ndarray
        :param detector_position: A 2D vector describing the position of the centre of the detector (x,y)
        :type detector_position: list, tuple, ndarray
        :param detector_direction_x: A 2D vector describing the direction of the detector_x (x,y)
        :type detector_direction_x: list, tuple, ndarray
        :param rotation_axis_position: A 2D vector describing the position of the axis of rotation (x,y)
        :type rotation_axis_position: list, tuple, ndarray, optional
        :param units: Label the units of distance used for the configuration, these should be consistent for the geometry and panel
        :type units: string
        :return: returns a configured AcquisitionGeometry object
        :rtype: AcquisitionGeometry        
     '''    
        AG = AcquisitionGeometry()
        AG.config = Configuration(units)
        AG.config.system = Cone2D(source_position, detector_position, detector_direction_x, rotation_axis_position, units)
        return AG   

    @staticmethod
    def create_Parallel3D(ray_direction=[0,1,0], detector_position=[0,0,0], detector_direction_x=[1,0,0], detector_direction_y=[0,0,1], rotation_axis_position=[0,0,0], rotation_axis_direction=[0,0,1], units='units distance'):
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
        :param units: Label the units of distance used for the configuration, these should be consistent for the geometry and panel
        :type units: string
        :return: returns a configured AcquisitionGeometry object
        :rtype: AcquisitionGeometry       
     '''
        AG = AcquisitionGeometry()
        AG.config = Configuration(units)
        AG.config.system = Parallel3D(ray_direction, detector_position, detector_direction_x, detector_direction_y, rotation_axis_position, rotation_axis_direction, units)
        return AG            

    @staticmethod
    def create_Cone3D(source_position, detector_position, detector_direction_x=[1,0,0], detector_direction_y=[0,0,1], rotation_axis_position=[0,0,0], rotation_axis_direction=[0,0,1], units='units distance'):
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
        :param units: Label the units of distance used for the configuration, these should be consistent for the geometry and panel
        :type units: string
        :return: returns a configured AcquisitionGeometry object
        :rtype: AcquisitionGeometry           
        '''
        AG = AcquisitionGeometry()
        AG.config = Configuration(units)
        AG.config.system = Cone3D(source_position, detector_position, detector_direction_x, detector_direction_y, rotation_axis_position, rotation_axis_direction, units)
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
            if geometry_new.geom_type == AcquisitionGeometry.PARALLEL or vertical == 'centre' or abs(geometry_new.pixel_num_v/2 - vertical) < 1e-6:
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
                if numpy.iscomplexobj(out.array):
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


class AcquisitionData(DataContainer, Partitioner):
    '''DataContainer for holding 2D or 3D sinogram'''
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
                 deep_copy=True, 
                 geometry = None,
                 **kwargs):

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
                self.__dict__['input'] = weakref.ref(dataset)
                self.__dict__['shouldRun'] = True
            else:
                raise ValueError('Input data not compatible')
        else:
            raise TypeError("Input type mismatch: got {0} expecting {1}" \
                            .format(type(dataset), DataContainer))
    

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
            self.__dict__['input'] =  weakref.ref(processor)
        else:
            raise TypeError("Input type mismatch: got {0} expecting {1}"\
                            .format(type(processor), DataProcessor))
        
    def get_input(self):
        '''returns the input DataContainer
        
        It is useful in the case the user has provided a DataProcessor as
        input
        '''
        if self.input() is None:
            raise ValueError("Input has been deallocated externally")
        elif issubclass(type(self.input()), DataProcessor):
            dsi = self.input().get_output()
        else:
            dsi = self.input()
        return dsi
        
    def process(self, out=None):
        raise NotImplementedError('process must be implemented')
    
    def __call__(self, x, out=None):
        
        self.set_input(x)    

        if out is None:
            out = self.get_output()      
        else:
            self.get_output(out=out)

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
            y = DataContainer(a * dsi.as_array(), True,
                              dimension_labels=dsi.dimension_labels)
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

        
        y = DataContainer(eval_func(dsi.as_array()), True,
                          dimension_labels=dsi.dimension_labels)
        return y
    

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

class DataOrder():

    ENGINES = ['astra','tigre','cil']

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
            raise ValueError("Unknown engine expected one of {0} got {1}".format(DataOrder.ENGINES, engine))
        
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



        
