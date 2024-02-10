import copy
import math

import numpy

from .label import acquisition_labels


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
        if val != acquisition_labels["CONE"] and val != acquisition_labels["PARALLEL"]:
            raise ValueError('geom_type = {} not recognised please specify \'cone\' or \'parallel\''.format(val))
        else:
            self._geometry = val

    def __init__(self, dof, geometry, units='units'):
        """Initialises the system component attributes for the acquisition type
        """
        self.dimension = dof
        self.geometry = geometry
        self.units = units

        if geometry == acquisition_labels["PARALLEL"]:
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
