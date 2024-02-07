import numpy


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
