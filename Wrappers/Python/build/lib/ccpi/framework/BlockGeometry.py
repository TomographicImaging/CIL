from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy
from numbers import Number
import functools
from ccpi.framework import BlockDataContainer
#from ccpi.optimisation.operators import Operator, LinearOperator
 
class BlockGeometry(object):
    '''Class to hold Geometry as column vector'''
    #__array_priority__ = 1
    def __init__(self, *args, **kwargs):
        ''''''
        self.geometries = args
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

    def allocate(self, value=0, dimension_labels=None):
        containers = [geom.allocate(value) for geom in self.geometries]
        return BlockDataContainer(*containers)
         
