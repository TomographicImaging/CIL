#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy
from numbers import Number
import functools
#from ccpi.framework import AcquisitionData, ImageData
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

#    def allocate(self):
#        containers = [geom.allocate() for geom in self.geometries]
#        BlockDataContainer(*containers)
            
if __name__ == '__main__':
    
    ig1 = ImageGeometry(2,3)
    ig2 = ImageGeometry(3,4)
    
    d = BlockGeometry(ig1,ig2)

            
