# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:57:52 2019

@author: ofn77899
"""

from ccpi.optimisation.operators import Operator
from ccpi.framework import ImageGeometry
import numpy


class LinearOperator(Operator):
    '''A Linear Operator that maps from a space X <-> Y'''
    def __init__(self):
        super(LinearOperator, self).__init__()
    def is_linear(self):
        '''Returns if the operator is linear'''
        return True
    def adjoint(self,x, out=None):
        '''returns the adjoint/inverse operation
        
        only available to linear operators'''
        raise NotImplementedError
    
    @staticmethod
    def PowerMethod(operator, iterations, x0=None):
        '''Power method to calculate iteratively the Lipschitz constant'''
        # Initialise random
        
        if x0 is None:
            #x0 = op.create_image_data()
            x0 = operator.domain_geometry().allocate(ImageGeometry.RANDOM_INT)
        
        x1 = operator.domain_geometry().allocate()
        y_tmp = operator.range_geometry().allocate()
        s = numpy.zeros(iterations)
        # Loop
        for it in numpy.arange(iterations):
            #x1 = operator.adjoint(operator.direct(x0))
            operator.direct(x0,out=y_tmp)
            operator.adjoint(y_tmp,out=x1)
            x1norm = x1.norm()
            #s[it] = (x1*x0).sum() / (x0.squared_norm())
            s[it] = x1.dot(x0) / x0.squared_norm()
            #x0 = (1.0/x1norm)*x1
            #x1 *= (1.0 / x1norm)
            #x0.fill(x1)
            x1.multiply((1.0/x1norm), out=x0)
        return numpy.sqrt(s[-1]), numpy.sqrt(s), x0

    @staticmethod
    def PowerMethodNonsquare(op,numiters , x0=None):
        # Initialise random
        # Jakob's
        # inputsize , outputsize = op.size()
        #x0 = ImageContainer(numpy.random.randn(*inputsize)
        # Edo's
        #vg = ImageGeometry(voxel_num_x=inputsize[0],
        #                   voxel_num_y=inputsize[1], 
        #                   voxel_num_z=inputsize[2])
        #
        #x0 = ImageData(geometry = vg, dimension_labels=['vertical','horizontal_y','horizontal_x'])
        #print (x0)
        #x0.fill(numpy.random.randn(*x0.shape))
        
        if x0 is None:
            #x0 = op.create_image_data()
            x0 = op.allocate_direct()
            x0.fill(numpy.random.randn(*x0.shape))
        
        s = numpy.zeros(numiters)
        # Loop
        for it in numpy.arange(numiters):
            x1 = op.adjoint(op.direct(x0))
            #x1norm = numpy.sqrt((x1*x1).sum())
            x1norm = x1.norm()
            #print ("x0 **********" ,x0)
            #print ("x1 **********" ,x1)
            s[it] = (x1*x0).sum() / (x0.squared_norm())
            x0 = (1.0/x1norm)*x1
        return numpy.sqrt(s[-1]), numpy.sqrt(s), x0


