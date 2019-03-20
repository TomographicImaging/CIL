#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 21:19:52 2019

@author: evangelos
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 12:36:40 2019

@author: ofn77899
"""

import numpy
from numbers import Number
import functools
from ccpi.framework import AcquisitionData, ImageData, BlockDataContainer, BlockGeometry
from ccpi.optimisation.operators import Operator, LinearOperator
from ccpi.optimisation.operators.BlockScaledOperator import BlockScaledOperator


       
class BlockOperatorOLD(Operator):
    '''Class to hold a block operator

    Class to hold a number of Operators in a block. 
    User may specify the shape of the block, by default is a row vector
    
    BlockOperators have a generic shape M x N, and when applied on an 
    Nx1 BlockDataContainer, will yield and Mx1 BlockDataContainer.
    Notice: BlockDatacontainer are only allowed to have the shape of N x 1, with
    N rows and 1 column.
    '''
    __array_priority__ = 1
    def __init__(self, *args, **kwargs):
        '''
        Class creator

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            vararg (Operator): Operators in the block.
            shape (:obj:`tuple`, optional): If shape is passed the Operators in 
                  vararg are considered input in a row-by-row fashion. 
                  Shape and number of Operators must match.
                  
        Example:
            BlockOperator(op0,op1) results in a row block
            BlockOperator(op0,op1,shape=(1,2)) results in a column block
        '''
        self.operators = args
        shape = kwargs.get('shape', None)
        if shape is None:
            shape = (len(args),1)
        self.shape = shape
        n_elements = functools.reduce(lambda x,y: x*y, shape, 1)
        if len(args) != n_elements:
            raise ValueError(
                    'Dimension and size do not match: expected {} got {}'
                    .format(n_elements,len(args)))
    def get_item(self, row, col):
        if row > self.shape[0]:
            raise ValueError('Requested row {} > max {}'.format(row, self.shape[0]))
        if col > self.shape[1]:
            raise ValueError('Requested col {} > max {}'.format(col, self.shape[1]))
        
        index = row*self.shape[1]+col
        return self.operators[index]
    
    def norm(self):
        norm = [op.norm() for op in self.operators]
        b = []
        for i in range(self.shape[0]):
            b.append([])
            for j in range(self.shape[1]):
                b[-1].append(norm[i*self.shape[1]+j])
        return numpy.asarray(b).sum()
    
    def direct(self, x, out=None):
        shape = self.get_output_shape(x.shape)
        res = []
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                
                if isinstance(x, ImageData):
                    get_x = x
                else:
                    get_x = x.get_item(col)
                    
                if col == 0:
                    prod = self.get_item(row,col).direct(get_x)
                else:
                    prod += self.get_item(row,col).direct(get_x)
            res.append(prod)
        return BlockDataContainer(*res, shape=shape)

    def adjoint(self, x, out=None):
        '''Adjoint operation for the BlockOperator

        BlockOperator may contain both LinearOperator and Operator
        This method exists in BlockOperator as it is not known what type of
        Operator it will contain.

        Raises: ValueError if the contained Operators are not linear
        '''
        if not functools.reduce(lambda x, y: x and y.is_linear(), self.operators, True):
            raise ValueError('Not all operators in Block are linear.')
        shape = self.get_output_shape(x.shape, adjoint=True)
        res = []
        for row in range(self.shape[1]):
            for col in range(self.shape[0]):
                
                if isinstance(x, ImageData):
                    get_x = x
                else:
                    get_x = x.get_item(col)
                
                if col == 0:
                    prod = self.get_item(row, col).adjoint(get_x)
                else:
                    prod += self.get_item(row, col).adjoint(get_x)
            res.append(prod)
        if self.shape[1]==1:
            return ImageData(*res)
        else:
            return BlockDataContainer(*res, shape=shape)
    
    def get_output_shape(self, xshape, adjoint=False):
        sshape = self.shape[1]
        oshape = self.shape[0]
        if adjoint:
            sshape = self.shape[0]
            oshape = self.shape[1]
        
        # check xshape[0]<1, to work with ImageData and not BlockDataContainer
        if sshape != xshape[0] and xshape[0]<1:
            raise ValueError('Incompatible shapes {} {}'.format(self.shape, xshape))
        return (oshape, xshape[-1])
    
    def __rmul__(self, scalar):
        '''Defines the left multiplication with a scalar

        Args: scalar (number or iterable containing numbers):

        Returns: a block operator with Scaled Operators inside'''
        if isinstance (scalar, list) or isinstance(scalar, tuple) or \
                isinstance(scalar, numpy.ndarray):
            if len(scalar) != len(self.operators):
                raise ValueError('dimensions of scalars and operators do not match')
            scalars = scalar
        else:
            scalars = [scalar for _ in self.operators]
        # create a list of ScaledOperator-s
        ops = [ v * op for v,op in zip(scalars, self.operators)]
        #return BlockScaledOperator(self, scalars ,shape=self.shape)
        return type(self)(*ops, shape=self.shape)
    
    
    def domain_geometry(self):
        
        tmp = []
        if self.shape[1]==1:
            return self.get_item(0,0).domain_geometry()
        else:
            for i in range(self.shape[1]):
                tmp.append(self.get_item(0, i).domain_geometry())
            return BlockGeometry(*tmp)

            
    def range_geometry(self):
        
        tmp = []
        for i in range(self.shape[0]):
                tmp.append(self.get_item(i,0).range_geometry())
        return BlockGeometry(*tmp)        
        
        
#        containers = [op.domain_geometry() for op in self.operators]
#        return BlockGeometry(*containers)
    
    @property
    def T(self):
        '''Return the transposed of self'''
        shape = (self.shape[1], self.shape[0])
        return type(self)(*self.operators, shape=shape)

if __name__ == '__main__':
    pass
