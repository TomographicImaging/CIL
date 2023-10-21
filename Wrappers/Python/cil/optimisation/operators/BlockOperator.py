# -*- coding: utf-8 -*-
#  Copyright 2019 United Kingdom Research and Innovation
#  Copyright 2019 The University of Manchester
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

import functools
import itertools
import operator

import numpy
from cil.framework import (BlockDataContainer, BlockGeometry, DataContainer,
                           ImageData)
from cil.optimisation.operators import LinearOperator, Operator

try:
    from sirf import SIRF
    from sirf.SIRF import DataContainer as SIRFDataContainer
    has_sirf = True
except ImportError:
    has_sirf = False


def rsum(seq):
    return functools.reduce(operator.add, seq)


class BlockOperator(Operator):
    r'''A Block matrix containing Operators

    The Block Framework is a generic strategy to treat variational problems in the
    following form:

    .. math::

      \min Regulariser + Fidelity


    BlockOperators have a generic shape M x N, and when applied on an
    Nx1 BlockDataContainer, will yield and Mx1 BlockDataContainer.
    Notice: BlockDatacontainer are only allowed to have the shape of N x 1, with
    N rows and 1 column.

    User may specify the shape of the block, by default is a row vector

    Operators in a Block are required to have the same domain column-wise and the
    same range row-wise.
    '''
    __array_priority__ = 1

    def __init__(self, *args, **kwargs):
        '''
        Class creator

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            :param: vararg (Operator): Operators in the block.
            :param: shape (:obj:`tuple`, optional): If shape is passed the Operators in
                  vararg are considered input in a row-by-row fashion.
                  Shape and number of Operators must match.

        Example:
            BlockOperator(op0,op1) results in a row block
            BlockOperator(op0,op1,shape=(1,2)) results in a column block
        '''
        self.operators = args
        shape = kwargs.get('shape', None)
        if shape is None:
            shape = (len(args), 1)
        self.shape = shape
        n_elements = functools.reduce(operator.mul, shape, 1)
        if len(args) != n_elements:
            raise ValueError(
                    'Dimension and size do not match: expected {} got {}'
                    .format(n_elements, len(args)))
        # TODO
        # until a decent way to check equality of Acquisition/Image geometries
        # required to fullfil "Operators in a Block are required to have the same
        # domain column-wise and the same range row-wise."
        # let us just not check if column/row-wise compatible, which is actually
        # the same achieved by the column_wise_compatible and row_wise_compatible methods.

        # # test if operators are compatible
        # if not self.column_wise_compatible():
        #     raise ValueError('Operators in each column must have the same domain')
        # if not self.row_wise_compatible():
        #     raise ValueError('Operators in each row must have the same range')

    def column_wise_compatible(self):
        '''Operators in a Block should have the same domain per column'''
        rows, cols = self.shape
        for col in range(cols):
            dgs = (self.get_item(row, col).domain_geometry() for row in range(rows))
            for dg0, dg1 in itertools.pairwise(dgs):
                if not (hasattr(dg0, 'handle') and hasattr(dg1, 'handle')):
                    if dg0.__dict__ != dg1.__dict__:
                        return False
        return True

    def row_wise_compatible(self):
        '''Operators in a Block should have the same range per row'''
        rows, cols = self.shape
        for row in range(rows):
            dgs = (self.get_item(row, col).range_geometry() for col in range(cols))
            for dg0, dg1 in itertools.pairwise(dgs):
                if not (hasattr(dg0,'handle') and hasattr(dg1,'handle')):
                    if dg0.__dict__ != dg1.__dict__:
                        return False
        return True

    def get_item(self, row, col):
        '''returns the Operator at specified row and col'''
        if row > self.shape[0]:
            raise ValueError('Requested row {} > max {}'.format(row, self.shape[0]))
        if col > self.shape[1]:
            raise ValueError('Requested col {} > max {}'.format(col, self.shape[1]))

        index = row*self.shape[1]+col
        return self.operators[index]

    def norm(self, **kwargs):
        '''Returns the norm of the BlockOperator

        if the operator in the block do not have method norm defined, i.e. they are SIRF
        AcquisitionModel's we use PowerMethod if applicable, otherwise we raise an Error
        '''
        norm = []
        for op in self.operators:
            if hasattr(op, 'norm'):
                norm.append(op.norm(**kwargs) ** 2.)
            elif op.is_linear():
                # use Power method
                norm.append(LinearOperator.PowerMethod(op, 20)[0])
            else:
                raise TypeError('Operator {} does not have a norm method and is not linear'.format(op))
        return numpy.sqrt(sum(norm))

    def direct(self, x, out=None):
        '''Direct operation for the BlockOperator

        BlockOperator work on BlockDataContainer, but they will work on DataContainers
        and inherited classes by simple wrapping the input in a BlockDataContainer of shape (1,1)
        '''
        x_b = x if isinstance(x, BlockDataContainer) else BlockDataContainer(x)
        shape = self.get_output_shape(x_b.shape)

        if out is None:
            res = (rsum(self.get_item(row, col).direct(x_b.get_item(col))
                        for col in range(self.shape[1]))
                   for row in range(self.shape[0]))
            return BlockDataContainer(*res, shape=shape)
        else:
            tmp = self.range_geometry().allocate()
            for row in range(self.shape[0]):
                a = out.get_item(row)
                for col in range(self.shape[1]):
                    if col == 0:
                        self.get_item(row,col).direct(
                                                      x_b.get_item(col),
                                                      out=a)
                    else:
                        self.get_item(row,col).direct(
                                                      x_b.get_item(col),
                                                      out=tmp.get_item(row))
                        a += tmp.get_item(row)

    def adjoint(self, x, out=None):
        '''Adjoint operation for the BlockOperator

        BlockOperator may contain both LinearOperator and Operator
        This method exists in BlockOperator as it is not known what type of
        Operator it will contain.

        BlockOperator work on BlockDataContainer, but they will work on DataContainers
        and inherited classes by simple wrapping the input in a BlockDataContainer of shape (1,1)

        Raises: ValueError if the contained Operators are not linear
        '''
        if not self.is_linear():
            raise ValueError('Not all operators in Block are linear.')
        x_b = x if isinstance(x, BlockDataContainer) else BlockDataContainer(x)
        shape = self.get_output_shape(x_b.shape, adjoint=True)
        if out is None:
            res = (rsum(self.get_item(row, col).adjoint(x_b.get_item(row))
                        for row in range(self.shape[0]))
                   for col in range(self.shape[1]))
            return next(res) if self.shape[1] == 1 else BlockDataContainer(*res, shape=shape)
        else:
            for col in range(self.shape[1]):
                for row in range(self.shape[0]):
                    if row == 0:
                        if issubclass(out.__class__, DataContainer) or \
                ( has_sirf and issubclass(out.__class__, SIRFDataContainer) ):
                            self.get_item(row, col).adjoint(
                                                x_b.get_item(row),
                                                out=out)
                        else:
                            self.get_item(row, col).adjoint(
                                                x_b.get_item(row),
                                                out=out.get_item(col))
                    else:
                        if issubclass(out.__class__, DataContainer) or \
                ( has_sirf and issubclass(out.__class__, SIRFDataContainer) ):
                            out += self.get_item(row,col).adjoint(
                                                        x_b.get_item(row))
                        else:
                            a = out.get_item(col)
                            a += self.get_item(row,col).adjoint(
                                                        x_b.get_item(row),
                                                        )

    def is_linear(self):
        '''returns whether all the elements of the BlockOperator are linear'''
        return functools.reduce(lambda x, y: x and y.is_linear(), self.operators, True)

    def get_output_shape(self, xshape, adjoint=False):
        '''returns the shape of the output BlockDataContainer

        A(N,M) direct u(M,1) -> N,1
        A(N,M)^T adjoint u(N,1) -> M,1
        '''
        rows, cols = self.shape
        xrows, xcols = xshape
        if xcols != 1:
            raise ValueError('BlockDataContainer cannot have more than 1 column')
        if adjoint:
            if rows != xrows:
                raise ValueError('Incompatible shapes {} {}'.format(self.shape, xshape))
            return (cols, xcols)
        if cols != xrows:
            raise ValueError('Incompatible shapes {} {}'.format((rows,cols), xshape))
        return (rows, xcols)

    def __rmul__(self, scalar):
        '''Defines the left multiplication with a scalar

        :paramer scalar: (number or iterable containing numbers):

        Returns: a block operator with Scaled Operators inside'''
        if isinstance(scalar, list) or isinstance(scalar, tuple) or \
                isinstance(scalar, numpy.ndarray):
            if len(scalar) != len(self.operators):
                raise ValueError('dimensions of scalars and operators do not match')
            scalars = scalar
        else:
            scalars = [scalar] * len(self.operators)
        # create a list of ScaledOperator-s
        ops = (v * op for v, op in zip(scalars, self.operators))
        #return BlockScaledOperator(self, scalars ,shape=self.shape)
        return type(self)(*ops, shape=self.shape)

    @property
    def T(self):
        '''Return the transposed of self

        input in a row-by-row'''
        newshape = (self.shape[1], self.shape[0])
        ops = (self.get_item(col,row)
               for col, row in itertools.product(range(newshape[1]), range(newshape[0])))
        return type(self)(*ops, shape=newshape)

    def domain_geometry(self):
        '''returns the domain of the BlockOperator

        If the shape of the BlockOperator is (N,1) the domain is a ImageGeometry or AcquisitionGeometry.
        Otherwise it is a BlockGeometry.
        '''
        if self.shape[1] == 1:
            # column BlockOperator
            return self.get_item(0,0).domain_geometry()
        else:
            # get the geometries column wise
            # we need only the geometries from the first row
            # since it is compatible from __init__
            tmp = (self.get_item(0,i).domain_geometry() for i in range(self.shape[1]))
            return BlockGeometry(*tmp)

            #shape = (self.shape[0], 1)
            #return BlockGeometry(*[el.domain_geometry() for el in self.operators],
            #        shape=self.shape)

    def range_geometry(self):
        '''returns the range of the BlockOperator'''
        tmp = (self.get_item(i, 0).range_geometry() for i in range(self.shape[0]))
        return BlockGeometry(*tmp)

        #shape = (self.shape[1], 1)
        #return BlockGeometry(*[el.range_geometry() for el in self.operators],
        #            shape=shape)

    def sum_abs_row(self):
        res = (rsum(self.get_item(row, col).sum_abs_row() for col in range(self.shape[1]))
               for row in range(self.shape[0]))
        return ImageData(rsum(res)) if self.shape[1] == 1 else BlockDataContainer(*res)

    def sum_abs_col(self):
        res = (rsum(self.get_item(row, col).sum_abs_col() for col in range(self.shape[1]))
               for row in range(self.shape[0]))
        return BlockDataContainer(*res)

    def __len__(self):
        return len(self.operators)

    def __getitem__(self, index):
        '''returns the index-th operator in the block irrespectively of it's shape'''
        return self.operators[index]

    def get_as_list(self):
        '''returns the list of operators'''
        return self.operators
