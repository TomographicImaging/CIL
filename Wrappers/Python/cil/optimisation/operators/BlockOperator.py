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

import numpy
import functools
from numbers import Number
from cil.framework import ImageData, BlockDataContainer, DataContainer
from cil.optimisation.operators import Operator, LinearOperator
from cil.framework import BlockGeometry
try:
    from sirf import SIRF
    from sirf.SIRF import DataContainer as SIRFDataContainer
    has_sirf = True
except ImportError as ie:
    has_sirf = False


class BlockOperator(Operator):
    r'''A Block matrix containing Operators

    Parameters
    ----------
    *args : Operator  
        Operators in the block.  
    **kwargs : dict  
        shape (:obj:`tuple`, optional): If shape is passed the Operators in vararg are considered input in a row-by-row fashion.  


    Note
    ----
    The Block Framework is a generic strategy to treat variational problems in the
    following form:

    .. math::

      \min Regulariser + Fidelity


    BlockOperators have a generic shape M x N, and when applied on an 
    Nx1 BlockDataContainer, will yield and Mx1 BlockDataContainer.
  
    Note
    -----
    BlockDatacontainer are only allowed to have the shape of N x 1, with
    N rows and 1 column.

    User may specify the shape of the block, by default is a row vector

    Operators in a Block are required to have the same domain column-wise and the
    same range row-wise.

    Examples
    -------

    BlockOperator(op0,op1) results in a row block

    BlockOperator(op0,op1,shape=(1,2)) results in a column block


    '''
    __array_priority__ = 1

    def __init__(self, *args, **kwargs):

        self.operators = args
        shape = kwargs.get('shape', None)
        if shape is None:
            shape = (len(args), 1)
        self.shape = shape
        n_elements = functools.reduce(lambda x, y: x*y, shape, 1)
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
        compatible = True
        for col in range(cols):
            column_compatible = True
            for row in range(1, rows):
                dg0 = self.get_item(row-1, col).domain_geometry()
                dg1 = self.get_item(row, col).domain_geometry()
                if hasattr(dg0, 'handle') and hasattr(dg1, 'handle'):
                    column_compatible = True and column_compatible
                else:
                    column_compatible = dg0.__dict__ == dg1.__dict__ and column_compatible
            compatible = compatible and column_compatible
        return compatible

    def row_wise_compatible(self):
        '''Operators in a Block should have the same range per row'''
        rows, cols = self.shape
        compatible = True
        for row in range(rows):
            row_compatible = True
            for col in range(1, cols):
                dg0 = self.get_item(row, col-1).range_geometry()
                dg1 = self.get_item(row, col).range_geometry()
                if hasattr(dg0, 'handle') and hasattr(dg1, 'handle'):
                    row_compatible = True and column_compatible
                else:
                    row_compatible = dg0.__dict__ == dg1.__dict__ and row_compatible

            compatible = compatible and row_compatible

        return compatible

    def get_item(self, row, col):
        '''Returns the Operator at specified row and col
        Parameters
        ----------
        row: `int`
            The row index required. 
        col: `int`
            The column index required. 
        '''
        if row > self.shape[0]:
            raise ValueError(
                'Requested row {} > max {}'.format(row, self.shape[0]))
        if col > self.shape[1]:
            raise ValueError(
                'Requested col {} > max {}'.format(col, self.shape[1]))

        index = row*self.shape[1]+col
        return self.operators[index]

    def norm(self):
        '''Returns the Euclidean norm of the norms of the individual operators in the BlockOperators '''
        return numpy.sqrt(numpy.sum(numpy.array(self.get_norms_as_list())**2))

    def get_norms_as_list(self, ):
        '''Returns a list of the individual norms of the Operators in the BlockOperator
        '''
        return [op.norm() for op in self.operators]

    def set_norms(self, norms):
        '''Uses the set_norm() function in Operator to set the norms of the operators in the BlockOperator from a list of custom values. 

        Parameters  
        ------------  
        norms: list  
            A list of positive real values the same length as the number of operators in the BlockOperator.  

        '''
        if len(norms) != self.size:
            raise ValueError(
                "The length of the list of norms should be equal to the number of operators in the BlockOperator")

        for j, value in enumerate(norms):
            self.operators[j].set_norm(value)

    def direct(self, x, out=None):
        '''Direct operation for the BlockOperator

        Note
        -----
        BlockOperators work on BlockDataContainers, but they will also work on DataContainers
        and inherited classes by simple wrapping the input in a BlockDataContainer of shape (1,1)
        '''

        if not isinstance(x, BlockDataContainer):
            x_b = BlockDataContainer(x)
        else:
            x_b = x
        shape = self.get_output_shape(x_b.shape)
        res = []

        if out is None:

            for row in range(self.shape[0]):
                for col in range(self.shape[1]):
                    if col == 0:
                        prod = self.get_item(row, col).direct(
                            x_b.get_item(col))
                    else:
                        prod += self.get_item(row,
                                              col).direct(x_b.get_item(col))
                res.append(prod)
            return BlockDataContainer(*res, shape=shape)

        else:

            tmp = self.range_geometry().allocate()
            for row in range(self.shape[0]):
                for col in range(self.shape[1]):  
                    if col == 0:       
                        self.get_item(row,col).direct(
                                                      x_b.get_item(col),
                                                      out=out.get_item(row))                        
                    else:
                        temp_out_row = out.get_item(row) # temp_out_row points to the element in out that we are adding to  
                        self.get_item(row,col).direct(
                                                      x_b.get_item(col), 
                                                      out=tmp.get_item(row))
                        temp_out_row += tmp.get_item(row)
                

    def adjoint(self, x, out=None):
        '''Adjoint operation for the BlockOperator

        Note
        -----
        BlockOperator may contain both LinearOperator and Operator
        This method exists in BlockOperator as it is not known what type of
        Operator it will contain.

        BlockOperators work on BlockDataContainers, but they will also work on DataContainers
        and inherited classes by simple wrapping the input in a BlockDataContainer of shape (1,1)

        Raises: ValueError if the contained Operators are not linear
        '''
        if not self.is_linear():
            raise ValueError('Not all operators in Block are linear.')
        if not isinstance(x, BlockDataContainer):
            x_b = BlockDataContainer(x)
        else:
            x_b = x
        shape = self.get_output_shape(x_b.shape, adjoint=True)
        if out is None:
            res = []
            for col in range(self.shape[1]):
                for row in range(self.shape[0]):
                    if row == 0:
                        prod = self.get_item(row, col).adjoint(
                            x_b.get_item(row))
                    else:
                        prod += self.get_item(row,
                                              col).adjoint(x_b.get_item(row))
                res.append(prod)
            if self.shape[1] == 1:
                # the output is a single DataContainer, so we can take it out
                return res[0]
            else:
                return BlockDataContainer(*res, shape=shape)
        else:

            for col in range(self.shape[1]):
                for row in range(self.shape[0]):
                    if row == 0:
                        if issubclass(out.__class__, DataContainer) or \
                                (has_sirf and issubclass(out.__class__, SIRFDataContainer)):
                            self.get_item(row, col).adjoint(
                                x_b.get_item(row),
                                out=out)
                        else:
                            op = self.get_item(row, col)
                            self.get_item(row, col).adjoint(
                                x_b.get_item(row),
                                out=out.get_item(col))
                    else:
                        if issubclass(out.__class__, DataContainer) or \
                                (has_sirf and issubclass(out.__class__, SIRFDataContainer)):
                            out += self.get_item(row, col).adjoint(
                                x_b.get_item(row))
                        else:

                            temp_out_col = out.get_item(col) # out_col_operator points to the column in out that we are updating 
                            temp_out_col += self.get_item(row,col).adjoint(
                                                        x_b.get_item(row),
                                                        )

    def is_linear(self):
        '''Returns whether all the elements of the BlockOperator are linear'''
        return functools.reduce(lambda x, y: x and y.is_linear(), self.operators, True)

    def get_output_shape(self, xshape, adjoint=False):
        '''Returns the shape of the output BlockDataContainer
        Parameters
        ----------
        xshape: BlockDataContainer

        adjoint: `bool`

        Examples
        --------
        A(N,M) direct u(M,1) -> N,1
        
        A(N,M)^T adjoint u(N,1) -> M,1
        '''
        rows, cols = self.shape
        xrows, xcols = xshape
        if xcols != 1:
            raise ValueError(
                'BlockDataContainer cannot have more than 1 column')
        if adjoint:
            if rows != xrows:
                raise ValueError(
                    'Incompatible shapes {} {}'.format(self.shape, xshape))
            return (cols, xcols)
        if cols != xrows:
            raise ValueError(
                'Incompatible shapes {} {}'.format((rows, cols), xshape))
        return (rows, xcols)

    def __rmul__(self, scalar):
        '''Defines the left multiplication with a scalar. Returns a block operator with Scaled Operators inside.

        Parameters
        ------------

        scalar: number or iterable containing numbers

        '''
        if isinstance(scalar, list) or isinstance(scalar, tuple) or \
                isinstance(scalar, numpy.ndarray):
            if len(scalar) != len(self.operators):
                raise ValueError(
                    'dimensions of scalars and operators do not match')
            scalars = scalar
        else:
            scalars = [scalar for _ in self.operators]
        # create a list of ScaledOperator-s
        ops = [v * op for v, op in zip(scalars, self.operators)]
        # return BlockScaledOperator(self, scalars ,shape=self.shape)
        return type(self)(*ops, shape=self.shape)

    @property
    def T(self):
        '''Returns the transposed of self.

        Recall the input list is shaped in a row-by-row fashion'''
        newshape = (self.shape[1], self.shape[0])
        oplist = []
        for col in range(newshape[1]):
            for row in range(newshape[0]):
                oplist.append(self.get_item(col, row))
        return type(self)(*oplist, shape=newshape)

    def domain_geometry(self):
        '''Returns the domain of the BlockOperator

        If the shape of the BlockOperator is (N,1) the domain is a ImageGeometry or AcquisitionGeometry.
        Otherwise it is a BlockGeometry.
        '''
        if self.shape[1] == 1:
            # column BlockOperator
            return self.get_item(0, 0).domain_geometry()
        else:
            # get the geometries column wise
            # we need only the geometries from the first row
            # since it is compatible from __init__
            tmp = []
            for i in range(self.shape[1]):
                tmp.append(self.get_item(0, i).domain_geometry())
            if self.shape[1] == 1:
                return tmp[0]
            return BlockGeometry(*tmp)

            # shape = (self.shape[0], 1)
            # return BlockGeometry(*[el.domain_geometry() for el in self.operators],
            #        shape=self.shape)

    def range_geometry(self):
        '''Returns the range of the BlockOperator'''

        tmp = []
        for i in range(self.shape[0]):
            tmp.append(self.get_item(i, 0).range_geometry())
        if self.shape[0] == 1:
            return tmp[0]
        return BlockGeometry(*tmp)

    def sum_abs_row(self):

        res = []
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                if col == 0:
                    prod = self.get_item(row, col).sum_abs_row()
                else:
                    prod += self.get_item(row, col).sum_abs_row()
            res.append(prod)

        if self.shape[1] == 1:
            tmp = sum(res)
            return ImageData(tmp)
        else:

            return BlockDataContainer(*res)

    def sum_abs_col(self):

        res = []
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                if col == 0:
                    prod = self.get_item(row, col).sum_abs_col()
                else:
                    prod += self.get_item(row, col).sum_abs_col()
            res.append(prod)

        return BlockDataContainer(*res)

    def __len__(self):
        return len(self.operators)

    @property
    def size(self):
        return len(self.operators)

    def __getitem__(self, index):
        '''Returns the index-th operator in the block irrespectively of it's shape'''
        return self.operators[index]

    def get_as_list(self):
        '''Returns the list of operators'''
        return self.operators
