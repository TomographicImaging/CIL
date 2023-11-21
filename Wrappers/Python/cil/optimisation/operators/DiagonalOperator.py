# -*- coding: utf-8 -*-
#  Copyright 2020 United Kingdom Research and Innovation
#  Copyright 2020 The University of Manchester
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

import numpy as np
from cil.framework import ImageData, DataContainer
from cil.optimisation.operators import Operator, BlockOperator,  LinearOperator
from cil.framework import BlockDataContainer
try:
    from sirf import SIRF
    from sirf.SIRF import DataContainer as SIRFDataContainer
    has_sirf = True
except ImportError as ie:
    has_sirf = False
from cil.framework import BlockGeometry
class DiagonalOperator(LinearOperator):

    r"""DiagonalOperator 

    Performs an element-wise multiplication, i.e., `Hadamard Product <https://en.wikipedia.org/wiki/Hadamard_product_(matrices)#:~:text=In%20mathematics%2C%20the%20Hadamard%20product,elements%20i%2C%20j%20of%20the>`_ 
    of a :class:`DataContainer` `x` and :class:`DataContainer` `diagonal`, `d` .

    .. math:: (D\circ x) = \sum_{i,j}^{M,N} D_{i,j} x_{i, j}

    In matrix-vector interpretation, if `D` is a :math:`M\times N` dense matrix and is flattened, we have a :math:`M*N \times M*N` vector.
    A sparse diagonal matrix, i.e., :class:`DigaonalOperator` can be created if we add the vector above to the main diagonal.
    If the :class:`DataContainer` `x` is also flattened, we have a :math:`M*N` vector.
    Now, matrix-vector multiplcation is allowed and results to a :math:`(M*N,1)` vector. After reshaping we recover a :math:`M\times N` :class:`DataContainer`. 

    Parameters
    ----------
    diagonal : DataContainer
        DataContainer with the same dimensions as the data to be operated on.
    domain_geometry : ImageGeometry
        Specifies the geometry of the operator domain. If 'None' will use the diagonal geometry directly. default=None .

    """

    
    def __init__(self, diagonal, domain_geometry=None):

        if domain_geometry is None:
            domain_geometry = diagonal.geometry.copy()

        super(DiagonalOperator, self).__init__(domain_geometry=domain_geometry, 
                                    range_geometry=domain_geometry)
        self.diagonal = diagonal

        
    def direct(self,x,out=None):
        
        "Returns :math:`D\circ x` "
        
        if out is None:
            return self.diagonal * x
        else:
            self.diagonal.multiply(x,out=out)
    

    def adjoint(self,x, out=None):
        
        "Returns :math:`D\circ x` "
        
        return self.direct(x, out=out) 

  
    def calculate_norm(self, **kwargs):
        
        r""" Returns the operator norm of DiagonalOperator which is the :math:`\infty` norm of `diagonal`
        
        .. math:: \|D\|_{\infty} = \max_{i}\{|D_{i}|\}

        """

        return self.diagonal.abs().max()





class BlockDiagonalOperator(Operator):

    r"""DiagonalOperator 

    Constructs a square diagonal operator. The main diagonal must be provided but then any other diagonals can also be included. 

    Parameters
    ----------
    diagonals : Lists of operators (if k=0) or a list of lists of operators if len(k)>0 
    domain_geometry : ImageGeometry
        Specifies the geometry of the BlockOperator domain. If 'None' will use the diagonal geometries directly. default=None .
    range_geometry: 
        Specifies the geometry of the BlockOperator domain. If 'None' will use the diagonal geometries directly. default=None .
    k: Either k=0 (default) or k is a list of diagonals to fill whose first entry should be 0. 
         Use k>0 for diagonals above the main diagonal, and k<0 for diagonals below the main diagonal.
    
    Examples
    --------
    
    """

    
    def __init__(self, diagonals, k=0, domain_geometry=None, range_geometry=None):
        if k==0:
            k=[0] 
            try:
                diagonals[0][0]
            except:
                diagonals=[diagonals]
        else:
            if k[0]!=0:
                raise TypeError("The main diagonal must always be provided first")
        self.shape=(len(diagonals[0]),len(diagonals[0])) #TODO: deal with the non-square case
        self.diagonals = diagonals
        self.k=k
        
        if domain_geometry is None:
            domain_geometry = self.get_domain_geometry()
        
        if range_geometry is None: 
            range_geometry = self.get_range_geometry()
        super(BlockDiagonalOperator, self).__init__(domain_geometry=domain_geometry, 
                                    range_geometry=domain_geometry)
        

    def get_output_shape(self, xshape, adjoint=False):
        '''returns the shape of the output BlockDataContainer
        
        A(N,M) direct u(M,1) -> N,1
        A(N,M)^T adjoint u(N,1) -> M,1
        '''
        rows , cols = self.shape
        xrows, xcols = xshape
        if xcols != 1:
            raise ValueError('BlockDataContainer cannot have more than 1 column')
        if adjoint:
            if rows != xrows:
                raise ValueError('Incompatible shapes {} {}'.format(self.shape, xshape))
            return (cols,xcols)
        if cols != xrows:
            raise ValueError('Incompatible shapes {} {}'.format((rows,cols), xshape))
        return (rows,xcols)
    
    
    
    def direct(self,x,out=None):
        
        '''Direct operation for the BlockOperator

        BlockOperator work on BlockDataContainer, but they will work on DataContainers
        and inherited classes by simple wrapping the input in a BlockDataContainer of shape (1,1)
        '''
        if not isinstance (x, BlockDataContainer):
            x_b = BlockDataContainer(x)
        else:
            x_b = x
        shape = self.get_output_shape(x_b.shape)
        res = []
        
        if out is None:
        
            for  row in range(self.shape[0]):
                for j, offset in enumerate(self.k):
                    col=row+offset
                    if 0<=col<self.shape[0]:
                        if j==0:
                            prod = self.diagonals[j][min(row,col)].direct(x_b.get_item(col))
                        else:
                            prod+= self.diagonals[j][min(row,col)].direct(x_b.get_item(col))
                res.append(prod)
            return BlockDataContainer(*res, shape=shape)
                
        else:
            
            tmp = self.range_geometry().allocate()
            for row in range(self.shape[0]):
                for j, offset in enumerate(self.k):
                    col=row+offset
                    if 0<=col<self.shape[0]:
                        if j == 0:       
                            self.diagonals[j][min(row,col)].direct(
                                                        x_b.get_item(col),
                                                        out=out.get_item(row))                        
                        else:
                            a = out.get_item(row) #TODO: change a! 
                            self.diagonals[j][min(row,col)].direct(
                                                        x_b.get_item(col), 
                                                        out=tmp.get_item(row))
                            a += tmp.get_item(row)
        
        

    def adjoint(self,x, out=None):
        
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
        if not isinstance (x, BlockDataContainer):
            x_b = BlockDataContainer(x)
        else:
            x_b = x
        shape = self.get_output_shape(x_b.shape, adjoint=True)
        if out is None:
            res = []
            for col in range(self.shape[1]):
                for j, offset in enumerate(self.k):
                    row=col-offset
                    if 0<=row<self.shape[0]:
                        if j == 0:
                            prod = self.diagonals[j][min(row,col)].adjoint(x_b.get_item(row))
                        else:
                            prod += self.diagonals[j][min(row,col)].adjoint(x_b.get_item(row))
                res.append(prod)
            if self.shape[1]==1:
                # the output is a single DataContainer, so we can take it out
                return res[0]
            else:
                return BlockDataContainer(*res, shape=shape)
        else:

            for col in range(self.shape[1]):
                for j, offset in enumerate(self.k):
                    row=col-offset
                    if 0<=row<self.shape[0]:
                        if j==0:
                            if issubclass(out.__class__, DataContainer) or \
                    ( has_sirf and issubclass(out.__class__, SIRFDataContainer) ):
                                self.diagonals[j][min(row,col)].adjoint(
                                                    x_b.get_item(row),
                                                    out=out)
                            else:
                                self.diagonals[j][min(row,col)].adjoint(
                                                    x_b.get_item(row),
                                                    out=out.get_item(col))
                        else:
                            if issubclass(out.__class__, DataContainer) or \
                    ( has_sirf and issubclass(out.__class__, SIRFDataContainer) ):
                                out += self.diagonals[j][min(row,col)].adjoint(
                                                            x_b.get_item(row))
                            else:
                                a = out.get_item(col) #TODO: get rid of a 
                                a += self.diagonals[j][min(row,col)].adjoint(
                                                            x_b.get_item(row),
                                                            )
    @property
    def T(self):
        '''Return the transposed of self TODO:''' 
        new_k=[]
        for i in range(len(self.k)):
            new_k.append(-self.k[i])
            
        return type(self)(k=new_k, range_geometry=self.range_geometry, domain_geometry=self.domain_geometry, diagonals=self.diagonals)

  
    def calculate_norm(self, **kwargs):
        
        r""" TODO:

        """
        hold=[]
        if self.k==[[0]]:
            for i in range(len(self.diagonals[0])):
                hold.append(self.diagonals[0][i].norm)
            return hold.max()
        else:
            raise NotImplementedError
    
    
    def get_domain_geometry(self):
        '''returns the domain of the BlockOperator

        If the shape of the BlockOperator is (N,1) the domain is a ImageGeometry or AcquisitionGeometry.
        Otherwise it is a BlockGeometry.
        '''
        if self.shape[1] == 1:
            # column BlockOperator
            return self.diagonals[0].get_item(0).domain_geometry()
        else:
            # get the geometries column wise
            # we need only the geometries from the first row
            # since it is compatible from __init__
            tmp = []
            for i in range(self.shape[1]):
                tmp.append(self.diagonals[0][i].domain_geometry())
            return BlockGeometry(*tmp)                
                                    


    def get_range_geometry(self):
        '''returns the range of the BlockOperator'''
        
        tmp = []
        for i in range(self.shape[0]):
            tmp.append(self.diagonals[0][i].range_geometry())
        return BlockGeometry(*tmp)            
        
     def is_linear(self):
        '''returns whether all the elements of the BlockOperator are linear'''
        return functools.reduce(lambda x, y: x and y.is_linear(), self.operators, True)
