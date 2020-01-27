# -*- coding: utf-8 -*-
#  CCP in Tomographic Imaging (CCPi) Core Imaging Library (CIL).

#   Copyright 2017 UKRI-STFC
#   Copyright 2017 University of Manchester

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from numbers import Number
import numpy as np

import functools
from ccpi.framework import ImageData, BlockDataContainer, DataContainer
from ccpi.framework import BlockGeometry
try:
    from sirf import SIRF
    from sirf.SIRF import DataContainer as SIRFDataContainer
    has_sirf = True
except ImportError as ie:
    has_sirf = False

class Operator(object):
        
    '''Operator that maps from a space X -> Y'''
    
    
    def __init__(self, domain_gm=None, range_gm=None):
        
        '''
        An operator should have domain geometry and range geometry
                
        '''
                         
        self.domain_gm = domain_gm
        self.range_gm = range_gm          
        self.__norm = None
        
        
    def is_linear(self):
        '''Returns if the operator is linear'''
        return False
    
        
    def direct(self, x, out=None):
        '''Returns the application of the Operator on x'''
        raise NotImplementedError
        
                
    def norm(self, **kwargs):
        '''Returns the norm of the Operator'''
        if self.__norm is None:
            self.__norm = self.calculate_norm(**kwargs)            
        return self.__norm
    
#    def calculate_norm(self, **kwargs):
#        '''Returns the norm of the LinearOperator as calculated by the PowerMethod'''
#        x0 = kwargs.get('x0', None)
#        iterations = kwargs.get('iterations', 25)
#        s1, sall, svec = PowerMethod(self, iterations, x_init=x0)
#        return s1
    
    def calculate_norm(self, **kwargs):
        '''Calculates the norm of the Operator'''
        raise NotImplementedError
        
         
    def range_geometry(self):
        '''Returns the range of the Operator: Y space'''
        return self.range_gm
        
        
    def domain_geometry(self):
        '''Returns the domain of the Operator: X space'''
        return self.domain_gm
        
        
    # ALGEBRA for operators, use same structure as Function class
    
        # Add operators
        # Subtract operator
        # - Operator style        
        # Multiply with Scalar    
    
    def __add__(self, other):
        
        return SumOperator(self, other)
        
    def __radd__(self, other):        
        """ Making addition commutative. """
        return self + other     
    
    def __rmul__(self, scalar):
        """Returns a operator multiplied by a scalar."""               
        return ScaledOperator(self, scalar)    
    
    def __mul__(self, scalar):
        return self.__rmul__(scalar)    
    
    def __neg__(self):
        """ Return -self """
        return -1 * self    
        
    def __sub__(self, other):
        """ Returns the subtraction of the operators."""
        return self + (-1) * other   
    
    def compose(self, other):
                
        if self.operator2.range_geometry != self.operator1.domain_geometry:
            raise ValueError('Cannot compose operators, check domain geometry of {} and range geometry of {}'.format(self.operato1,self.operator2))    
        
        return CompositionOperator(self, other) 

###############################################################################
################   Linear  Operator  ###########################################
###############################################################################      


class LinearOperator(Operator):
    '''A Linear Operator that maps from a space X <-> Y'''
    
    def __init__(self, domain_gm=None, range_gm=None):
        super(LinearOperator, self).__init__(domain_gm, range_gm)
        
    def is_linear(self):
        '''Returns if the operator is linear'''
        return True
    
    def adjoint(self,x, out=None):
        '''returns the adjoint/inverse operation only available to linear operators'''
        raise NotImplementedError
    
    @staticmethod
    def PowerMethod(operator, iterations, x_init=None):
        '''Power method to calculate iteratively the Lipschitz constant
        
        :param operator: input operator
        :type operator: :code:`LinearOperator`
        :param iterations: number of iterations to run
        :type iteration: int
        :param x_init: starting point for the iteration in the operator domain
        :returns: tuple with: L, list of L at each iteration, the data the iteration worked on.
        '''
        
        # Initialise random
        if x_init is None:
            x0 = operator.domain_geometry().allocate('random')
        else:
            x0 = x_init.copy()
            
        x1 = operator.domain_geometry().allocate()
        y_tmp = operator.range_geometry().allocate()
        s = np.zeros(iterations)
        # Loop
        for it in np.arange(iterations):
            operator.direct(x0,out=y_tmp)
            operator.adjoint(y_tmp,out=x1)
            x1norm = x1.norm()
            if hasattr(x0, 'squared_norm'):
                s[it] = x1.dot(x0) / x0.squared_norm()
            else:
                x0norm = x0.norm()
                s[it] = x1.dot(x0) / (x0norm * x0norm) 
            x1.multiply((1.0/x1norm), out=x0)
        return np.sqrt(s[-1]), np.sqrt(s), x0

    def calculate_norm(self, **kwargs):
        '''Returns the norm of the LinearOperator as calculated by the PowerMethod'''
        x0 = kwargs.get('x0', None)
        iterations = kwargs.get('iterations', 25)
        s1, sall, svec = LinearOperator.PowerMethod(self, iterations, x_init=x0)
        return s1

    @staticmethod
    def dot_test(operator, domain_init=None, range_init=None, verbose=False):
        r'''Does a dot linearity test on the operator
        
        Evaluates if the following equivalence holds
        
        .. math::
        
          Ax\times y = y \times A^Tx
        
        :param operator: operator to test
        :param range_init: optional initialisation container in the operator range 
        :param domain_init: optional initialisation container in the operator domain 
        :returns: boolean, True if the test is passed.
        '''
        if range_init is None:
            y = operator.range_geometry().allocate('random_int')
        else:
            y = range_init
        if domain_init is None:
            x = operator.domain_geometry().allocate('random_int')
        else:
            x = domain_init
            
        fx = operator.direct(x)
        by = operator.adjoint(y)
        a = fx.dot(y)
        b = by.dot(x)
        if verbose:
            print ('Left hand side  {}, \nRight hand side {}'.format(a, b))
        try:
            np.testing.assert_almost_equal(abs((a-b)/a), 0, decimal=4)
            return True
        except AssertionError as ae:
            print (ae)
            return False
        
###############################################################################
################   BlockOperator Operator  ###########################################
###############################################################################      
         
    
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
            shape = (len(args),1)
        self.shape = shape
        n_elements = functools.reduce(lambda x,y: x*y, shape, 1)
        if len(args) != n_elements:
            raise ValueError(
                    'Dimension and size do not match: expected {} got {}'
                    .format(n_elements,len(args)))
        # test if operators are compatible
        if not self.column_wise_compatible():
            raise ValueError('Operators in each column must have the same domain')
        if not self.row_wise_compatible():
            raise ValueError('Operators in each row must have the same range')
    
    def column_wise_compatible(self):
        '''Operators in a Block should have the same domain per column'''
        rows, cols = self.shape
        compatible = True
        for col in range(cols):
            column_compatible = True
            for row in range(1,rows):
                dg0 = self.get_item(row-1,col).domain_geometry()
                dg1 = self.get_item(row,col).domain_geometry()
                column_compatible = dg0.__dict__ == dg1.__dict__ and column_compatible
            compatible = compatible and column_compatible
        return compatible
    
    def row_wise_compatible(self):
        '''Operators in a Block should have the same range per row'''
        rows, cols = self.shape
        compatible = True
        for row in range(rows):
            row_compatible = True
            for col in range(1,cols):
                dg0 = self.get_item(row,col-1).range_geometry()
                dg1 = self.get_item(row,col).range_geometry()
                row_compatible = dg0.__dict__ == dg1.__dict__ and row_compatible
            compatible = compatible and row_compatible
        return compatible

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
            else:
                # use Power method
                if op.is_linear():
                    norm.append(
                            LinearOperator.PowerMethod(op, 20)[0]
                            )
                else:
                    raise TypeError('Operator {} does not have a norm method and is not linear'.format(op))
        return np.sqrt(sum(norm))    
    
    def direct(self, x, out=None):
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
        
            for row in range(self.shape[0]):
                for col in range(self.shape[1]):
                    if col == 0:
                        prod = self.get_item(row,col).direct(x_b.get_item(col))
                    else:
                        prod += self.get_item(row,col).direct(x_b.get_item(col))
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
                        a = out.get_item(row)
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
        if not isinstance (x, BlockDataContainer):
            x_b = BlockDataContainer(x)
        else:
            x_b = x
        shape = self.get_output_shape(x_b.shape, adjoint=True)
        if out is None:
            res = []
            for col in range(self.shape[1]):
                for row in range(self.shape[0]):
                    if row == 0:
                        prod = self.get_item(row, col).adjoint(x_b.get_item(row))
                    else:
                        prod += self.get_item(row, col).adjoint(x_b.get_item(row))
                res.append(prod)
            if self.shape[1]==1:
                # the output is a single DataContainer, so we can take it out
                return res[0]
            else:
                return BlockDataContainer(*res, shape=shape)
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
                            op = self.get_item(row,col)
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
        
    def __rmul__(self, scalar):
        '''Defines the left multiplication with a scalar

        :paramer scalar: (number or iterable containing numbers):

        Returns: a block operator with Scaled Operators inside'''
        if isinstance (scalar, list) or isinstance(scalar, tuple) or \
                isinstance(scalar, np.ndarray):
            if len(scalar) != len(self.operators):
                raise ValueError('dimensions of scalars and operators do not match')
            scalars = scalar
        else:
            scalars = [scalar for _ in self.operators]
        # create a list of ScaledOperator-s
        ops = [ v * op for v,op in zip(scalars, self.operators)]
        #return BlockScaledOperator(self, scalars ,shape=self.shape)
        return type(self)(*ops, shape=self.shape)
    @property
    def T(self):
        '''Return the transposed of self
        
        input in a row-by-row'''
        newshape = (self.shape[1], self.shape[0])
        oplist = []
        for col in range(newshape[1]):
            for row in range(newshape[0]):
                oplist.append(self.get_item(col,row))
        return type(self)(*oplist, shape=newshape)

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
            tmp = []
            for i in range(self.shape[1]):
                tmp.append(self.get_item(0,i).domain_geometry())
            return BlockGeometry(*tmp)                
                                    
            #shape = (self.shape[0], 1)
            #return BlockGeometry(*[el.domain_geometry() for el in self.operators],
            #        shape=self.shape)

    def range_geometry(self):
        '''returns the range of the BlockOperator'''
        
        tmp = []
        for i in range(self.shape[0]):
            tmp.append(self.get_item(i,0).range_geometry())
        return BlockGeometry(*tmp)            
        
        
        #shape = (self.shape[1], 1)
        #return BlockGeometry(*[el.range_geometry() for el in self.operators],
        #            shape=shape)
        
    def sum_abs_row(self):
        
        res = []
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):                            
                if col == 0:
                    prod = self.get_item(row,col).sum_abs_row()
                else:
                    prod += self.get_item(row,col).sum_abs_row()
            res.append(prod)
            
        if self.shape[1]==1:
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

        
###############################################################################
################   Scaled Operator  ###########################################
###############################################################################      
    
class ScaledOperator(Operator):
    
    '''ScaledOperator

    A class to represent the scalar multiplication of an Operator with a scalar.
    It holds an operator and a scalar. Basically it returns the multiplication
    of the result of direct and adjoint of the operator with the scalar.
    For the rest it behaves like the operator it holds.
    
    Args:
       :param operator (Operator): a Operator or LinearOperator
       :param scalar (Number): a scalar multiplier
    
    Example:
       The scaled operator behaves like the following:

    .. code-block:: python

      sop = ScaledOperator(operator, scalar)
      sop.direct(x) = scalar * operator.direct(x)
      sop.adjoint(x) = scalar * operator.adjoint(x)
      sop.norm() = operator.norm()
      sop.range_geometry() = operator.range_geometry()
      sop.domain_geometry() = operator.domain_geometry()

    '''
    
    def __init__(self, operator, scalar):
        
        if not isinstance (scalar, Number):
            raise TypeError('expected scalar: got {}'.format(type(scalar)))   
            
        self.scalar = scalar
        self.operator = operator            
        
        super(ScaledOperator, self).__init__(self.operator.domain_geometry(),
                                             self.operator.range_geometry())

        
    def direct(self, x, out=None):
        if out is None:
            return self.scalar * self.operator.direct(x, out=out)
        else:
            self.operator.direct(x, out=out)
            out *= self.scalar
            
    def adjoint(self, x, out=None):
        
        if self.operator.is_linear():
            if out is None:
                return self.scalar * self.operator.adjoint(x, out=out)
            else:
                self.operator.adjoint(x, out=out)
                out *= self.scalar
        else:
            raise TypeError('No adjoint operation with non-linear operators')
            
    def calculate_norm(self, **kwargs):
        return np.abs(self.scalar) * self.operator.norm(**kwargs)
      
    def is_linear(self):
        return self.operator.is_linear() 
    
###############################################################################
################   SumOperator  ###########################################
###############################################################################      
    
class SumOperator(Operator):
    
    
    def __init__(self, operator1, operator2):
                
        self.operator1 = operator1
        self.operator2 = operator2
        
#        if self.operator1.domain_geometry() != self.operator2.domain_geometry():
#            raise ValueError('Domain geometry of {} is not equal with domain geometry of {}'.format(self.operator1.__class__.__name__,self.operator2.__class__.__name__))    
                
#        if self.operator1.range_geometry() != self.operator2.range_geometry():
#            raise ValueError('Range geometry of {} is not equal with range geometry of {}'.format(self.operator1.__class__.__name__,self.operator2.__class__.__name__))    
            
        self.linear_flag = self.operator1.is_linear() and self.operator2.is_linear()            
        
        super(SumOperator, self).__init__(self.operator1.domain_geometry(),
                                          self.operator1.range_geometry()) 
                                  
    def direct(self, x, out=None):
        
        if out is None:
            return self.operator1.direct(x) + self.operator2.direct(x)
        else:
            #TODO check if allcating with tmp is faster            
            self.operator1.direct(x, out=out)
            out.add(self.operator2.direct(x), out=out)     

    def adjoint(self, x, out=None):
        
        if self.linear_flag:        
            if out is None:
                return self.operator1.adjoint(x) + self.operator2.adjoint(x)
            else:
                #TODO check if allcating with tmp is faster            
                self.operator1.adjoint(x, out=out)
                out.add(self.operator2.adjoint(x), out=out) 
        else:
            raise ValueError('No adjoint operation with non-linear operators')
                                        
    def is_linear(self):
        return self.linear_flag 
    
    def calculate_norm(self, **kwargs):
        
        x0 = kwargs.get('x0', None)
        iterations = kwargs.get('iterations', 25)
        s1, sall, svec = LinearOperator.PowerMethod(self, iterations, x_init=x0)
        return s1
        
    
#        if self.linear_flag:
#        
#        return self.operator1.norm(**kwargs) + self.operator2.norm(**kwargs)   

###############################################################################
################   Composition  ###########################################
###############################################################################             
    
class CompositionOperator(Operator):
    
    def __init__(self, operator1, operator2):
        
        self.operator1 = operator1
        self.operator2 = operator2
        
        self.linear_flag = self.operator1.is_linear() and self.operator2.is_linear()        
        
        if self.operator2.range_geometry() != self.operator1.domain_geometry():
            raise ValueError('Domain geometry of {} is not equal with range geometry of {}'.format(self.operator1.__class__.__name__,self.operator2.__class__.__name__))    
                
        super(CompositionOperator, self).__init__(self.operator1.domain_geometry(),
                                          self.operator2.range_geometry()) 
        
    def direct(self, x, out = None):

        if out is None:
            return self.operator1.direct(self.operator2.direct(x))
        else:
            tmp = self.operator2.range_geometry().allocate()
            self.operator2.direct(x, out = tmp)
            self.operator1.direct(tmp, out = out)
            
    def adjoint(self, x, out = None):
        
        if self.linear_flag: 
            
            if out is None:
                return self.operator2.adjoint(self.operator1.adjoint(x))
            else:
                
                tmp = self.operator1.domain_geometry().allocate()
                self.operator1.adjoint(x, out=tmp)
                self.operator2.adjoint(tmp, out=out)
        else:
            raise ValueError('No adjoint operation with non-linear operators')
            

    def is_linear(self):
        return self.linear_flag             
            
    def calculate_norm(self, **kwargs):
        
        x0 = kwargs.get('x0', None)
        iterations = kwargs.get('iterations', 25)
        s1, sall, svec = LinearOperator.PowerMethod(self, iterations, x_init=x0)
        return s1            
                          
if __name__ == "__main__":
    
    from ccpi.optimisation.operators import Identity
    from ccpi.framework import ImageGeometry
    
    ig = ImageGeometry(1000,500)
    
    
    Id1 = Identity(ig, ig)
    Id2 = Identity(ig, ig)
    
    A = Operator()


    Id = 5*Id1 + 2*Id2 

    x = ig.allocate('random_int')

    res1 = Id.direct(x)
    res2 = 2*x
    
    print(res1.as_array(), res2.as_array())
    
    print(Id.norm())
    
    
    class my_op(Operator):
#        
        def __init__(self, domain_gm, range_gm=None):
        
            # this is only to get self.__norm = None????
            super(my_op, self).__init__()         
            
            self.domain_gm = domain_gm
            self.range_gm = range_gm          
            
            if self.range_gm is None:
                self.range_gm = self.domain_gm
                                           
        
        def direct(self, x, out=None):
            
            '''Returns Id(x)'''
            
            if out is None:
                return np.exp(x)
            else:
                out.fill(np.exp(x))   
                
#    Z = Id2 + my_op(ig, ig)
#    Z.norm()
                
                
    # Check composition opeartor
    
    ig1 = ImageGeometry(9,10)
    ig2 = ImageGeometry(23,43)
    
    bg2 = ig2.clone()
    
    ig3 = ImageGeometry(3,4)
    
    A1 = Identity(ig1, ig2)
    A2 = Identity(bg2, ig3)
    
    
    A = A2.compose(A1)
    
    print(A.norm())
    
#    B = A1.compose(A2)




    
    
    
    
    
    
    
  
    
