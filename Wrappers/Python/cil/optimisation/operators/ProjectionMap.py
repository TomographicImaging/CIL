#  Copyright 2021 United Kingdom Research and Innovation
#  Copyright 2021 The University of Manchester
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

from cil.optimisation.operators import LinearOperator
from cil.framework import BlockGeometry



class ProjectionMap(LinearOperator):

    r""" Projection Map or Canonical Projection (https://en.wikipedia.org/wiki/Projection_(mathematics))

    Takes an element :math:`x = (x_{0},\dots,x_{i},\dots,x_{n})` from a Cartesian product space :math:`X_{1}\times\cdots\times X_{n}\rightarrow X_{i}`
    and projects it to element :math:`x_{i}` specified by the index :math:`i`.

    .. math:: \pi_{i}: X_{1}\times\cdots\times X_{n}\rightarrow X_{i}

    .. math:: \pi_{i}(x_{0},\dots,x_{i},\dots,x_{n}) = x_{i}

    The adjoint operation, is defined as

    .. math:: \pi_{i}^{*}(x_{i}) = (0, \cdots, x_{i}, \cdots, 0)

    Parameters
    ----------
    
    domain_geometry:`BlockGeometry`
        The domain of the `ProjectionMap`. A `BlockGeometry` is expected.

    index: int
        Index to project to the corresponding `ImageGeometry`
    
    

    """


    def __init__(self, domain_geometry, index, range_geometry=None):

        self.index = index

        if not isinstance(domain_geometry, BlockGeometry):
            raise ValueError("BlockGeometry is expected, {} is passed.".format(domain_geometry.__class__.__name__))

        if self.index > len(domain_geometry.geometries):
            raise ValueError("Index = {} is larger than the total number of geometries = {}".format(index, len(domain_geometry.geometries)))

        if range_geometry is None:
            range_geometry = domain_geometry.geometries[self.index]

        super(ProjectionMap, self).__init__(domain_geometry=domain_geometry,
                                            range_geometry=range_geometry)

    def direct(self,x,out=None):
        r"""
        Returns the ith (`param_index`) element of the Block data container, :math:`x`
        
        Parameters
        ----------
        x: `BlockDataContainer`
        
        out: `DataContainer`, default `None`
            If `out` is not `None` the output of the `ProjectionMap` will be filled in `out`, otherwise a new object is instantiated and returned.
        
        Returns
        --------
        `DataContainer`
        
        """
        
    
    
        if out is None:
            return x[self.index].copy()
        else:
            out.fill(x[self.index])

    def adjoint(self,x, out=None):
        r"""
        Returns a `BlockDataContainer` of zeros with the ith (`param_index`) filled with the `DataContainer`, :math:`x`
        
        Parameters
        ----------
        x: `DataContainer`
        
        out: `BlockDataContainer`, default `None`
            If `out` is not `None` the output of the adjoint of the `ProjectionMap` will be filled in `out`, otherwise a new object is instantiated and returned.
        
        Returns
        --------
        `BlockDataContainer`
        
        """
        
        if out is None:
            tmp = self.domain_geometry().allocate(0)
            tmp[self.index].fill(x)
            return tmp
        else:
            out*=0
            out[self.index].fill(x)
