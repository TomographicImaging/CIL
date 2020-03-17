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

import numpy
from scipy.sparse.linalg import svds
from ccpi.framework import VectorGeometry
from ccpi.optimisation.operators import LinearOperator

class LinearOperatorMatrix(LinearOperator):
    """ Matrix wrapped into a LinearOperator
    
    :param: a numpy matrix 
    
    """
    
    def __init__(self,A):
        '''creator

        :param A: numpy ndarray representing a matrix
        '''
        self.A = A
        M_A, N_A = self.A.shape
        domain_geometry = VectorGeometry(N_A)
        range_geometry = VectorGeometry(M_A)
        self.s1 = None   # Largest singular value, initially unknown
        super(LinearOperatorMatrix, self).__init__(domain_geometry=domain_geometry,
                                                   range_geometry=range_geometry)
        
    def direct(self,x, out=None):        
                
        if out is None:
            tmp = self.range_geometry().allocate()
            tmp.fill(numpy.dot(self.A,x.as_array()))
            return tmp
        else:
            # Below use of out is not working, see
            # https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html            
            # numpy.dot(self.A, x.as_array(), out = out.as_array())            
            out.fill(numpy.dot(self.A, x.as_array()))

    def adjoint(self,x, out=None):
        if out is None:
            tmp = self.domain_geometry().allocate()
            tmp.fill(numpy.dot(self.A.transpose(),x.as_array()))
            return tmp
        else:            
            out.fill(numpy.dot(self.A.transpose(),x.as_array()))

    def size(self):
        return self.A.shape
    
    def calculate_norm(self, **kwargs):
        # If unknown, compute and store. If known, simply return it.
        return svds(self.A,1,return_singular_vectors=False)[0]
    
