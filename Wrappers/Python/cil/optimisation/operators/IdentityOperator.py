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

from cil.optimisation.operators import LinearOperator
import scipy.sparse as sp
import numpy as np


class IdentityOperator(LinearOperator):

    r''' `IdentityOperator`: :math:`\mathrm{Id}: X \rightarrow Y`,  :math:`\mathrm{Id}(x) = x`

                   :math:`X` : domain
                   :math:`Y` : range ( Default: :math:`Y = X` )

    Parameters
    ----------
    domain_geometry: CIL Geometry
        domain of the operator
    range_geometry: CIL Geometry, optional
        range of the operator, default: same as domain
    '''


    def __init__(self, domain_geometry, range_geometry=None):


        if range_geometry is None:
            range_geometry = domain_geometry

        super(IdentityOperator, self).__init__(domain_geometry=domain_geometry,
                                       range_geometry=range_geometry)

    def direct(self,x,out=None):

        r'''Returns :math:`\mathrm{Id}(x)`
        
        Parameters
        ----------
        x : DataContainer or BlockDataContainer
            Input data
        out : DataContainer or BlockDataContainer, optional
            If out is not None the output of the Operator will be filled in out, otherwise a new object is instantiated and returned. The default is None.
        
        Returns
        -------
        DataContainer or BlockDataContainer
            :math:`\mathrm{Id}(x) = x`
            
        '''

        if out is None:
            return x.copy()
        else:
            out.fill(x)
            return out

    def adjoint(self,x, out=None):
        r'''Returns :math:`\mathrm{Id}^*(x)=x`
        
        Parameters
        ----------
        x : DataContainer or BlockDataContainer
            Input data
        out : DataContainer or BlockDataContainer, optional
            If out is not None the output of the Operator will be filled in out, otherwise a new object is instantiated and returned. The default is None.
            
        Returns
        -------
        DataContainer or BlockDataContainer
            :math:`\mathrm{Id}^*(x)=x`
        
        '''


        if out is None:
            return x.copy()
        else:
            out.fill(x)
            return out

    def calculate_norm(self, **kwargs):

        '''Evaluates operator norm of  `IdentityOperator`'''

        return 1.0


    ###########################################################################
    ###############  For preconditioning ######################################
    ###########################################################################
    def matrix(self):

        return sp.eye(np.prod(self.gm_domain.shape))

    def sum_abs_row(self):

        return self.gm_range.allocate(1)

    def sum_abs_col(self):

        return self.gm_domain.allocate(1)
    
    def is_orthogonal(self):
        '''Returns if the operator is orthogonal
        
        Returns
        -------
        `Bool`
            Always returns `True` for `IdentityOperator`
        '''
        return True 
