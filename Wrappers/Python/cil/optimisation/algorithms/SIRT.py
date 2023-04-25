# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library (CIL) developed by CCPi 
#   (Collaborative Computational Project in Tomographic Imaging), with 
#   substantial contributions by UKRI-STFC and University of Manchester.

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
from cil.optimisation.algorithms import Algorithm
from cil.optimisation.functions import IndicatorBox
from cil.framework import BlockDataContainer
from numpy import inf
import numpy
import logging

class SIRT(Algorithm):    

    r"""Simultaneous Iterative Reconstruction Technique, see :cite:`Kak2001`.

    Simultaneous Iterative Reconstruction Technique (SIRT) solves
    the following problem

    .. math:: A x = b

    The SIRT algorithm is 

    .. math:: x^{k+1} =  \mathrm{proj}_{C}( x^{k} + D ( A^{T} ( M * (b - Ax) ) ) ),

    where :math:`M = \frac{1}{A*\mathbb{1}}`, :math:`D = \frac{1}{A^{T}\mathbb{1}}`, :math:`\mathbb{1}` is a :code:`DataContainer` of ones
    and :math:`\mathrm{prox}_{C}` is the projection over a set :math:`C`.

    Parameters
    ----------

    initial : DataContainer, default = None
              Starting point of the algorithm, default value = Zero DataContainer 
    operator : LinearOperator
              The operator A.
    data : DataContainer
           The data b.
    lower : :obj:`float`, default = None
            Lower bound constraint, default value = :code:`-inf`.
    upper : :obj:`float`, default = None
            Upper bound constraint, default value = :code:`-inf`.
    constraint : Function, default = None
                A function with :code:`proximal` method, e.g., :class:`.IndicatorBox` function and :meth:`.IndicatorBox.proximal`,
                or :class:`.TotalVariation` function and :meth:`.TotalVariation.proximal`.

    kwargs:
        Keyword arguments used from the base class :class:`.Algorithm`.    

    Note 
    ----
    
    If :code:`constraint` is not passed, then :code:`lower` and :code:`upper` are looked at and an :class:`.IndicatorBox`
    function is created.

    If :code:`constraint` is passed, :code:`proximal` method is required to be implemented.

    Note
    ----

    The preconditioning arrays (weights) :code:`M` and :code:`D` used in SIRT are defined as

    .. math:: M = \frac{1}{A*\mathbb{1}} = \frac{1}{\sum_{j}a_{i,j}}

    .. math:: D = \frac{1}{A*\mathbb{1}} = \frac{1}{\sum_{i}a_{i,j}}

    In case of division errors above, :meth:`.fix_weights` can be used, where :code:`np.nan`, :code:`+np.inf` and :code:`-np.inf` values
    are replaced with 1.0.


    Examples
    --------
    .. math:: \underset{x}{\mathrm{argmin}} \| x - d\|^{2}
    
    >>> sirt = SIRT(initial = ig.allocate(0), operator = A, data = d, max_iteration = 5) 

    """    


    def __init__(self, initial, operator, data, lower=None, upper=None, constraint=None, **kwargs):

        super(SIRT, self).__init__(**kwargs)
        
        self.set_up(initial=initial, operator=operator, data=data, lower=lower, upper=upper, constraint=constraint)         

    def set_up(self, initial, operator, data, lower=None, upper=None, constraint=None):

        """
        Initialisation of the algorithm    
        """

        logging.info("{} setting up".format(self.__class__.__name__, ))
        
        self.x = initial.copy()
        self.tmp_x = self.x * 0.0
        self.operator = operator
        self.data = data
        
        self.r = data.copy()
        
        # relaxation parameter was set to 1.0 and not exposed to the user: removed
        
        self.constraint = constraint
        if constraint is None:
            if lower is not None or upper is not None:
                # IndicatorBox accepts None for lower and/or upper
                self.constraint=IndicatorBox(lower=lower,upper=upper)
                
        # Set up scaling matrices D and M.
        self._set_up_scaling_matrices()

        self.configured = True
        logging.info("{} configured".format(self.__class__.__name__, ))

    def _set_up_scaling_matrices(self):
        self.M = 1./self.operator.direct(self.operator.domain_geometry().allocate(value=1.0))                
        self.D = 1./self.operator.adjoint(self.operator.range_geometry().allocate(value=1.0))

    def fix_weights(self):

        r""" In case of division error when the preconditioning arrays :code:`M` and :code:`D`
        are defined, the :code:`np.nan`, :code:`+np.inf` and :code:`-np.inf` values are replaced with one.
        """
        
        # fix for possible +inf, -inf, nan values 
        for arr in [self.M, self.D]:  
            self._remove_nan_or_inf(arr, replace_with=1.0)

    def _remove_nan_or_inf(self, datacontainer, replace_with=1.0):
        """Replace nan and inf in datacontainer with a given value.
        
        Parameters:
        -------------
        
        datacontainer: DataContainer, BlockDataContainer
        
        replace_with: float, default 1.0
            Value to replace elements that evaluate to NaN or inf
            
            
        In case the input datacontainer is a :code:`BlockDataContainer` the substitution is executed for each container in the :code:`BlockDataContainer`."""
        if isinstance(datacontainer, BlockDataContainer):
            for block in datacontainer.containers:
                self._remove_nan_or_inf(block, replace_with=replace_with)
            return
        tmp = datacontainer.as_array()
        arr_replace = numpy.isfinite(tmp)
        tmp[~arr_replace] = replace_with     
        datacontainer.fill(tmp)

    def update(self):

        r""" Performs a single iteration of the SIRT algorithm

        .. math:: x^{k+1} =  \mathrm{proj}_{C}( x^{k} + D ( A^{T} ( M * (b - Ax) ) ) )

        """
        
        # self.r = self.data - self.operator.direct(self.x)
        self.operator.direct(self.x, out=self.r)
        self.r.sapyb(-1, self.data, 1.0, out=self.r)
        
        # self.x += self.relax_par * (self.D*self.operator.adjoint(self.M*self.r))
        self.r *= self.M
        self.operator.adjoint(self.r, out=self.tmp_x)
        self.x.sapyb(1.0, self.tmp_x, self.D, out=self.x)

        
        if self.constraint is not None:
            # IndicatorBox allows inplace operation for proximal
            self.constraint.proximal(self.x, tau=1, out=self.x)

    def update_objective(self):
        r"""Returns the objective 

        .. math:: \|A x - b\|^{2}

        """
        self.loss.append(self.r.squared_norm())
