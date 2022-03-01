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
from numpy import inf
import numpy
import warnings

class SIRT(Algorithm):    

    r"""Simultaneous Iterative Reconstruction Technique 

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
    constraint : IndicatorBox function, default = None
                A constraint, e.g., :class:`.IndicatorBox` function is enforced in every iteration.
    kwargs:
        Keyword arguments used from the base class :class:`.Algorithm`.    

    Note 
    ----
    
    If :code:`constraint` is not passed, then :code:`lower` and :code:`upper` are looked at.

    If :code:`constraint` is passed, it should be an :class:`.IndicatorBox` function, 
    and in that case :code:`lower` and :code:`upper` inputs are ignored. 

    If :math:`M = \frac{1}{A*\mathbb{1}}`, :math:`D = \frac{1}{A^{T}\mathbb{1}}` contain :code:`NaN`, :math:`\pm\inf`
    they are replaced by :math:`1`.


    

    
    
    Examples
    --------
    .. math:: \underset{x}{\mathrm{argmin}} \| x - 5\|^{2}
    
    >>> from cil.framework import ImageGeometry
    >>> from cil.optimisation.operators import IdentityOperator    
    >>> from cil.optimisation.algorithms import SIRT
    >>> ig = ImageGeometry(3,4)
    >>> A = IdentityOperator(ig)
    >>> b = ig.allocate(5.0)
    >>> sirt = SIRT(initial = ig.allocate(0), operator = A, data=b, max_iteration=5)  
    >>> sirt.run() 
    >>> sirt.solution.array
    array([[5., 5., 5.],
           [5., 5., 5.],
           [5., 5., 5.],
           [5., 5., 5.]], dtype=float32)


    """    


    def __init__(self, initial, operator, data, lower=None, upper=None, constraint=None, **kwargs):

        super(SIRT, self).__init__(**kwargs)
        if kwargs.get('x_init', None) is not None:
            if initial is None:
                warnings.warn('The use of the x_init parameter is deprecated and will be removed in following version. Use initial instead',
                   DeprecationWarning, stacklevel=4)
                initial = kwargs.get('x_init', None)
            else:
                raise ValueError('{} received both initial and the deprecated x_init parameter. It is not clear which one we should use.'\
                    .format(self.__class__.__name__))
        
        self.set_up(initial=initial, operator=operator, data=data, lower=lower, upper=upper, constraint=constraint)         

    def set_up(self, initial, operator, data, lower=None, upper=None, constraint=None):

        """
        Initialisation of the algorithm    
        """

        print("{} setting up".format(self.__class__.__name__, ))
        
        self.x = initial.copy()
        self.operator = operator
        self.data = data
        
        self.r = data.copy()
        
        self.relax_par = 1.0
        
        self.constraint = constraint
        if constraint is None:
            if lower is not None or upper is not None:
                if lower is None:
                    lower=-inf
                if upper is None:
                    upper=inf
                self.constraint=IndicatorBox(lower=lower,upper=upper)
                
        # Set up scaling matrices D and M.
        self.M = 1./self.operator.direct(self.operator.domain_geometry().allocate(value=1.0))                
        self.D = 1./self.operator.adjoint(self.operator.range_geometry().allocate(value=1.0))

        # fix for possible inf values (code from numpy.nan_to_nam)
        # TODO replace with
        # numpy.nan_to_num(self.M, copy = False, nan = 1, neginf=1, posinf=1) 
        # numpy.nan_to_num(self.D, copy = False, nan = 1, neginf=1, posinf=1) 

        idx_nan1 = numpy.isnan(self.M.as_array())
        idx_pinf1 = numpy.isnan(self.M.as_array())
        idx_ninf1 = numpy.isnan(self.M.as_array())        
        numpy.copyto(self.M.as_array(), 1., where=idx_nan1)
        numpy.copyto(self.M.as_array(), 1., where=idx_pinf1)
        numpy.copyto(self.M.as_array(), 1., where=idx_ninf1)

        idx_nan2 = numpy.isnan(self.D.as_array())
        idx_pinf2 = numpy.isnan(self.D.as_array())
        idx_ninf2 = numpy.isnan(self.D.as_array())        
        numpy.copyto(self.D.as_array(), 1., where=idx_nan2)
        numpy.copyto(self.D.as_array(), 1., where=idx_pinf2)
        numpy.copyto(self.D.as_array(), 1., where=idx_ninf2)

        self.configured = True
        print("{} configured".format(self.__class__.__name__, ))
          
    def update(self):

        r""" Performs a single iteration of the SIRT algorithm

        .. math:: x^{k+1} =  \mathrm{proj}_{C}( x^{k} + D ( A^{T} ( M * (b - Ax) ) ) )

        """
        
        self.r = self.data - self.operator.direct(self.x)
        
        self.x += self.relax_par * (self.D*self.operator.adjoint(self.M*self.r))
        
        if self.constraint is not None:
            self.x = self.constraint.proximal(self.x, None)

    def update_objective(self):
        r"""Returns the objective 

        .. math:: \|A x - b\|^{2}

        """
        self.loss.append(self.r.squared_norm())

