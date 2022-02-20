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
import warnings

class SIRT(Algorithm):    

    r"""Simultaneous Iterative Reconstruction Technique 

    Simultaneous Iterative Reconstruction Technique (SIRT) used to solve
    the following problem

    .. math:: A x = b

    .. math:: x^{k+1} =  \mathrm{proj}_{C}( x^{k} + D ( A^{T} ( M * (b - Ax) ) ) ),

    where :math:`M = A*\mathbb{1}`, :math:`D = A^{T}\mathbb{1}`, :math:`\mathbb{1}` is a :code:`DataContainer` of ones
    and :math:`\mathrm{prox}_{C}` is the projection over a set :math:`C`.

    Parameters
    ----------

    initial : DataContainer, default = None
              Starting point of the algorithm, default value = :math:`0` 
    operator : LinearOperator or MatrixLinearOperator
              The operator A .
    data : DataContainer
           The data b .
    lower : :obj:`float`, default = None
            Lower bound constraint, default value = :code:`-inf`
    upper : :obj:`float`, default = None
            Upper bound constraint, default value = :code:`-inf`.
    constraint : IndicatorBox function, default = None
                 Enforce box constraint using the :class:`.IndicatorBox` function.


    
    **kwargs:
        Keyword arguments used from the base class :class:`.Algorithm`.    
    
        max_iteration : :obj:`int`, optional, default=0
            Maximum number of iterations.
        update_objective_interval : :obj:`int`, optional, default=1
            Evaluates the objective: :math:`\|A x - b\|^{2}` every             

    Note 
    ----
    
    If :code:`constraint` is passed, it should be an :class:`.IndicatorBox` function, 
    and in that case :code:`lower` and :code:`upper` inputs are ignored. 
    
    If :code:`constraint` is passed, then :code:`lower` and :code:`upper` are looked at, 
    and if at least one is not None, then an :class:`.IndicatorBox` is set up which 
    provides the proximal mapping to enforce lower and upper bounds.

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

        # fix for possible inf values
        self.M.array[self.M.array==inf] = 1
        self.D.array[self.M.array==inf] = 1

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
        """
        self.loss.append(self.r.squared_norm())

