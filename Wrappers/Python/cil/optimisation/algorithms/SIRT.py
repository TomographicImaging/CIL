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

from cil.optimisation.algorithms import Algorithm
from cil.optimisation.functions import IndicatorBox
from cil.framework import BlockDataContainer
from cil.utilities.errors import InPlaceError
import numpy
import logging

log = logging.getLogger(__name__)


class SIRT(Algorithm):

    r"""Simultaneous Iterative Reconstruction Technique, see :cite:`Kak2001`.

    Simultaneous Iterative Reconstruction Technique (SIRT) solves
    the following problem

    .. math:: A x = b

    The SIRT algorithm is

    .. math:: x^{k+1} =  \mathrm{proj}_{C}( x^{k} + \omega * D ( A^{T} ( M * (b - Ax^{k}) ) ) ),

    where,
    :math:`M = \frac{1}{A*\mathbb{1}}`,
    :math:`D = \frac{1}{A^{T}\mathbb{1}}`,
    :math:`\mathbb{1}` is a :code:`DataContainer` of ones,
    :math:`\mathrm{prox}_{C}` is the projection over a set :math:`C`,
    and :math:`\omega` is the relaxation parameter.

    Parameters
    ----------

    initial : DataContainer, default = None
        Starting point of the algorithm, default value = Zero DataContainer
    operator : LinearOperator
        The operator A.
    data : DataContainer
        The data b.
    lower : :obj:`float`, default = None
        Lower bound constraint
    upper : :obj:`float`, default = None
        Upper bound constraint
    constraint : Function, default = None
        A function with :code:`proximal` method, e.g., :class:`.IndicatorBox` function and :meth:`.IndicatorBox.proximal`,
        or :class:`.TotalVariation` function and :meth:`.TotalVariation.proximal`.

    kwargs:
        Keyword arguments used from the base class :class:`.Algorithm`.

    Note
    ----
    If :code:`constraint` is not passed, :code:`lower` and :code:`upper` are used to create an :class:`.IndicatorBox` and apply its :code:`proximal`.

    If :code:`constraint` is passed, :code:`proximal` method is required to be implemented.

    Note
    ----

    The preconditioning arrays (weights) :code:`M` and :code:`D` used in SIRT are defined as

    .. math:: M = \frac{1}{A*\mathbb{1}} = \frac{1}{\sum_{j}a_{i,j}}

    .. math:: D = \frac{1}{A*\mathbb{1}} = \frac{1}{\sum_{i}a_{i,j}}


    Examples
    --------
    .. math:: \underset{x}{\mathrm{argmin}} \frac{1}{2}\| x - d\|^{2}

    >>> sirt = SIRT(initial = ig.allocate(0), operator = A, data = d, max_iteration = 5)

    """


    def __init__(self, initial, operator, data, lower=None, upper=None, constraint=None, **kwargs):

        super(SIRT, self).__init__(**kwargs)

        self.set_up(initial=initial, operator=operator, data=data, lower=lower, upper=upper, constraint=constraint)

    def set_up(self, initial, operator, data, lower=None, upper=None, constraint=None):
        """Initialisation of the algorithm"""
        log.info("%s setting up", self.__class__.__name__)
        self.x = initial.copy()
        self.tmp_x = self.x * 0.0
        self.operator = operator
        self.data = data

        self.r = data.copy()

        self.constraint = constraint
        if constraint is None:
            if lower is not None or upper is not None:
                # IndicatorBox accepts None for lower and/or upper
                self.constraint=IndicatorBox(lower=lower,upper=upper)

        self._relaxation_parameter = 1

        # Set up scaling matrices D and M.
        self._set_up_weights()

        self.configured = True
        log.info("%s configured", self.__class__.__name__)

    @property
    def relaxation_parameter(self):
        return self._relaxation_parameter

    @property
    def D(self):
        return self._Dscaled / self._relaxation_parameter

    def set_relaxation_parameter(self, value=1.0):
        """Set the relaxation parameter :math:`\omega`

        Parameters
        ----------
        value : float
            The relaxation parameter to be applied to the update. Must be between 0 and 2 to guarantee asymptotic convergence.

        """
        if value <= 0 or value >= 2:
            raise ValueError("Expected relaxation parameter to be in range 0-2. Got {}".format(value))

        self._relaxation_parameter = value
        self._set_up_weights()
        self._Dscaled *= self._relaxation_parameter


    def _set_up_weights(self):
        self.M = 1./self.operator.direct(self.operator.domain_geometry().allocate(value=1.0))
        self._Dscaled = 1./self.operator.adjoint(self.operator.range_geometry().allocate(value=1.0))

        for arr in [self.M, self._Dscaled]:
            self._remove_nan_or_inf(arr, replace_with=1.0)


    def _remove_nan_or_inf(self, datacontainer, replace_with=1.0):
        """Replace nan and inf in datacontainer with a given value.

        Parameters:
        -------------

        datacontainer: DataContainer, BlockDataContainer

        replace_with: float, default 1.0
            Value to replace elements that evaluate to NaN or inf


        In case the input datacontainer is a :code:`BlockDataContainer` the substitution is executed for each container in the :code:`BlockDataContainer`.
        """
        if isinstance(datacontainer, BlockDataContainer):
            for block in datacontainer.containers:
                self._remove_nan_or_inf(block, replace_with=replace_with)
            return
        tmp = datacontainer.as_array()
        numpy.nan_to_num(tmp, copy=False, nan=replace_with, posinf=replace_with, neginf=replace_with)
        datacontainer.fill(tmp)


    def update(self):

        r""" Performs a single iteration of the SIRT algorithm

        .. math:: x^{k+1} =  \mathrm{proj}_{C}( x^{k} + \omega * D ( A^{T} ( M * (b - Ax) ) ) )

        """

        # self.r = self.data - self.operator.direct(self.x)
        self.operator.direct(self.x, out=self.r)
        self.r.sapyb(-1, self.data, 1.0, out=self.r)

        # self.D is prescaled by _relaxation_parameter (default 1)
        self.r *= self.M
        self.operator.adjoint(self.r, out=self.tmp_x)
        self.x.sapyb(1.0, self.tmp_x, self._Dscaled, out=self.x)

        if self.constraint is not None:
            try:
                self.constraint.proximal(self.x, tau=1, out=self.x)
            except InPlaceError:
                self.x=self.constraint.proximal(self.x, tau=1)

    def update_objective(self):
        r"""Returns the objective

        .. math:: \frac{1}{2}\|A x - b\|^{2}

        """
        self.loss.append(0.5*(self.operator.direct(self.x)-self.data).squared_norm())

