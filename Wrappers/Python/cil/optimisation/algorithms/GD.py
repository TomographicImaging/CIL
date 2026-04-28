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
import logging
from cil.optimisation.utilities import ConstantStepSize, ArmijoStepSizeRule, StepSizeRule
from numbers import Real

log = logging.getLogger(__name__)


class GD(Algorithm):
    """Gradient Descent algorithm

    Parameters
    ----------
    initial: DataContainer (e.g. ImageData)
        The initial point for the optimisation 
    f: CIL function (:meth:`~cil.optimisation.functions.Function`. ) with a defined gradient method 
        The function to be minimised. 
    step_size: positive real float or subclass of :meth:`~cil.optimisation.utilities.StepSizeRule`, default = None 
        If you pass a float this will be used as a constant step size. If left as None and do not pass a step size rule then the Armijo rule will be used to perform backtracking to choose a step size at each iteration (:meth:`~cil.optimisation.utilities.ArmijoStepSizeRule`). If a child class of :meth:`cil.optimisation.utilities.StepSizeRule`' is passed then its method `get_step_size` is called for each update. 
    preconditioner: class with a `apply` method or a function that takes an initialised CIL function as an argument and modifies a provided `gradient`.
            This could be a custom `preconditioner` or one provided in :meth:`~cil.optimisation.utilities.preconditioner`. If None is passed  then `self.gradient_update` will remain unmodified. 

    """

    def __init__(self, initial=None, f=None, step_size=None, preconditioner=None, **kwargs):

        super().__init__(**kwargs)

        if initial is not None and f is not None:
            self.set_up(initial=initial, f=f, step_size=step_size,
                        preconditioner=preconditioner)

    def set_up(self, initial, f, step_size, preconditioner):
        '''initialisation of the algorithm

        Parameters
        ----------
        initial: DataContainer (e.g. ImageData)
            The initial point for the optimisation 
        f: CIL function with a defined gradient 
            The function to be minimised. 
        step_size: positive real float or subclass of :meth:`~cil.optimisation.utilities.StepSizeRule`, default = None 
            If you pass a float this will be used as a constant step size. If left as None and do not pass a step_size_rule then the Armijo rule will be used to perform backtracking to choose a step size at each iteration. If a child class of :meth:`cil.optimisation.utilities.StepSizeRule`' is passed then it's method `get_step_size` is called for each update. 
        preconditioner: class with a `apply` method or a function that takes an initialised CIL function as an argument and modifies a provided `gradient`.
            This could be a custom `preconditioner` or one provided in :meth:`~cil.optimisation.utilities.preconditioner`. If None is passed  then `self.gradient_update` will remain unmodified. 

        '''

        log.info("%s setting up", self.__class__.__name__)

        self.x = initial.copy()
        self._objective_function = f

        if step_size is None:
            self.step_size_rule = ArmijoStepSizeRule()
        elif isinstance(step_size, Real):
            self.step_size_rule = ConstantStepSize(step_size)
        elif isinstance(step_size, StepSizeRule):
            self.step_size_rule = step_size
        else:
            raise TypeError(
                '`step_size` must be `None`, a Real float or a child class of :meth:`cil.optimisation.utilities.StepSizeRule`')
        self.gradient_update = initial.copy()

        self.configured = True

        self.preconditioner = preconditioner

        log.info("%s configured", self.__class__.__name__)

    def update(self):
        '''Performs a single iteration of the gradient descent algorithm'''
        self._objective_function.gradient(self.x, out=self.gradient_update)

        if self.preconditioner is not None:
            self.preconditioner.apply(
                self, self.gradient_update, out=self.gradient_update)

        self._step_size = self.step_size_rule.get_step_size(self)

        self.x.sapyb(1.0, self.gradient_update, -self._step_size, out=self.x)

    def update_objective(self):
        self.loss.append(self._objective_function(self.solution))

    @property
    def step_size(self):
        '''
        Returns the most recently used step size. Note, if the step-size is set by a non-constant step size rule, you must use the algorithm run or update method before this getter will return the most recently used step size. 
        '''

        if isinstance(self.step_size_rule, ConstantStepSize):
            return self.step_size_rule.step_size
        else:
            try:
                return self._step_size
            except AttributeError:
                raise NotImplementedError(
                    "Note the step-size is set by a step-size rule and could change with each iteration. Call the algorithm run or update method first and then this function will give the most recently used step size.")

    def calculate_objective_function_at_point(self, x):
        """ Calculates the objective at a given point x

        .. math:: f(x)

        Parameters
        ----------
        x : DataContainer

        """
        return self._objective_function(x)
