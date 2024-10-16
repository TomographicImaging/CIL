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

import numpy
from cil.optimisation.algorithms import Algorithm
import logging
from cil.optimisation.utilities import ConstantStepSize, ArmijoStepSizeRule, StepSizeRule
from warnings import warn
from numbers import Real

log = logging.getLogger(__name__)


class GD(Algorithm):
    """Gradient Descent algorithm

    Parameters
    ----------
    initial: DataContainer (e.g. ImageData)
        The initial point for the optimisation 
    objective_function: CIL function (:meth:`~cil.optimisation.functions.Function`. ) with a defined gradient method 
        The function to be minimised. 
    step_size: positive real float or subclass of :meth:`~cil.optimisation.utilities.StepSizeRule`, default = None 
        If you pass a float this will be used as a constant step size. If left as None and do not pass a step_size_rule then the Armijio rule will be used to perform backtracking to choose a step size at each iteration. If a child class of :meth:`cil.optimisation.utilities.StepSizeRule`' is passed then it's method `get_step_size` is called for each update. 
    preconditioner: class with a `apply` method or a function that takes an initialised CIL function as an argument and modifies a provided `gradient`.
            This could be a custom `preconditioner` or one provided in :meth:`~cil.optimisation.utilities.preconditioner`. If None is passed  then `self.gradient_update` will remain unmodified. 

    """

    def __init__(self, initial=None, objective_function=None, step_size=None,   preconditioner=None, **kwargs):
        '''GD algorithm creator
        '''

        self.alpha = kwargs.pop('alpha', None)
        self.beta = kwargs.pop('beta', None)
        self.rtol = kwargs.pop('rtol', 0) # to be deprecated
        self.atol = kwargs.pop('atol', 0) # to be deprecated
        
        super().__init__(**kwargs)

        if self.alpha is not None or self.beta is not None:
            warn('To modify the parameters for the Armijo rule please use `step_size_rule=ArmijoStepSizeRule(alpha, beta, kmax)`. The arguments `alpha` and `beta` will be deprecated. ', DeprecationWarning, stacklevel=2)


        if self.rtol!=0 or self.atol!=0: # to be deprecated
            warn('`rtol` and `atol` are deprecated. For early stopping, please use a callback (cil.optimisation.utilities.callbacks) instead for example `EarlyStoppingObjectiveValue`.', DeprecationWarning, stacklevel=2)
        else:
            logging.info('In a break with backwards compatibility, GD no longer automatically stops if the objective function is close to zero. For this functionality, please use a callback (cil.optimisation.utilities.callbacks).' )    
            
        if initial is not None and objective_function is not None:
            self.set_up(initial=initial, objective_function=objective_function,
                        step_size=step_size,  preconditioner=preconditioner)

    def set_up(self, initial, objective_function, step_size, preconditioner):
        '''initialisation of the algorithm

        Parameters
        ----------
        initial: DataContainer (e.g. ImageData)
            The initial point for the optimisation 
        objective_function: CIL function with a defined gradient 
            The function to be minimised. 
        step_size: positive real float or subclass of :meth:`~cil.optimisation.utilities.StepSizeRule`, default = None 
            If you pass a float this will be used as a constant step size. If left as None and do not pass a step_size_rule then the Armijio rule will be used to perform backtracking to choose a step size at each iteration. If a child class of :meth:`cil.optimisation.utilities.StepSizeRule`' is passed then it's method `get_step_size` is called for each update. 
        preconditioner: class with a `apply` method or a function that takes an initialised CIL function as an argument and modifies a provided `gradient`.
            This could be a custom `preconditioner` or one provided in :meth:`~cil.optimisation.utilities.preconditioner`. If None is passed  then `self.gradient_update` will remain unmodified. 

        '''

        log.info("%s setting up", self.__class__.__name__)

        self.x = initial.copy()
        self.objective_function = objective_function

        if step_size is None:
            self.step_size_rule = ArmijoStepSizeRule(
                alpha=self.alpha, beta=self.beta)
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
        self.objective_function.gradient(self.x, out=self.gradient_update)

        if self.preconditioner is not None:
            self.preconditioner.apply(
                self, self.gradient_update, out=self.gradient_update)

        step_size = self.step_size_rule.get_step_size(self)

        self.x.sapyb(1.0, self.gradient_update, -step_size, out=self.x)

    def update_objective(self):
        self.loss.append(self.objective_function(self.solution))

    def should_stop(self): # to be deprecated 
        '''Stopping criterion for the gradient descent algorithm '''
        return super().should_stop() or \
            numpy.isclose(self.get_last_objective(), 0., rtol=self.rtol,
                          atol=self.atol, equal_nan=False)

    @property
    def step_size(self):
        if isinstance(self.step_size_rule, ConstantStepSize):
            return self.step_size_rule.step_size
        else:
            raise TypeError(
                "There is not a constant step size, it is set by a step-size rule")
