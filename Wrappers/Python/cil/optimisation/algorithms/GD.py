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
from cil.optimisation.utilities import ConstantStepSize, ArmijoStepSizeRule

log = logging.getLogger(__name__)


class GD(Algorithm):
    """Gradient Descent algorithm

    Parameters
    ----------
    initial: DataContainer (e.g. ImageData)
        The initial point for the optimisation 
    objective_function: CIL function with a defined gradient 
        The function to be minimised. 
    step_size: float, default = None 
        Step size for gradient descent iteration if you want to use a constant step size. If left as None and do not pass a step_size_rule then the Armijio rule will be used to perform backtracking to choose a step size at each iteration. 
    step_size_rule: class with a `get_step_size` method or a function that takes an initialised CIL function as an argument and outputs a step size, default is None
        This could be a custom `step_size_rule` or one provided in :meth:`~cil.optimisation.utilities.StepSizeMethods`. If None is passed  then the algorithm will use either `ConstantStepSize` or `ArmijioStepSizeRule` depending on if a `step_size` is provided. 
    preconditioner: class with a `apply` method or a function that takes an initialised CIL function as an argument and modifies a provided `gradient`.
            This could be a custom `preconditioner` or one provided in :meth:`~cil.optimisation.utilities.preconditoner`. If None is passed  then `self.gradient_update` will remain unmodified. 

    rtol: positive float, default 1e-5
        optional parameter defining the relative tolerance comparing the current objective function to 0, default 1e-5, see numpy.isclose
    atol: positive float, default 1e-8
        optional parameter defining the absolute tolerance comparing the current objective function to 0, default 1e-8, see numpy.isclose

    """

    def __init__(self, initial=None, objective_function=None, step_size=None, rtol=1e-5, atol=1e-8, step_size_rule=None, preconditioner=None, **kwargs):
        '''GD algorithm creator
        '''
        super().__init__(**kwargs)
         
        self.alpha = kwargs.get('alpha',None)
        self.beta = kwargs.get('beta', None)
        if self.alpha is not None or self.beta is not None:
            raise DeprecationWarning('To modify the parameters for the Armijo rule please use `step_size_rule=ArmijoStepSizeRule(alpha, beta, kmax)`. The arguments `alpha` and `beta` will be deprecated. ')
        
        self.rtol = rtol
        self.atol = atol
        if initial is not None and objective_function is not None:
            self.set_up(initial=initial, objective_function=objective_function,
                        step_size=step_size, step_size_rule=step_size_rule, preconditioner=preconditioner)

    def set_up(self, initial, objective_function, step_size, step_size_rule, preconditioner):
        '''initialisation of the algorithm

        Parameters
        ----------
        initial: DataContainer (e.g. ImageData)
            The initial point for the optimisation 
        objective_function: CIL function with a defined gradient 
            The function to be minimised. 
        step_size: float, default = None 
            Step size for gradient descent iteration if you want to use a constant step size. If left as None and do not pass a step_size_rule then the Armijo rule will be used to perform backtracking to choose a step size at each iteration. 
        step_size_rule: class with a `get_step_size` method or a function that takes an initialised CIL function as an argument and outputs a step size, default is None
            This could be a custom `step_size_rule` or one provided in :meth:`~cil.optimisation.utilities.StepSizeMethods`. If None is passed  then the algorithm will use either `ConstantStepSize` or `ArmijoStepSizeRule` depending on if a `step_size` is provided. 
        preconditioner: class with a `apply` method or a function that takes an initialised CIL function as an argument and modifies a provided `gradient`.
            This could be a custom `preconditioner` or one provided in :meth:`~cil.optimisation.utilities.preconditoner`. If None is passed  then `self.gradient_update` will remain unmodified. 

        '''

        log.info("%s setting up", self.__class__.__name__)

        self.x = initial.copy()
        self.objective_function = objective_function

        
        
        if step_size_rule is None:
            if step_size is None:
                step_size_rule = ArmijoStepSizeRule(
                    alpha=self.alpha, beta=self.beta)
            else:
                step_size_rule = ConstantStepSize(step_size)
        else:
            if step_size is not None:
                raise TypeError(
                    'You have passed both a `step_size` and a `step_size_rule`, please pass one or the other')

        self.gradient_update = initial.copy()

        self.configured = True

        self.step_size_rule = step_size_rule

        self.preconditioner = preconditioner

        log.info("%s configured", self.__class__.__name__)

    def update(self):
        '''Performs a single iteration of the gradient descent algorithm'''
        self.objective_function.gradient(self.x, out=self.gradient_update)

        if self.preconditioner is not None:
            self.preconditioner.apply(self, self.gradient_update, out=self.gradient_update) #TODO:  Think about another name? 

        step_size = self.step_size_rule.get_step_size(self) 

        self.x.sapyb(1.0, self.gradient_update, -step_size, out=self.x)

    def update_objective(self):
        self.loss.append(self.objective_function(self.x))

    def should_stop(self):
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
