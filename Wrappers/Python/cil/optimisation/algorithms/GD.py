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
from cil.optimisation.utilities import ConstantStepSize, ArmijoStepSize, AdaptiveSensitivity, Sensitivity

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
    step_size_rule: class with a `__call__` method or a function that takes an initialised CIL function as an argument and outputs a step size, default is None
        This could be a custom `step_size_rule` or one provided in :meth:`~cil.optimisation.utilities.StepSizeMethods`. If None is passed  then the algorithm will use either `ConstantStepSize` or `ArmijioStepSize` depending on if a `step_size` is provided. 
    precondition: class with a `__call__` method or a function that takes an initialised CIL function as an argument and modifies se
    
    rtol: positive float, default 1e-5
        optional parameter defining the relative tolerance comparing the current objective function to 0, default 1e-5, see numpy.isclose
    atol: positive float, default 1e-8
        optional parameter defining the absolute tolerance comparing the current objective function to 0, default 1e-8, see numpy.isclose
    
    """
    def __init__(self, initial=None, objective_function=None, step_size=None, alpha=1e6, beta=0.5, rtol=1e-5, atol=1e-8, step_size_rule= None, preconditioner=None, **kwargs):
        '''GD algorithm creator
        '''
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.rtol = rtol
        self.atol = atol
        if initial is not None and objective_function is not None:
            self.set_up(initial=initial, objective_function=objective_function, step_size=step_size, step_size_rule=step_size_rule, preconditioner=preconditioner)
            
        

    def set_up(self, initial, objective_function, step_size, step_size_rule, preconditioner):
        '''initialisation of the algorithm

        Parameters
        ----------
        initial: DataContainer (e.g. ImageData)
            The initial point for the optimisation 
        objective_function: CIL function with a defined gradient 
            The function to be minimised. 
        step_size: float, default = None 
            Step size for gradient descent iteration if you want to use a constant step size. If left as None and do not pass a step_size_rule then the Armijio rule will be used to perform backtracking to choose a step size at each iteration. 
        step_size_rule: class with a `__call__` method or a function that takes an initialised CIL function as an argument and outputs a step size, default is None
            This could be a custom `step_size_rule` or one provided in :meth:`~cil.optimisation.utilities.StepSizeMethods`. If None is passed  then the algorithm will use either `ConstantStepSize` or `ArmijioStepSize` depending on if a `step_size` is provided. 
        preconditioner: 
        
        '''

        log.info("%s setting up", self.__class__.__name__)


        self.x = initial.copy()
        self.objective_function = objective_function

        if step_size_rule is None: 
            if step_size is None:
                step_size_rule=ArmijoStepSize(alpha=self.alpha, beta=self.beta)
            else:
                step_size_rule=ConstantStepSize(step_size)
        else:
            if step_size is not None:
                raise TypeError('You have passed both a `step_size` and a `step_size_rule`, please pass one or the other')

        self.update_objective()

        self.x_update = initial.copy()

        self.configured = True
        
        self.step_size_rule=step_size_rule
        
        self.preconditioner = preconditioner 
        

        log.info("%s configured", self.__class__.__name__)


    def update(self):
        '''Performs a single iteration of the gradient descent algorithm'''
        self.objective_function.gradient(self.x, out=self.x_update)
        
        if self.preconditioner is not None:
            self.preconditioner(self)

        step_size = self.step_size_rule(self)

        self.x.sapyb(1.0, self.x_update, -step_size, out=self.x)

    def update_objective(self):
        self.loss.append(self.objective_function(self.x))

    def should_stop(self):
        '''Stopping criterion for the gradient descent algorithm '''
        return super().should_stop() or \
            numpy.isclose(self.get_last_objective(), 0., rtol=self.rtol,
                atol=self.atol, equal_nan=False)
