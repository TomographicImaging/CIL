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
from numbers import Integral
from typing import List, Optional
from warnings import warn

import numpy as np

from cil.optimisation.utilities.callbacks import Callback, LogfileCallback, _OldCallback, ProgressCallback


class Algorithm:
    '''Base class for iterative algorithms

      provides the minimal infrastructure.

      Algorithms are iterables so can be easily run in a for loop. They will
      stop as soon as the stop criterion is met.
      The user is required to implement the :code:`set_up`, :code:`__init__`, :code:`update` and
      and :code:`update_objective` methods

      A courtesy method :code:`run` is available to run :code:`n` iterations. The method accepts
      a :code:`callbacks` list of callables, each of which receive the current Algorithm object
      (which in turn contains the iteration number and the actual objective value)
      and can be used to trigger print to screens and other user interactions. The :code:`run`
      method will stop when the stopping criterion is met or `StopIteration` is raised.
   '''

    def __init__(self, update_objective_interval=1, max_iteration=None, log_file=None):
        '''Set the minimal number of parameters:

        :param update_objective_interval: the interval every which we would save the current\
                                       objective. 1 means every iteration, 2 every 2 iteration\
                                       and so forth. This is by default 1 and should be increased\
                                       when evaluating the objective is computationally expensive.
        :type update_objective_interval: int, optional, default 1
        '''
        self.iteration = -1
        self.__max_iteration = 1
        if max_iteration is not None:
            warn("use `Algorithm.run(iterations)` instead of `Algorithm(max_iteration)`", DeprecationWarning, stacklevel=2)
            self.__max_iteration = max_iteration
        self.__loss = []
        self.memopt = False
        self.configured = False
        self._iteration = []
        self.update_objective_interval = update_objective_interval
        # self.x = None
        self.iter_string = 'Iter'
        if log_file is not None:
            warn("use `run(callbacks=[LogfileCallback(log_file)])` instead of `log_file`",
                 DeprecationWarning, stacklevel=2)
            self.__log_file = log_file

    def set_up(self, *args, **kwargs):
        '''Set up the algorithm'''
        raise NotImplementedError
    def update(self):
        '''A single iteration of the algorithm'''
        raise NotImplementedError

    def should_stop(self):
        '''default stopping criterion: number of iterations

        The user can change this in concrete implementation of iterative algorithms.'''
        return self.iteration > self.max_iteration

    def __set_up_logger(self, *_, **__):
        warn("use `run(callbacks=[LogfileCallback(log_file)])` instead", DeprecationWarning, stacklevel=2)

    def max_iteration_stop_criterion(self):
        warn("use `should_stop()` instead of `max_iteration_stop_criterion()`", DeprecationWarning, stacklevel=2)
        return self.iteration > self.max_iteration

    def __iter__(self):
        '''Algorithm is an iterable'''
        return self

    def __next__(self):
        '''Algorithm is an iterable

        This method triggers :code:`update()` and :code:`update_objective()`
        '''
        if self.should_stop():
            raise StopIteration
        if self.iteration == -1 and self.update_objective_interval > 0:
            self._iteration.append(self.iteration)
            self.update_objective()
            self.iteration += 1
            return self.iteration
        if not self.configured:
            raise ValueError('Algorithm not configured correctly. Please run set_up.')
        self.update()
        self.iteration += 1

        self._update_previous_solution()

        if self.iteration >= 0 and self.update_objective_interval > 0 and\
            self.iteration % self.update_objective_interval == 0:

            self._iteration.append(self.iteration)
            self.update_objective()
        return self.iteration

    def _update_previous_solution(self):
        """ Update the previous solution with the current one

        The concrete algorithm calls update_previous_solution. Normally this would
        entail the swapping of pointers:

        .. highlight:: python
        .. code-block:: python

            tmp = self.x_old
            self.x_old = self.x
            self.x = tmp

        """
        pass

    def get_output(self):
        " Returns the current solution. "
        return self.x

    def _provable_convergence_condition(self):
        raise NotImplementedError(" Convergence criterion is not implemented for this algorithm. ")

    def is_provably_convergent(self):
        """ Check if the algorithm is convergent based on the provable convergence criterion.
        """
        return self._provable_convergence_condition()

    @property
    def solution(self):
        return self.get_output()

    def get_last_loss(self, return_all=False):
        '''Returns the last stored value of the loss function

        if update_objective_interval is 1 it is the value of the objective at the current
        iteration. If update_objective_interval > 1 it is the last stored value.
        '''
        try:
            objective = self.__loss[-1]
        except IndexError:
            objective = np.nan
        if isinstance(objective, list):
            return objective if return_all else objective[0]
        return [objective, np.nan, np.nan] if return_all else objective

    get_last_objective = get_last_loss # alias

    def update_objective(self):
        '''calculates the objective with the current solution'''
        raise NotImplementedError

    @property
    def iterations(self):
        '''returns the iterations at which the objective has been evaluated'''
        return self._iteration
    @property
    def loss(self):
        '''returns the list of the values of the objective during the iteration

        The length of this list may be shorter than the number of iterations run when
        the update_objective_interval > 1
        '''
        return self.__loss

    objective = loss # alias

    @property
    def max_iteration(self):
        '''gets the maximum number of iterations'''
        return self.__max_iteration

    @max_iteration.setter
    def max_iteration(self, value):
        '''sets the maximum number of iterations'''
        assert isinstance(value, Integral)
        self.__max_iteration = value

    @property
    def update_objective_interval(self):
        return self.__update_objective_interval

    @update_objective_interval.setter
    def update_objective_interval(self, value):
        if not isinstance(value, Integral) or value < 0:
            raise ValueError('interval must be an integer >= 0')
        self.__update_objective_interval = value

    def run(self, iterations=None, callbacks: Optional[List[Callback]]=None, verbose=1, **kwargs):
        '''run upto :code:`iterations` with callbacks/logging.

        :param iterations: number of iterations to run. If not set the algorithm will
          run until :code:`should_stop()` is reached
        :param verbose: 0=quiet, 1=info, 2=debug
        :param callbacks: list of callables which are passed the current Algorithm
          object each iteration. Defaults to :code:`[ProgressCallback(verbose)]`.
        '''
        if 'print_interval' in kwargs:
            warn("use `TextProgressCallback(miniters)` instead of `run(print_interval)`",
                 DeprecationWarning, stacklevel=2)
        if callbacks is None:
            callbacks = [ProgressCallback(verbose=verbose)]
        # transform old-style callbacks into new
        callback = kwargs.get('callback', None)
        if callback is not None:
            callbacks.append(_OldCallback(callback, verbose=verbose))
        if hasattr(self, '__log_file'):
            callbacks.append(LogfileCallback(self.__log_file, verbose=verbose))

        if self.should_stop():
            print("Stop criterion has been reached.")
        if iterations is None:
            warn("`run()` missing `iterations`", DeprecationWarning, stacklevel=2)
            iterations = self.max_iteration

        if self.iteration == -1 and self.update_objective_interval>0:
            iterations+=1

        # call `__next__` upto `iterations` times or until `StopIteration` is raised
        self.max_iteration = self.iteration + iterations
        for _ in zip(range(self.iteration, self.iteration + iterations), self):
            try:
                for callback in callbacks:
                    callback(self)
            except StopIteration:
                break

    def objective_to_dict(self, verbose=False):
        obj = self.get_last_objective(return_all=verbose)
        if isinstance(obj, list) and len(obj) == 3:
            if not np.isnan(obj[1:]).all():
                return {'primal': obj[0], 'dual': obj[1], 'primal_dual': obj[2]}
            obj = obj[0]
        return {'objective': obj}

    def objective_to_string(self, verbose=False):
        warn("consider using `run(callbacks=[LogfileCallback(log_file)])` instead", DeprecationWarning, stacklevel=2)
        return str(self.objective_to_dict(verbose=verbose))

    def verbose_output(self, *_, **__):
        warn("use `run(callbacks=[ProgressCallback()])` instead", DeprecationWarning, stacklevel=2)

    def verbose_header(self, *_, **__):
        warn("consider using `run(callbacks=[LogfileCallback(log_file)])` instead", DeprecationWarning, stacklevel=2)
