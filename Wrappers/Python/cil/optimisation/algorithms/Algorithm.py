# -*- coding: utf-8 -*-
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

import functools
import logging
from abc import ABC, abstractmethod
from numbers import Integral
from warnings import warn

import numpy as np
from tqdm.auto import tqdm as tqdm_auto


class Callback(ABC):
    '''Base Callback to inherit from for use in :code:`Algorithm.run(callbacks: list[Callback])`.'''
    def __init__(self, verbose=1):
        self.verbose = verbose

    @abstractmethod
    def __call__(self, algorithm):
        pass


class OldCallback(Callback):
    '''Converts an old-style :code:`function(iteration, objective, x)`
      to a new-style :code:`Callback`.
    '''
    def __init__(self, function, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = function

    def __call__(self, algorithm):
        if algorithm.update_objective_interval > 0 and algorithm.iteration % algorithm.update_objective_interval == 0:
            self.func(algorithm.iteration, algorithm.get_last_objective(return_all=self.verbose), algorithm.x)


class ProgressCallback(Callback):
    ''':code:`tqdm`-based progress bar.'''
    def __init__(self, verbose=1, tqdm_class=tqdm_auto, **tqdm_kwargs):
        '''
        :param tqdm_kwargs: passed to :code:`tqdm_class`
          (e.g. :code:`file=some_logfile`).
        '''
        super().__init__(verbose=verbose)
        self.tqdm_class = tqdm_class
        self.tqdm_kwargs = tqdm_kwargs

    def __call__(self, algorithm):
        if not hasattr(self, 'pbar'):
            tqdm_kwargs = self.tqdm_kwargs
            tqdm_kwargs.setdefault('total', algorithm.max_iteration)
            tqdm_kwargs.setdefault('disable', not self.verbose)
            self.pbar = self.tqdm_class(**tqdm_kwargs)
        if self.pbar.update(algorithm.iteration - self.pbar.n): # only if screen was updated
            self.pbar.set_postfix(objective=algorithm.objective_to_string(self.verbose>=2))
            # if algorithm.logger: algorithm.logger.debug(self.pbar)


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

    def __init__(self, max_iteration=0, update_objective_interval=1, log_file=None, **kwargs):
        '''Constructor

        Set the minimal number of parameters:


        :param max_iteration: maximum number of iterations
        :type max_iteration: int, optional, default 0
        :param update_objective_interval: the interval every which we would save the current\
                                       objective. 1 means every iteration, 2 every 2 iteration\
                                       and so forth. This is by default 1 and should be increased\
                                       when evaluating the objective is computationally expensive.
        :type update_objective_interval: int, optional, default 1
        :param log_file: log verbose output to file
        :type log_file: str, optional, default None
        '''
        self.iteration = -1
        self.__max_iteration = max_iteration
        self.__loss = []
        self.memopt = False
        self.configured = False
        self._iteration = []
        self.update_objective_interval = update_objective_interval
        # self.x = None
        self.iter_string = 'Iter'
        self.logger = None
        self.__set_up_logger(log_file)

    def set_up(self, *args, **kwargs):
        '''Set up the algorithm'''
        raise NotImplementedError
    def update(self):
        '''A single iteration of the algorithm'''
        raise NotImplementedError

    def should_stop(self):
        '''default stopping criterion: number of iterations

        The user can change this in concrete implementation of iterative algorithms.'''
        return self.max_iteration_stop_criterion()

    def __set_up_logger(self, fname):
        """Set up the logger if desired"""
        if fname:
            print("Will output results to: " +  fname)
            handler = logging.FileHandler(fname)
            self.logger = logging.getLogger("obj_fn")
            self.logger.setLevel(logging.INFO)
            self.logger.addHandler(handler)

    def max_iteration_stop_criterion(self):
        '''default stop criterion for iterative algorithm: max_iteration reached'''
        return self.iteration > self.max_iteration

    def __iter__(self):
        '''Algorithm is an iterable'''
        return self

    def __next__(self):
        '''Algorithm is an iterable

        calling this method triggers update and update_objective
        '''
        if self.should_stop():
            raise StopIteration
        if self.iteration == -1 and self.update_objective_interval > 0:
            self._iteration.append(self.iteration)
            self.update_objective()
            self.iteration += 1
            return
        if not self.configured:
            raise ValueError('Algorithm not configured correctly. Please run set_up.')
        self.update()
        self.iteration += 1

        self._update_previous_solution()

        if self.iteration >= 0 and self.update_objective_interval > 0 and\
            self.iteration % self.update_objective_interval == 0:

            self._iteration.append(self.iteration)
            self.update_objective()


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
        else:
            return [objective, np.nan, np.nan] if return_all else objective
    def get_last_objective(self, **kwargs):
        '''alias to get_last_loss'''
        return self.get_last_loss(**kwargs)

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

    @property
    def objective(self):
        '''alias of loss'''
        return self.loss

    @property
    def max_iteration(self):
        '''gets the maximum number of iterations'''
        return self.__max_iteration

    @max_iteration.setter
    def max_iteration(self, value):
        '''sets the maximum number of iterations'''
        assert isinstance(value, int)
        self.__max_iteration = value

    @property
    def update_objective_interval(self):
        return self.__update_objective_interval

    @update_objective_interval.setter
    def update_objective_interval(self, value):
        if isinstance(value, Integral):
            if value >= 0:
                self.__update_objective_interval = value
            else:
                raise ValueError('Update objective interval must be an integer >= 0')
        else:
            raise ValueError('Update objective interval must be an integer >= 0')

    def run(self, iterations=None, callbacks: list[Callback] | None=None, verbose=1, **kwargs):
        '''run n iterations and update the user with the callback if specified

        :param iterations: number of iterations to run. If not set the algorithm will
          run until max_iteration or until stop criterion is reached
        :param verbose: sets the verbosity output to screen: 0=quiet, 1=info, 2=debug
        :param callbacks: list of callables which are passed the current Algorithm
          object each iteration. Defaults to :code:`[ProgressCallback(verbose)]`.
        '''
        very_verbose = verbose>=2

        if callbacks is None:
            callbacks = [ProgressCallback(verbose=verbose)]
        # transform old-style callbacks into new
        callback = kwargs.get('callback', None)
        if callback is not None:
            callbacks += OldCallback(callback, verbose=very_verbose)

        if self.should_stop():
            print("Stop criterion has been reached.")
        if iterations is None:
            iterations = self.max_iteration

        if self.iteration == -1 and self.update_objective_interval>0:
            iterations+=1

        for _ in range(iterations):
            try:
                next(self)
                for callback in callbacks:
                    callback(self)
            except StopIteration:
                break

    def objective_to_string(self, verbose=False):
        # TODO: return a dict for `tqdm.set_postfix` instead
        # TODO: then deprecate this function
        # NOTE: see `verbose_header` for dict keys
        el = self.get_last_objective(return_all=verbose)
        if self.update_objective_interval == 0 or \
            self.iteration % self.update_objective_interval != 0:
            el = [ np.nan, np.nan, np.nan] if verbose else np.nan
        if isinstance (el, list):
            if np.isnan(el[0]):
                string = functools.reduce(lambda x,y: x+' {:>13s}'.format(''), el[:-1],'')
            elif not np.isnan(el[0]) and np.isnan(el[1]):
                string = ' {:>13.5e}'.format(el[0])
                string += ' {:>13s}'.format('')
            else:
                string = functools.reduce(lambda x,y: x+' {:>13.5e}'.format(y), el[:-1],'')
            if np.isnan(el[-1]):
                string += '{:>15s}'.format('')
            else:
                string += '{:>15.5e}'.format(el[-1])
        else:
            if np.isnan(el):
                string = '{:>20s}'.format('')
            else:
                string = "{:>20.5e}".format(el)
        return string

    def verbose_output(self, *_, **__):
        warn("use `run(callbacks=[ProgressCallback()])` instead", DeprecationWarning, stacklevel=2)

    def verbose_header(self, *_, **__):
        # el = self.get_last_objective(return_all=verbose)
        # Iter = self.iter_string
        # if type(el) == list:
        #     out = (f"{Iter:>9} {'Max ' + Iter:>10} {'Time/' + Iter:>13} {'Primal':>13} {'Dual':>13} {'Primal-Dual':>15}\n"
        #            f"{'':>9} {'':>10} {'[s]':>13} {'Objective':>13} {'Objective':>13} {'Gap':>15}")
        # else:
        #     out = (f"{Iter:>9} {'Max ' + Iter:>10} {'Time/' + Iter:>13} {'Objective':>20}\n"
        #            f"{'':>9} {'':>10} {'[s]':>13} {'':>20}")
        # if self.logger:
        #     self.logger.info(out)
        # return out
        warn("no longer needed", DeprecationWarning, stacklevel=2)
