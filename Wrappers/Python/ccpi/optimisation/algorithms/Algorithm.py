# -*- coding: utf-8 -*-
#========================================================================
# Copyright 2019 Science Technology Facilities Council
# Copyright 2019 University of Manchester
#
# This work is part of the Core Imaging Library developed by Science Technology
# Facilities Council and University of Manchester
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#=========================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time, functools
from numbers import Integral, Number
import logging

class DataContainerWithHistory(object):
    def __init__(self, geometry, initialisation=None):
        if isinstance(initialisation, Number):
            self.datacontainers = [
                geometry.allocate(0),
                geometry.allocate(initialisation),
            ]
        else:
            self.datacontainers = [
                geometry.allocate(0),
                geometry.allocate(0)
            ]
            try:
                self.datacontainers[1].fill(initialisation.as_array())
            except:
                pass

    @property
    def previous(self):
        return self.datacontainers[1]
    @property
    def current(self):
        return self.datacontainers[0]
    def update_indices(self):
        self.datacontainers.reverse()

    @previous.setter
    def previous(self, value):
        self.datacontainers[1] = value.copy()
    @current.setter
    def current(self, value):
        self.datacontainers[0] = value.copy()

class Algorithm(object):
    '''Base class for iterative algorithms

      provides the minimal infrastructure.

      Algorithms are iterables so can be easily run in a for loop. They will
      stop as soon as the stop cryterion is met.
      The user is required to implement the :code:`set_up`, :code:`__init__`, :code:`update` and
      and :code:`update_objective` methods
      
      A courtesy method :code:`run` is available to run :code:`n` iterations. The method accepts
      a :code:`callback` function that receives the current iteration number and the actual objective
      value and can be used to trigger print to screens and other user interactions. The :code:`run`
      method will stop when the stopping cryterion is met. 
   '''

    def __init__(self, **kwargs):
        '''Constructor
        
        Set the minimal number of parameters:
        
        
        :param max_iteration: maximum number of iterations
        :type max_iteration: int, optional, default 0
        :param update_objectice_interval: the interval every which we would save the current\
                                       objective. 1 means every iteration, 2 every 2 iteration\
                                       and so forth. This is by default 1 and should be increased\
                                       when evaluating the objective is computationally expensive.
        :type update_objective_interval: int, optional, default 1
        :param log_file: log verbose output to file
        :type log_file: str, optional, default None
        '''
        self.iteration = 0
        self.__max_iteration = kwargs.get('max_iteration', 0)
        self.__loss = []
        self.memopt = False
        self.configured = False
        self.timing = []
        self._iteration = []
        self.update_objective_interval = kwargs.get('update_objective_interval', 1)
        # self.x = None
        self.iter_string = 'Iter'
        self.logger = None
        self.__set_up_logger(kwargs.get('log_file', None))

    def set_up(self, *args, **kwargs):
        '''Set up the algorithm'''
        raise NotImplementedError()
    def update(self):
        '''A single iteration of the algorithm'''
        raise NotImplementedError()
    
    def should_stop(self):
        '''default stopping cryterion: number of iterations
        
        The user can change this in concrete implementatition of iterative algorithms.'''
        return self.max_iteration_stop_cryterion()
    
    def __set_up_logger(self, fname):
        """Set up the logger if desired"""
        if fname:
            print("Will output results to: " +  fname)
            handler = logging.FileHandler(fname)
            self.logger = logging.getLogger("obj_fn")
            self.logger.setLevel(logging.INFO)
            self.logger.addHandler(handler)
    
    def max_iteration_stop_cryterion(self):
        '''default stop cryterion for iterative algorithm: max_iteration reached'''
        return self.iteration >= self.max_iteration
    def __iter__(self):
        '''Algorithm is an iterable'''
        return self
    def next(self):
        '''Algorithm is an iterable
        
        python2 backwards compatibility'''
        return self.__next__()
    def __next__(self):
        '''Algorithm is an iterable
        
        calling this method triggers update and update_objective
        '''
        if self.should_stop():
            raise StopIteration()
        else:
            time0 = time.time()
            if not self.configured:
                raise ValueError('Algorithm not configured correctly. Please run set_up.')
            if self.iteration == 0:
                self.update_objective()
                self._iteration.append(self.iteration)
            self.update()
            self.timing.append( time.time() - time0 )
            if self.iteration > 0 and self.iteration % self.update_objective_interval == 0:
                self.update_objective()
            self.iteration += 1
            self.update_indices()

    def update_indices(self):
        '''Update the indices of DataContainerWithHistory
        
        The concrete algorithm needs to call update_indices for the DataContainerWithHistory that exists:
        if only one DataContainerWithHistory exists with name self._x, then you need to 

        self._x.update_indices()
        '''
        pass
    def get_output(self):
        '''Returns the solution found'''
        return self.x
    
    def get_last_loss(self, **kwargs):
        '''Returns the last stored value of the loss function
        
        if update_objective_interval is 1 it is the value of the objective at the current
        iteration. If update_objective_interval > 1 it is the last stored value. 
        '''
        return_all =  kwargs.get('return_all', False)
        objective = self.__loss[-1]
        if return_all:
            return list(objective)
        if isinstance(objective, list):
            return objective[0]
        else:
            return objective
    def get_last_objective(self, **kwargs):
        '''alias to get_last_loss'''
        return self.get_last_loss(**kwargs)
    def update_objective(self):
        '''calculates the objective with the current solution'''
        raise NotImplementedError()
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
            if value >= 1:
                self.__update_objective_interval = value
            else:
                raise ValueError('Update objective interval must be an integer >= 1')
        else:
            raise ValueError('Update objective interval must be an integer >= 1')
    def run(self, iterations=None, verbose=True, callback=None, very_verbose=False):
        '''run n iterations and update the user with the callback if specified
        
        :param iterations: number of iterations to run. If not set the algorithm will
          run until max_iteration or until stop criterion is reached
        :param verbose: toggles verbose output to screen
        :param callback: is a function that receives: current iteration number, 
          last objective function value and the current solution
        :param very_verbose: bool, useful for algorithms with primal and dual objectives (PDHG), 
                            prints to screen both primal and dual
        '''
        if self.should_stop():
            print ("Stop cryterion has been reached.")
        i = 0
        if verbose:
            print (self.verbose_header(very_verbose))
        
        for _ in self:
            # __next__ is called

            # the following code is just for displaying purposes of the status of the minimisation

            # self.iteration is incremented in __next__, so now we have 
            # self.iteration is one iteration larger than what we want to display
            self.iteration -= 1
            if (self.iteration) % self.update_objective_interval == 0: 
                if verbose:
                    print (self.verbose_output(very_verbose))
            if callback is not None:
                callback(self.iteration, self.get_last_objective(return_all=very_verbose), self.x)
            
            # restore self.iteration value to what it should be
            self.iteration += 1

            # check if run has to stop
            i += 1
            if i == iterations:
                break
            
        if verbose:
            if self.iteration != self._iteration[-1]:
                # if the objective hasn't already been calculated as not on 
                # the right update_objective_interval 
                self.update_objective()
                
            start = 3 # I don't understand why this
            bars = ['-' for i in range(start+9+10+13+20)]
            if (very_verbose):
                bars = ['-' for i in range(start+9+10+13+13+13+15)]
            # print a nice ---- with proper length at the end
            # print (functools.reduce(lambda x,y: x+y, bars, ''))
            out = "{}\n{}\n{}\n".format(functools.reduce(lambda x,y: x+y, bars, '') ,
                                        self.verbose_output(very_verbose),
                                        "Stop criterion has been reached.")
            print (out)
            # print (self.verbose_output(very_verbose))
            # print ("Stop criterion has been reached.")
            # Print to log file if desired
            if self.logger:
                self.logger.info(out)

        

    def verbose_output(self, verbose=False):
        '''Creates a nice tabulated output'''
        timing = self.timing[-self.update_objective_interval-1:-1]
        self._iteration.append(self.iteration)
        if len (timing) == 0:
            t = 0
        else:
            t = sum(timing)/len(timing)
        out = "{:>9} {:>10} {:>13} {}".format(
                 self.iteration, 
                 self.max_iteration,
                 "{:.3f}".format(t), 
                 self.objective_to_string(verbose)
               )
        # Print to log file if desired
        if self.logger:
            self.logger.info(out)
        return out

    def objective_to_string(self, verbose=False):
        el = self.get_last_objective(return_all=verbose)
        if type(el) == list:
            string = functools.reduce(lambda x,y: x+' {:>13.5e}'.format(y), el[:-1],'')
            string += '{:>15.5e}'.format(el[-1])
        else:
            string = "{:>20.5e}".format(el)
        return string
    def verbose_header(self, verbose=False):
        el = self.get_last_objective(return_all=verbose)
        if type(el) == list:
            out = "{:>9} {:>10} {:>13} {:>13} {:>13} {:>15}\n".format(self.iter_string, 
                                                      'Max {}'.format(self.iter_string),
                                                      'Time/{}'.format(self.iter_string),
                                                      'Primal' , 'Dual', 
                                                      'Primal-Dual')
            out += "{:>9} {:>10} {:>13} {:>13} {:>13} {:>15}".format('', 
                                                      '',
                                                      '[s]',
                                                      'Objective' , 
                                                      'Objective', 
                                                      'Gap')
        else:
            out = "{:>9} {:>10} {:>13} {:>20}\n".format(self.iter_string, 
                                                      'Max {}'.format(self.iter_string),
                                                      'Time/{}'.format(self.iter_string),
                                                      'Objective')
            out += "{:>9} {:>10} {:>13} {:>20}".format('', 
                                                      '',
                                                      '[s]',
                                                      '')
        # Print to log file if desired
        if self.logger:
            self.logger.info(out)

        return out
