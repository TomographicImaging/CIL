# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library developed by
#   Visual Analytics and Imaging System Group of the Science Technology
#   Facilities Council, STFC

#   Copyright 2019 Edoardo Pasca

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
import time

class Algorithm(object):
    '''Base class for iterative algorithms

      provides the minimal infrastructure.
      Algorithms are iterables so can be easily run in a for loop. They will
      stop as soon as the stop cryterion is met.
      The user is required to implement the set_up, __init__, update and
      and update_objective methods
      
      A courtesy method run is available to run additional n iterations. The method accepts
      a callback function that receives the current iteration number and the actual objective
      value and can be used to trigger print to screens and other user interactions.
   '''

    def __init__(self):
        self.iteration = 0
        self.stop_cryterion = 'max_iter'
        self.__max_iteration = 0
        self.__loss = []
        self.memopt = False
        self.timing = []
    def set_up(self, *args, **kwargs):
        raise NotImplementedError()
    def update(self):
        raise NotImplementedError()
    
    def should_stop(self):
        '''default stopping cryterion: number of iterations'''
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
            self.update()
            self.timing.append( time.time() - time0 )
            # TODO update every N iterations
            self.update_objective()
            self.iteration += 1
    def get_output(self):
        '''Returns the solution found'''
        return self.x
    def get_current_loss(self):
        '''Returns the current value of the loss function'''
        return self.__loss[-1]
    def get_current_objective(self):
        '''alias to get_current_loss'''
        return self.get_current_loss()
    def update_objective(self):
        '''calculates the current objective'''
        raise NotImplementedError()
    @property
    def loss(self):
        '''returns the list of the values of the objective during the iteration'''
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
    def run(self, iterations, callback=None):
        '''run n iterations and update the user with the callback if specified'''
        self.max_iteration += iterations
        for _ in self:
            if callback is not None:
                callback(self.iteration, self.get_current_loss())
    
