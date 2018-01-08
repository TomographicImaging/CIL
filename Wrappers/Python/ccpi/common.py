# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library developed by
#   Visual Analytics and Imaging System Group of the Science Technology
#   Facilities Council, STFC

#   Copyright 2018 Edoardo Pasca

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
import abc
import numpy
import os
import sys
import time

if sys.version_info[0] >= 3 and sys.version_info[1] >= 4:
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})

def find_key(dic, val):
    """return the key of dictionary dic given the value"""
    return [k for k, v in dic.items() if v == val][0]

class CCPiBaseClass(object):
    def __init__(self, **kwargs):
        self.acceptedInputKeywords = []
        self.pars = {}
        self.debug = True
    
    def setParameter(self, **kwargs):
        '''set named parameter for the reconstructor engine
        
        raises Exception if the named parameter is not recognized
        
        '''
        for key , value in kwargs.items():
            if key in self.acceptedInputKeywords:
                self.pars[key] = value
            else:
                raise KeyError('Wrong parameter "{0}" for {1}'.format(key, 
                               self.__class__.__name__ ))
    # setParameter

    def getParameter(self, key):
        if type(key) is str:
            if key in self.acceptedInputKeywords:
                return self.pars[key]
            else:
                raise KeyError('Unrecongnised parameter: {0} '.format(key) )
        elif type(key) is list:
            outpars = []
            for k in key:
                outpars.append(self.getParameter(k))
            return outpars
        else:
            raise Exception('Unhandled input {0}' .format(str(type(key))))
    #getParameter
    
    def log(self, msg):
        if self.debug:
            print ("{0}: {1}".format(self.__class__.__name__, msg))
            
class DataSet(ABC):
    '''Abstract class to hold data'''
    
    def __init__ (self, array, deep_copy=True, dimension_labels=None, 
                  **kwargs):
        '''Holds the data'''
        
        self.shape = numpy.shape(array)
        self.number_of_dimensions = len (self.shape)
        self.dimension_labels = {}
        
        if dimension_labels is not None and \
           len (dimension_labels) == self.number_of_dimensions:
            for i in range(self.number_of_dimensions):
                self.dimension_labels[i] = dimension_labels[i]
        else:
            for i in range(self.number_of_dimensions):
                self.dimension_labels[i] = 'dimension_{0:02}'.format(i)
            
        if deep_copy:
            self.array = array[:]
        else:
            self.array = array
        
    def as_array(self, dimensions=None):
        if dimensions is None:
            return self.array
        else:
            # check that all the requested dimensions are in the array
            # this is done by checking the dimension_labels
            proceed = True
            axis_order = []
            if type(dimensions) == list:
                for dl in dimensions:
                    if dl not in self.dimension_labels.values():
                        proceed = False
                        break
                    else:
                        axis_order.append(find_key(self.dimension_labels, dl))
                print (axis_order)        
                # transpose the array and slice away the unwanted data
                unwanted_dimensions = self.dimension_labels.copy()
                for ax in axis_order:
                    unwanted_dimensions.pop(ax)
                new_shape = []
                #for i in range(axis_order):
                #    new_shape.append(self.shape(axis_order[i]))
                new_shape = [self.shape[ax] for ax in axis_order]
                return numpy.reshape( 
                        numpy.delete( self.array , unwanted_dimensions.keys() ) ,
                        new_shape
                        )
                #return numpy.transpose(self.array, new_shape)
                        
                    
        
            
            
if __name__ == '__main__':
    shape = (2,3,4,5)
    size = shape[0]
    for i in range(1, len(shape)):
        size = size * shape[i]
    a = numpy.asarray([i for i in range( size )])
    a = numpy.reshape(a, shape)
    ds = DataSet(a, False, ['X', 'Y','Z' ,'W'])
    b = ds.as_array(['Z' ,'W'])    
        
        
        
        