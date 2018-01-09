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
        '''Returns the DataSet as Numpy Array
        
        Returns the pointer to the array if dimensions is not set.
        If dimensions is set, it first creates a new DataSet with the subset
        and then it returns the pointer to the array'''
        if dimensions is not None:
            return self.subset(dimensions).as_array()
        return self.array
        
    def subset(self, dimensions=None):
        '''Creates a DataSet containing a subset of self according to the 
        labels in dimensions'''
        if dimensions is None:
            return self.array
        else:
            # check that all the requested dimensions are in the array
            # this is done by checking the dimension_labels
            proceed = True
            unknown_key = ''
            # axis_order contains the order of the axis that the user wants
            # in the output DataSet
            axis_order = []
            if type(dimensions) == list:
                for dl in dimensions:
                    if dl not in self.dimension_labels.values():
                        proceed = False
                        unknown_key = dl
                        break
                    else:
                        axis_order.append(find_key(self.dimension_labels, dl))
                if not proceed:
                    raise KeyError('Unknown key specified {0}'.format(dl))
                    
                # slice away the unwanted data from the array
                unwanted_dimensions = self.dimension_labels.copy()
                left_dimensions = []
                for ax in sorted(axis_order):
                    this_dimension = unwanted_dimensions.pop(ax)
                    left_dimensions.append(this_dimension)
                #print ("unwanted_dimensions {0}".format(unwanted_dimensions))
                #print ("left_dimensions {0}".format(left_dimensions))
                #new_shape = [self.shape[ax] for ax in axis_order]
                #print ("new_shape {0}".format(new_shape))
                command = "self.array"
                for i in range(self.number_of_dimensions):
                    if self.dimension_labels[i] in unwanted_dimensions.values():
                        command = command + "[0]"
                    else:
                        command = command + "[:]"
                #print ("command {0}".format(command))
                cleaned = eval(command)
                # cleaned has collapsed dimensions in the same order of
                # self.array, but we want it in the order stated in the 
                # "dimensions". 
                # create axes order for numpy.transpose
                axes = []
                for key in dimensions:
                    #print ("key {0}".format( key))
                    for i in range(len( left_dimensions )):
                        ld = left_dimensions[i]
                        #print ("ld {0}".format( ld))
                        if ld == key:
                            axes.append(i)
                #print ("axes {0}".format(axes))
                
                cleaned = numpy.transpose(cleaned, axes).copy()
                
                return DataSet(cleaned , True, dimensions)
                    
                    
        
            
            
if __name__ == '__main__':
    shape = (2,3,4,5)
    size = shape[0]
    for i in range(1, len(shape)):
        size = size * shape[i]
    a = numpy.asarray([i for i in range( size )])
    a = numpy.reshape(a, shape)
    ds = DataSet(a, False, ['X', 'Y','Z' ,'W'])
    print ("ds label {0}".format(ds.dimension_labels))
    subset = ['W' ,'X']
    b = ds.subset( subset )
    print ("b label {0} shape {1}".format(b.dimension_labels, 
           numpy.shape(b.as_array())))
    c = ds.as_array(['Z','W'])
    
    
        
        
        
        