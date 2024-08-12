#  Copyright 2018 United Kingdom Research and Innovation
#  Copyright 2018 The University of Manchester
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
import weakref

from .data_container import DataContainer


def find_key(dic, val):
    """return the key of dictionary dic given the value"""
    return [k for k, v in dic.items() if v == val][0]


class Processor(object):

    '''Defines a generic DataContainer processor

    accepts a DataContainer as input
    returns a DataContainer
    `__setattr__` allows additional attributes to be defined

    `store_output` boolian defining whether a copy of the output is stored. Default is False.
    If no attributes are modified get_output will return this stored copy bypassing `process`
    '''

    def __init__(self, **attributes):
        if not 'store_output' in attributes.keys():
            attributes['store_output'] = False

        attributes['output'] = None
        attributes['shouldRun'] = True
        attributes['input'] = None

        for key, value in attributes.items():
            self.__dict__[key] = value

    def __setattr__(self, name, value):
        if name == 'input':
            self.set_input(value)
        elif name in self.__dict__.keys():

            self.__dict__[name] = value

            if name == 'shouldRun':
                pass
            elif name == 'output':
                self.__dict__['shouldRun'] = False
            else:
                self.__dict__['shouldRun'] = True
        else:
            raise KeyError('Attribute {0} not found'.format(name))

    def set_input(self, dataset):
        """
        Set the input data to the processor

        Parameters
        ----------
        input : DataContainer
            The input DataContainer
        """

        if issubclass(type(dataset), DataContainer):
            if self.check_input(dataset):
                self.__dict__['input'] = weakref.ref(dataset)
                self.__dict__['shouldRun'] = True
            else:
                raise ValueError('Input data not compatible')
        else:
            raise TypeError("Input type mismatch: got {0} expecting {1}" \
                            .format(type(dataset), DataContainer))


    def check_input(self, dataset):
        '''Checks parameters of the input DataContainer

        Should raise an Error if the DataContainer does not match expectation, e.g.
        if the expected input DataContainer is 3D and the Processor expects 2D.
        '''
        raise NotImplementedError('Implement basic checks for input DataContainer')

    def get_output(self, out=None):
        """
        Runs the configured processor and returns the processed data

        Parameters
        ----------
        out : DataContainer, optional
           Fills the referenced DataContainer with the processed data and suppresses the return

        Returns
        -------
        DataContainer
            The processed data. Suppressed if `out` is passed
        """
        if self.output is None or self.shouldRun:
            if out is None:
                out = self.process()
            else:
                self.process(out=out)

            if self.store_output:
                self.output = out.copy()

            return out

        else:
            return self.output.copy()


    def set_input_processor(self, processor):
        if issubclass(type(processor), DataProcessor):
            self.__dict__['input'] =  weakref.ref(processor)
        else:
            raise TypeError("Input type mismatch: got {0} expecting {1}"\
                            .format(type(processor), DataProcessor))

    def get_input(self):
        '''returns the input DataContainer

        It is useful in the case the user has provided a DataProcessor as
        input
        '''
        if self.input() is None:
            raise ValueError("Input has been deallocated externally")
        elif issubclass(type(self.input()), DataProcessor):
            dsi = self.input().get_output()
        else:
            dsi = self.input()
        return dsi

    def process(self, out=None):
        raise NotImplementedError('process must be implemented')

    def __call__(self, x, out=None):

        self.set_input(x)

        if out is None:
            out = self.get_output()
        else:
            self.get_output(out=out)

        return out


class DataProcessor(Processor):
    '''Basically an alias of Processor Class'''
    pass

class DataProcessor23D(DataProcessor):
    '''Regularizers DataProcessor
    '''

    def check_input(self, dataset):
        '''Checks number of dimensions input DataContainer

        Expected input is 2D or 3D
        '''
        if dataset.number_of_dimensions == 2 or \
           dataset.number_of_dimensions == 3:
               return True
        else:
            raise ValueError("Expected input dimensions is 2 or 3, got {0}"\
                             .format(dataset.number_of_dimensions))

###### Example of DataProcessors

class AX(DataProcessor):
    '''Example DataProcessor
    The AXPY routines perform a vector multiplication operation defined as

    y := a*x
    where:

    a is a scalar

    x a DataContainer.
    '''

    def __init__(self):
        kwargs = {'scalar':None,
                  'input':None,
                  }

        #DataProcessor.__init__(self, **kwargs)
        super(AX, self).__init__(**kwargs)

    def check_input(self, dataset):
        return True

    def process(self, out=None):

        dsi = self.get_input()
        a = self.scalar
        if out is None:
            y = DataContainer(a * dsi.as_array(), True,
                              dimension_labels=dsi.dimension_labels)
            #self.setParameter(output_dataset=y)
            return y
        else:
            out.fill(a * dsi.as_array())


###### Example of DataProcessors

class CastDataContainer(DataProcessor):
    '''Example DataProcessor
    Cast a DataContainer array to a different type.

    y := a*x
    where:

    a is a scalar

    x a DataContainer.
    '''

    def __init__(self, dtype=None):
        kwargs = {'dtype':dtype,
                  'input':None,
                  }

        #DataProcessor.__init__(self, **kwargs)
        super(CastDataContainer, self).__init__(**kwargs)

    def check_input(self, dataset):
        return True

    def process(self, out=None):

        dsi = self.get_input()
        dtype = self.dtype
        if out is None:
            y = numpy.asarray(dsi.as_array(), dtype=dtype)

            return type(dsi)(numpy.asarray(dsi.as_array(), dtype=dtype),
                                dimension_labels=dsi.dimension_labels )
        else:
            out.fill(numpy.asarray(dsi.as_array(), dtype=dtype))

class PixelByPixelDataProcessor(DataProcessor):
    '''Example DataProcessor

    This processor applies a python function to each pixel of the DataContainer

    f is a python function

    x a DataSet.
    '''

    def __init__(self):
        kwargs = {'pyfunc':None,
                  'input':None,
                  }
        #DataProcessor.__init__(self, **kwargs)
        super(PixelByPixelDataProcessor, self).__init__(**kwargs)

    def check_input(self, dataset):
        return True

    def process(self, out=None):

        pyfunc = self.pyfunc
        dsi = self.get_input()

        eval_func = numpy.frompyfunc(pyfunc,1,1)


        y = DataContainer(eval_func(dsi.as_array()), True,
                          dimension_labels=dsi.dimension_labels)
        return y
