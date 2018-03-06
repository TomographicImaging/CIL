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
#   limitations under the License

from ccpi.framework import DataSetProcessor, DataSet, VolumeData, SinogramData
import numpy
import h5py

class NormalizationDataSetProcessor(DataSetProcessor):
    '''Normalization based on flat and dark
    
    This processor read in a SinogramDataSet and normalises it based on 
    the instrument reading with and without incident photons or neutrons.
    
    Input: SinogramDataSet
    Parameter: 2D projection with flat field (or stack)
               2D projection with dark field (or stack)
    Output: SinogramDataSet
    '''
    
    def __init__(self):
        kwargs = {
                  'flat_field'  :None, 
                  'dark_field'  :None,
                  # very small number. Used when there is a division by zero
                  'tolerance'   : 1e-5
                  }
        
        #DataSetProcessor.__init__(self, **kwargs)
        super(NormalizationDataSetProcessor, self).__init__(**kwargs)
    
    def checkInput(self, dataset):
        if dataset.number_of_dimensions == 3:
               return True
        else:
            raise ValueError("Expected input dimensions is 2 or 3, got {0}"\
                             .format(dataset.number_of_dimensions))
    
    def setDarkField(self, df):
        if type(df) is numpy.ndarray:
            if len(numpy.shape(df)) == 3:
                raise ValueError('Dark Field should be 2D')
            elif len(numpy.shape(df)) == 2:
                self.dark_field = df
        elif issubclass(type(df), DataSet):
            self.dark_field = self.setDarkField(df.as_array())
    
    def setFlatField(self, df):
        if type(df) is numpy.ndarray:
            if len(numpy.shape(df)) == 3:
                raise ValueError('Flat Field should be 2D')
            elif len(numpy.shape(df)) == 2:
                self.flat_field = df
        elif issubclass(type(df), DataSet):
            self.flat_field = self.setDarkField(df.as_array())
    
    @staticmethod
    def normalizeProjection(projection, flat, dark, tolerance):
        a = (projection - dark)
        b = (flat-dark)
        with numpy.errstate(divide='ignore', invalid='ignore'):
            c = numpy.true_divide( a, b )
            c[ ~ numpy.isfinite( c )] = tolerance # set to not zero if 0/0 
        return c
    
    def process(self):
        
        projections = self.getInput()
        dark = self.dark_field
        flat = self.flat_field
        
        if not (projections.shape[1:] == dark.shape and \
           projections.shape[1:] == flat.shape):
            raise ValueError('Flats/Dark and projections size do not match.')
            
               
        a = numpy.asarray(
                [ NormalizationDataSetProcessor.normalizeProjection(
                        projection, flat, dark, self.tolerance) \
                 for projection in projections.as_array() ]
                )
        y = DataSet( a , True, 
                    dimension_labels=projections.dimension_labels )
        return y
    
    
def loadNexus(filename):
    '''Load a dataset stored in a NeXuS file (HDF5)'''
    ###########################################################################
    ## Load a dataset
    nx = h5py.File(filename, "r")
    
    data = nx.get('entry1/tomo_entry/data/rotation_angle')
    angles = numpy.zeros(data.shape)
    data.read_direct(angles)
    
    data = nx.get('entry1/tomo_entry/data/data')
    stack = numpy.zeros(data.shape)
    data.read_direct(stack)
    data = nx.get('entry1/tomo_entry/instrument/detector/image_key')
    
    itype = numpy.zeros(data.shape)
    data.read_direct(itype)
    # 2 is dark field
    darks = [stack[i] for i in range(len(itype)) if itype[i] == 2 ]
    dark = darks[0]
    for i in range(1, len(darks)):
        dark += darks[i]
    dark = dark / len(darks)
    #dark[0][0] = dark[0][1]
    
    # 1 is flat field
    flats = [stack[i] for i in range(len(itype)) if itype[i] == 1 ]
    flat = flats[0]
    for i in range(1, len(flats)):
        flat += flats[i]
    flat = flat / len(flats)
    #flat[0][0] = dark[0][1]
    
    
    # 0 is projection data
    proj = [stack[i] for i in range(len(itype)) if itype[i] == 0 ]
    angle_proj = [angles[i] for i in range(len(itype)) if itype[i] == 0 ]
    angle_proj = numpy.asarray (angle_proj)
    angle_proj = angle_proj.astype(numpy.float32)
    
    return angle_proj , numpy.asarray(proj) , dark, flat
    
    
    
if __name__ == '__main__':
    angles, proj, dark, flat = loadNexus('../../../data/24737_fd.nxs')
    
    sino = SinogramData( proj )
        
    
    normalizer = NormalizationDataSetProcessor()
    normalizer.setInput(sino)
    normalizer.setFlatField(flat)
    normalizer.setDarkField(dark)
    norm = normalizer.getOutput()
    print ("Processor min {0} max {1}".format(norm.as_array().min(), norm.as_array().max()))
    
    norm1 = numpy.asarray(
            [NormalizationDataSetProcessor.normalizeProjection( p, flat, dark, 1e-5 ) 
            for p in proj]
            )
    
    print ("Numpy min {0} max {1}".format(norm1.min(), norm1.max()))