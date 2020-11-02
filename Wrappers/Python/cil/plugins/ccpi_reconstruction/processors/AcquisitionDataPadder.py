# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 13:43:52 2019

@author: ofn77899
"""
from cil.framework import DataProcessor, DataContainer, AcquisitionData,\
 AcquisitionGeometry, ImageGeometry, ImageData
from ccpi.reconstruction.parallelbeam import alg as pbalg
import numpy

class AcquisitionDataPadder(DataProcessor):
    '''Normalization based on flat and dark
    
    This processor read in a AcquisitionData and normalises it based on 
    the instrument reading with and without incident photons or neutrons.
    
    Input: AcquisitionData
    Parameter: 2D projection with flat field (or stack)
               2D projection with dark field (or stack)
    Output: AcquisitionDataSetn
    '''
    
    def __init__(self, 
                 center_of_rotation   = None,
                 acquisition_geometry = None,
                 pad_value            = 1e-5):
        kwargs = {
                  'acquisition_geometry' : acquisition_geometry, 
                  'center_of_rotation'   : center_of_rotation,
                  'pad_value'            : pad_value
                  }
        
        super(AcquisitionDataPadder, self).__init__(**kwargs)
        
    def check_input(self, dataset):
        if self.acquisition_geometry is None:
            self.acquisition_geometry = dataset.geometry
        if dataset.number_of_dimensions == 3:
               return True
        else:
            raise ValueError("Expected input dimensions is 2 or 3, got {0}"\
                             .format(dataset.number_of_dimensions))

    def process(self, out=None):
        projections = self.get_input()
        w = projections.get_dimension_size('horizontal')
        print ("horizontal ", w)
        delta = w - 2 * self.center_of_rotation
               
        padded_width = int (
                numpy.ceil(abs(delta)) + w
                )
        delta_pix = padded_width - w
        
        voxel_per_pixel = 1
        geom = pbalg.pb_setup_geometry_from_acquisition(projections.as_array(),
                                            self.acquisition_geometry.angles,
                                            self.center_of_rotation,
                                            voxel_per_pixel )
        
        print ("geom", geom)
        padded_geometry = self.acquisition_geometry.clone()
        
        padded_geometry.pixel_num_h = geom['n_h']
        padded_geometry.pixel_num_v = geom['n_v']
        
        delta_pix_h = padded_geometry.pixel_num_h - self.acquisition_geometry.pixel_num_h
        delta_pix_v = padded_geometry.pixel_num_v - self.acquisition_geometry.pixel_num_v
        
        if delta_pix_h == 0:
            delta_pix_h = delta_pix
            padded_geometry.pixel_num_h = padded_width
        #initialize a new AcquisitionData with values close to 0
        out = AcquisitionData(geometry=padded_geometry)
        out = out + self.pad_value
        
        
        #pad in the horizontal-vertical plane -> slice on angles
        if delta > 0:
            #pad left of middle
            command = "out.array["
            for i in range(out.number_of_dimensions):
                if out.dimension_labels[i] == 'horizontal':
                    value = '{0}:{1}'.format(delta_pix_h, delta_pix_h+w)
                    command = command + str(value)
                else:
                    if out.dimension_labels[i] == 'vertical' :
                        value = '{0}:'.format(delta_pix_v)
                        command = command + str(value)
                    else:
                        command = command + ":"
                if i < out.number_of_dimensions -1:
                    command = command + ','
            command = command + '] = projections.array'
            #print (command)    
        else:
            #pad right of middle
            command = "out.array["
            for i in range(out.number_of_dimensions):
                if out.dimension_labels[i] == 'horizontal':
                    value = '{0}:{1}'.format(0, w)
                    command = command + str(value)
                else:
                    if out.dimension_labels[i] == 'vertical' :
                        value = '{0}:'.format(delta_pix_v)
                        command = command + str(value)
                    else:
                        command = command + ":"
                if i < out.number_of_dimensions -1:
                    command = command + ','
            command = command + '] = projections.array'
            #print (command)    
            #cleaned = eval(command)
        exec(command)
        return out
