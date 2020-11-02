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

from cil.framework import DataProcessor, AcquisitionData,\
     AcquisitionGeometry, ImageGeometry, ImageData
from ccpi.reconstruction.parallelbeam import alg as pbalg
import numpy

def setupCCPiGeometries(ig, ag, counter):
        Phantom_ccpi = ig.allocate(dimension_labels=[ImageGeometry.HORIZONTAL_X, 
                                                 ImageGeometry.HORIZONTAL_Y,
                                                 ImageGeometry.VERTICAL])    
    
        voxel_per_pixel = 1
        angles = ag.angles
        geoms = pbalg.pb_setup_geometry_from_image(Phantom_ccpi.as_array(),
                                                    angles,
                                                    voxel_per_pixel )
        
        pg = AcquisitionGeometry('parallel',
                                  '3D',
                                  angles,
                                  geoms['n_h'], 1.0,
                                  geoms['n_v'], 1.0 #2D in 3D is a slice 1 pixel thick
                                  )
        
        center_of_rotation = Phantom_ccpi.get_dimension_size('horizontal_x') / 2
        #ad = AcquisitionData(geometry=pg,dimension_labels=['angle','vertical','horizontal'])
        ad = pg.allocate(dimension_labels=[AcquisitionGeometry.ANGLE, 
                                           AcquisitionGeometry.VERTICAL,
                                           AcquisitionGeometry.HORIZONTAL])
        geoms_i = pbalg.pb_setup_geometry_from_acquisition(ad.as_array(),
                                                    angles,
                                                    center_of_rotation,
                                                    voxel_per_pixel )
        
        counter+=1
        
        if counter < 4:
            print (geoms, geoms_i)
            if (not ( geoms_i == geoms )):
                print ("not equal and {} {} {}".format(counter, geoms['output_volume_z'], geoms_i['output_volume_z']))
                X = max(geoms['output_volume_x'], geoms_i['output_volume_x'])
                Y = max(geoms['output_volume_y'], geoms_i['output_volume_y'])
                Z = max(geoms['output_volume_z'], geoms_i['output_volume_z'])
                return setupCCPiGeometries(X,Y,Z,angles, counter)
            else:
                print ("happy now {} {} {}".format(counter, geoms['output_volume_z'], geoms_i['output_volume_z']))
                
                return geoms
        else:
            return geoms_i


class CCPiForwardProjector(DataProcessor):
    '''Normalization based on flat and dark
    
    This processor read in a AcquisitionData and normalises it based on 
    the instrument reading with and without incident photons or neutrons.
    
    Input: AcquisitionData
    Parameter: 2D projection with flat field (or stack)
               2D projection with dark field (or stack)
    Output: AcquisitionDataSetn
    '''
    
    def __init__(self,
                 image_geometry       = None, 
                 acquisition_geometry = None,
                 output_axes_order    = None):
        if output_axes_order is None:
            # default ccpi projector image storing order
            output_axes_order = ['angle','vertical','horizontal']
        
        kwargs = {
                  'image_geometry'       : image_geometry, 
                  'acquisition_geometry' : acquisition_geometry,
                  'output_axes_order'    : output_axes_order,
                  'default_image_axes_order' : ['horizontal_x','horizontal_y','vertical'],
                  'default_acquisition_axes_order' : ['angle','vertical','horizontal'] 
                  }
        
        super(CCPiForwardProjector, self).__init__(**kwargs)
        
    def check_input(self, dataset):
        if dataset.number_of_dimensions == 3 or dataset.number_of_dimensions == 2:
            # sort in the order that this projector needs it
            return True
        else:
            raise ValueError("Expected input dimensions is 2 or 3, got {0}"\
                             .format(dataset.number_of_dimensions))

    def process(self, out=None):
        
        volume = self.get_input()
        volume_axes = volume.get_data_axes_order(new_order=self.default_image_axes_order)
        if not volume_axes == [0,1,2]:
            volume.array = numpy.transpose(volume.array, volume_axes)
        pixel_per_voxel = 1 # should be estimated from image_geometry and 
                            # acquisition_geometry
        if self.acquisition_geometry.geom_type == 'parallel':

            pixels = pbalg.pb_forward_project(volume.as_array(), 
                                                  self.acquisition_geometry.angles, 
                                                  pixel_per_voxel)
            
            out = self.acquisition_geometry.allocate(
                dimension_labels=self.output_axes_order)
            out_axes = out.get_data_axes_order(new_order=self.output_axes_order)
            if not out_axes == [0,1,2]:
                print ("transpose")
                pixels = numpy.transpose(pixels, out_axes)
            out.fill(pixels)
            
            return out
        else:
            raise ValueError('Cannot process cone beam')

class CCPiBackwardProjector(DataProcessor):
    '''Backward projector
    
    This processor reads in a AcquisitionData and performs a backward projection, 
    i.e. project to reconstruction space.
    Notice that it assumes that the center of rotation is in the middle
    of the horizontal axis: in case when that's not the case it can be chained 
    with the AcquisitionDataPadder.
    
    Input: AcquisitionData
    Parameter: 2D projection with flat field (or stack)
               2D projection with dark field (or stack)
    Output: AcquisitionDataSetn
    '''
    
    def __init__(self, 
                 image_geometry = None, 
                 acquisition_geometry = None,
                 output_axes_order=None):
        if output_axes_order is None:
            # default ccpi projector image storing order
            #output_axes_order = ['horizontal_x','horizontal_y','vertical']
            output_axes_order = ['vertical', 'horizontal_y','horizontal_x',]
        kwargs = {
                  'image_geometry'       : image_geometry, 
                  'acquisition_geometry' : acquisition_geometry,
                  'output_axes_order'    : output_axes_order,
                  'default_image_axes_order' : ['horizontal_x','horizontal_y','vertical'],
                  'default_acquisition_axes_order' : ['angle','vertical','horizontal'] 
                  }
        
        super(CCPiBackwardProjector, self).__init__(**kwargs)
        
    def check_input(self, dataset):
        if dataset.number_of_dimensions == 3 or dataset.number_of_dimensions == 2:
            
            return True
        else:
            raise ValueError("Expected input dimensions is 2 or 3, got {0}"\
                             .format(dataset.number_of_dimensions))

    def process(self, out=None):
        projections = self.get_input()
        projections_axes = projections.get_data_axes_order(new_order=self.default_acquisition_axes_order)
        if not projections_axes == [0,1,2]:
            projections.array = numpy.transpose(projections.array, projections_axes)
        
        pixel_per_voxel = 1 # should be estimated from image_geometry and acquisition_geometry
        image_geometry = ImageGeometry(voxel_num_x = self.acquisition_geometry.pixel_num_h,
                                       voxel_num_y = self.acquisition_geometry.pixel_num_h,
                                       voxel_num_z = self.acquisition_geometry.pixel_num_v)
        # input centered/padded acquisitiondata
        center_of_rotation = projections.get_dimension_size('horizontal') / 2
        
        if self.acquisition_geometry.geom_type == 'parallel':
            back = pbalg.pb_backward_project(
                         projections.as_array(), 
                         self.acquisition_geometry.angles, 
                         center_of_rotation, 
                         pixel_per_voxel
                         )
            out = self.image_geometry.allocate()
            out_axes = out.get_data_axes_order(new_order=self.output_axes_order)
            if not out_axes == [0,1,2]:
                back = numpy.transpose(back, out_axes)
            out.fill(back)
            
            return out
            
        else:
            raise ValueError('Cannot process cone beam')
            
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