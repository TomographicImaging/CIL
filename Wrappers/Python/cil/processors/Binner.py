# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library (CIL) developed by CCPi 
#   (Collaborative Computational Project in Tomographic Imaging), with 
#   substantial contributions by UKRI-STFC and University of Manchester.

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
from cil.framework import DataProcessor, AcquisitionData, ImageData, DataContainer, AcquisitionGeometry, ImageGeometry
import numpy as np

class Binner(DataProcessor):

    """This creates a Binner processor.
    
    The processor will crop the data, and then average together n input pixels along a dimension from the starting index.

    The output will be a data container with the data, and geometry updated to reflect the operation.

    Parameters
    ----------

    roi : dict
        The region-of-interest to bin {'axis_name1':(start,stop,step), 'axis_name2':(start,stop,step)}
        The `key` being the axis name to apply the processor to, the `value` holding a tuple containing the ROI description

        Start: Starting index of input data. Must be an integer, or `None` defaults to index 0.
        Stop: Stopping index of input data. Must be an integer, or `None` defaults to index N.
        Step: Number of pixels to average together. Must be an integer or `None` defaults to 1.


    Example
    -------
    from cil.processors import Binner
    roi = {'horizontal':(10,-10,2),'vertical':(10,-10,2)}
    processor = Binner(roi)
    processor.set_input(data)
    data_binned = processor.get_output()


    Example
    -------
    from cil.processors import Binner
    roi = {'horizontal':(10,-10,2),'vertical':(10,-10,2)}
    processor = Binner(roi)
    processor.set_input(data.geometry)
    geometry_binned = processor.get_output()


    Note
    ----
    The indices provided are start inclusive, stop exclusive.

    All elements along a dimension will be included if the axis does not appear in the roi dictionary, or if passed as {'axis_name',-1}
    
    If only one number is provided, then it is interpreted as Stop. i.e. {'axis_name1':(stop)}
    If two numbers are provided, then they are interpreted as Start and Stop  i.e. {'axis_name1':(start, stop)}

    Negative indexing can be used to specify the index. i.e. {'axis_name1':(10, -10)} will crop the dimension symmetrically
    
    If Stop - Start is not multiple of Step, then 
    the resulted dimension will have (Stop - Start) // Step 
    elements, i.e. (Stop - Start) % Step elements will be ignored

    """


    def __init__(self,
                 roi = None):

        kwargs = {'roi_input': roi, 'roi_ordered':None, 'data_array': False, 'geometry': None, 'processed_dims':None}

        super(Binner, self).__init__(**kwargs)
    

    def check_input(self, data):

        if issubclass(data, (ImageData,AcquisitionData)):
            self.data_array = True
            self.geometry = data.geometry

        elif isinstance(data, DataContainer):
            self.data_array = True
            self.geometry = None

        elif isinstance(data, (ImageGeometry, AcquisitionGeometry)):
            self.data_array = False
            self.geometry = data

        else:
            raise TypeError('Processor supports following data types:\n' +
                            ' - ImageData\n - AcquisitionData\n - DataContainer\n - ImageGeometry\n - AcquisitionGeometry')

        if self.data_array:
            if data.dtype != np.float32:
                raise TypeError("Expected float32")

        if (self.roi_input == None):
            raise ValueError('Please, specify roi')

        for key in self.roi_input.keys():
            if key not in data.dimension_labels:
                raise ValueError('Wrong label is specified for roi, expected one of {}.'.format(data.dimension_labels))
        
        
        return True 


    def _bin_acquisition_geometry(self):

        geometry_new = self.geometry.copy()

        processed_dims = self.processed_dims.copy()

        #deal with vertical first as it may change the geometry type
        if 'vertical' in self.geometry.dimension_labels:
            vert_ind = self.geometry.dimension_labels.index('vertical')
            if self.processed_dims[vert_ind]:
                roi = self.roi_ordered[vert_ind]
                n_elements = len(roi)

                if n_elements > 1:
                    geometry_new.config.panel.num_pixels[1] = n_elements
                else:
                    geometry_new = geometry_new.get_slice(vertical = (roi.start + roi.step/2))

                geometry_new.config.panel.pixel_size[1] *= roi.step
                processed_dims[vert_ind] = False #set to false locally


        for i, axis  in enumerate(self.geometry.dimension_labels):

            if not processed_dims[i]:
                continue
            
            roi = self.roi_ordered[i]
            n_elements = len(roi)

            if axis == 'channel':
                geometry_new.set_channels(num_channels=n_elements)

            elif axis == 'angle':
                shape = (n_elements, roi.step)
                geometry_new.config.angles.angle_data = self.geometry.angles[roi.start:roi.start+n_elements*roi.step].reshape(shape).mean(1)
                
            elif axis == 'horizontal':
                geometry_new.config.panel.num_pixels[0] = n_elements
                geometry_new.config.panel.pixel_size[0] *= roi.step

        return geometry_new


    def _bin_image_geometry(self):

        geometry_new = self.geometry.copy()
        for i, axis  in enumerate(self.geometry.dimension_labels):

            if not self.processed_dims[i]:
                continue

            roi = self.roi_ordered[i]
            n_elements = len(roi)

            if axis == 'channel':
                geometry_new.channels = n_elements
                geometry_new.channel_spacing *= roi.step

            elif axis == 'vertical':
                geometry_new.voxel_num_z = n_elements
                geometry_new.voxel_size_z *= roi.step

            elif axis == 'horizontal_x':
                geometry_new.voxel_num_x = n_elements
                geometry_new.voxel_size_x *= roi.step

            elif axis == 'horizontal_y':
                geometry_new.voxel_num_y = n_elements
                geometry_new.voxel_size_y *= roi.step
             
        return geometry_new


    def _binned_data_shape(self, shape_in):

        shape_binned = []
        for i in range(len(shape_in)):
            shape_binned.append(shape_in[i][self.roi_ordered[i]])
        return shape_binned


    def _bin_array(self, data, binned_array):

        shape_object = []
        slice_object = []
        denom = self.roi_ordered[0].step

        for roi in self.roi_ordered[1::]:
            shape_object.append(len(roi)) #reshape the data to add a 'bin' dimensions
            shape_object.append(roi.step)
            slice_object.append(slice(roi.start, roi.stop)) #crop data (i.e. no bin/step)
            denom *=roi.step

        
        axes_sum = tuple(range(1,len(shape_object),2))

        # needs a single 'outer dimension' (default channel) in memory at a time
        count = 0
        for l in self.roi_ordered[0]: 

            slice_proj = tuple([slice(l,l+1)]+slice_object)
            slice_resized = data.as_array()[slice_proj].copy()

            for k in range(1,self.roi_ordered[0].step):
                slice_proj = tuple([slice(l+k,l+k+1)]+slice_object)
                slice_resized += data.as_array()[slice_proj]

            slice_resized = np.squeeze(slice_resized.reshape(shape_object).sum(axis=axes_sum))
            slice_resized /= denom

            if len(self.roi_ordered[0]) > 1:
                binned_array[count] = slice_resized
            else:
                binned_array[:] = slice_resized

            count +=1


    def _construct_roi_object(self, ndim, dimension_labels, shape):
        '''
        This converts the slice style input to a range object for each dimension
        '''
        sl_list = []
        sliced_dims= [False]*ndim

        for i in range(ndim):
            roi = self.roi_input.get(dimension_labels[i],[None,None,None])

            if roi == -1:
                roi = [None,None,None]
            else:
                roi = list(roi)
                length = len(roi)
                if length == 1:
                    roi.prepend(None)
                if length == 2:
                    roi.append(None)
                
            if roi[0] == None:
                roi[0] = 0
            elif roi[0] < 0:
                roi[0] += shape[i]

            if roi[1] == None:
                roi[1] = shape[i]
            elif roi[1] < 0:
                roi[1] += shape[i]

            if roi[2] == None:
                roi[2] = 1

            roi[1]  = ((roi[1] - roi[0])// roi[2])*roi[2]+roi[0]

            if roi[0] != 0 or roi[1] != shape[i] or roi[2] != 1:
                sliced_dims[i] = True

            sl_list.append(range(*roi))
        
        self.roi_ordered=sl_list
        self.processed_dims = sliced_dims


    def process(self, out=None):

        data = self.get_input()
        num_dims = data.ndim

        self._construct_roi_object(num_dims, data.dimension_labels, data.shape)

        if isinstance(self.geometry, ImageGeometry):
            binned_geometry = self._bin_image_geometry()
        elif isinstance(self.geometry, AcquisitionGeometry):
            binned_geometry = self._bin_acquisition_geometry()
        else:
            binned_geometry = None

        # return if just acting on geometry
        if not self.data_array:
            return binned_geometry

        # calculate binned data shape
        if binned_geometry is not None:
            binned_shape = binned_geometry.shape
        else:
            binned_shape = self._binned_data_shape(data.shape)

        # create output array
        if out is None:
            if binned_geometry is not None:
                data_out = binned_geometry.allocate(None)
                binned_array = data_out.as_array()
            else:
                binned_array = np.empty(binned_shape,dtype=np.float32)
                data_out = DataContainer(binned_array,False, data.dimension_labels)

        else:
            if out.shape != binned_shape:
                raise ValueError("Shape of `out` not as expected. Got {0}, expected {1}".format(out.shape, binned_shape))
            if binned_geometry is not None:
                if out.geometry != binned_geometry:
                    raise ValueError("Geometry of `out` not as expected. Got {0}, expected {1}".format(out.geometry, binned_geometry))
            
                binned_array = out.as_array()

        self._bin_array(data, binned_array)

        if out is None:
            return data_out
