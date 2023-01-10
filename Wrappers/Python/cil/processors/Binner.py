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
import weakref
import logging

# IppInit should be called once.
# get slice geometry needs updating for off centre slices
# Look at input paramaters.... maybe time to break backward compatibility 

try:
    from cil.processors.cilacc_binner import Binner_IPP
    has_ipp = True
except:
    has_ipp = False


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

    accelerated : boolean, default=True
        Used the CIL accelerated backend if `True`, numpy if `False`.


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
                 roi = None,accelerated=True):


        if accelerated and not has_ipp:
            raise RuntimeError("Cannot run accelerated Binner without the IPP libraries.")


        kwargs = {
            '_roi_input': roi, 
            '_accelerated':accelerated,
            '_roi_ordered':None, 
            '_data_array': False, 
            '_geometry': None, 
            '_processed_dims':None, 
            '_shape_in':None, 
            '_shape_out':None, 
            '_labels_in':None, 
            '_binning':None, 
            '_index_start':None, 
            }

        super(Binner, self).__init__(**kwargs)
    

    def set_input(self, dataset):
        """
        Set the input data to the processor

        Parameters
        ----------
        input : DataContainer, Geometry
            The input DataContainer
        """

        if issubclass(type(dataset), DataContainer) or isinstance(dataset,(AcquisitionGeometry,ImageGeometry)):
            if self.check_input(dataset):
                self.__dict__['input'] = weakref.ref(dataset)
                self.__dict__['shouldRun'] = True
            else:
                raise ValueError('Input data not compatible')
        else:
            raise TypeError("Input type mismatch: got {0} expecting {1}"\
                            .format(type(dataset), DataContainer))


    def check_input(self, data):

        if isinstance(data, (ImageData,AcquisitionData)):
            self._data_array = True
            self._geometry = data.geometry

        elif isinstance(data, DataContainer):
            self._data_array = True
            self._geometry = None

        elif isinstance(data, (ImageGeometry, AcquisitionGeometry)):
            self._data_array = False
            self._geometry = data

        else:
            raise TypeError('Processor supports following data types:\n' +
                            ' - ImageData\n - AcquisitionData\n - DataContainer\n - ImageGeometry\n - AcquisitionGeometry')

        if self._data_array:
            if data.dtype != np.float32:
                raise TypeError("Expected float32")

        if (self._roi_input == None):
            raise ValueError('Please, specify roi')

        for key in self._roi_input.keys():
            if key not in data.dimension_labels:
                raise ValueError('Wrong label is specified for roi, expected one of {}.'.format(data.dimension_labels))

        self._parse_roi(data.ndim, data.shape, data.dimension_labels)
        
        return True 


    def _parse_roi(self, ndim, shape, dimension_labels):
        '''
        This processes the roi input style
        '''
        offset = 4-ndim
        labels_in = [None]*4
        labels_in[offset::] = dimension_labels
        shape_in = [1]*4
        shape_in[offset::] = shape

        # defaults
        shape_out = shape_in.copy()
        processed_dim = np.zeros((4,),np.bool_)
        binning = [1,1,1,1]
        index_start =[0,0,0,0]
        index_end = list(shape_in.copy())

        for i in range(ndim):

            roi = self._roi_input.get(dimension_labels[i],None)

            if roi == None or roi == -1:
                continue

            start = index_start[offset + i]
            stop = shape_out[offset + i]
            step = binning[offset + i]

            roi = list(roi)
            length = len(roi)

            if length == 1:
                if roi[0] is not None:
                    stop = roi[0]
            elif length > 1:
                if roi[0] is not None:
                    start = roi[0]
                if roi[1] is not None:
                    stop = roi[1]
            if length > 2:
                if roi[2] is not None:
                    step = roi[2]

            # deal with negative indexing
            if start < 0:
                start += shape_in[offset + i]

            if stop < 0:
                stop += shape_in[offset + i]

            #set values
            binning[offset + i]  = int(step)
            index_start[offset + i]  = int(start)
            shape_out[offset + i]  = (stop - start)// step

            #end pixel based on binning
            index_end[offset + i]  = int(index_start[offset + i] + shape_out[offset + i] * binning[offset + i])

            if shape_out[offset + i] != shape_in[offset + i]:
                processed_dim[offset + i] = 1

        range_list = []
        for i in range(4):
            range_list.append(range(index_start[i], index_end[i],binning[i]))
            
        # set values
        self._shape_in = shape_in
        self._shape_out = shape_out
        self._labels_in = labels_in
        self._processed_dims = processed_dim
        self._roi_ordered = range_list
        self._binning = binning 
        self._index_start = index_start

    def _bin_acquisition_geometry(self):
        """
        Creates the binned acquisition geometry
        """
        geometry_new = self._geometry.copy()
        system_detector = geometry_new.config.system.detector

        processed_dims = self._processed_dims.copy()

        # deal with vertical first as it may change the geometry type
        if 'vertical' in self._geometry.dimension_labels:
            vert_ind = self._labels_in.index('vertical')
            if processed_dims[vert_ind]:
                roi = self._roi_ordered[vert_ind]
                n_elements = len(roi)

                if n_elements > 1:
                    centre_offset = geometry_new.config.panel.pixel_size[1] * ((n_elements * roi.step)*0.5 + roi.start - geometry_new.config.panel.num_pixels[1] * 0.5 )

                    geometry_new.config.panel.num_pixels[1] = n_elements
                    system_detector.position = system_detector.position + centre_offset * system_detector.direction_y
                else:
                    geometry_new = geometry_new.get_slice(vertical = (roi.start + roi.step/2))

                geometry_new.config.panel.pixel_size[1] *= roi.step
                processed_dims[vert_ind] = False


        for i, axis  in enumerate(self._labels_in):

            if not processed_dims[i]:
                continue
            
            roi = self._roi_ordered[i]
            n_elements = len(roi)

            if axis == 'channel':
                geometry_new.set_channels(num_channels=n_elements)

            elif axis == 'angle':
                shape = (n_elements, roi.step)
                geometry_new.config.angles.angle_data = self._geometry.angles[roi.start:roi.start+n_elements*roi.step].reshape(shape).mean(1)
                
            elif axis == 'horizontal':
                centre_offset = geometry_new.config.panel.pixel_size[0] * ( (n_elements * roi.step)*0.5 + roi.start - geometry_new.config.panel.num_pixels[0] * 0.5 )
                geometry_new.config.panel.num_pixels[0] = n_elements
                geometry_new.config.panel.pixel_size[0] *= roi.step
                system_detector.position = system_detector.position + centre_offset * system_detector.direction_x

        return geometry_new


    def _bin_image_geometry(self):
        """
        Creates the binned image geometry
        """
        geometry_new = self._geometry.copy()

        for i, axis in enumerate(self._labels_in):

            if not self._processed_dims[i]:
                continue

            roi = self._roi_ordered[i]
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


    def _bin_array_numpy(self, array_in, array_binned):
        """
        Bins the array using numpy. This method is slower and less memory efficient than self._bin_array_acc
        """
        shape_object = []
        slice_object = []

        for i in range(4):
            # reshape the data to add each 'bin' dimensions
            shape_object.append(self._shape_out[i]) 
            shape_object.append(self._binning[i])
            slice_object.append(slice(self._index_start[i], self._index_start[i] + self._shape_out[i] * self._binning[i])) #crop data (i.e. no bin/step)
        
        shape_object = tuple(shape_object)
        slice_object = tuple(slice_object)

        data_resized = array_in.reshape(self._shape_in)[slice_object].reshape(shape_object)

        mean_order = (-1, 1, 2, 3)
        for i in range(4):
            data_resized = data_resized.mean(mean_order[i])
            
        np.copyto(array_binned, data_resized)


    def _bin_array_acc(self, array_in, array_binned):
        """
        Bins the array using the accelerated CIL backend
        """
        binner_ipp = Binner_IPP(self._shape_in, self._shape_out, self._index_start, self._binning)

        res = binner_ipp.bin(array_in, array_binned)
        if res != 0:
            raise RuntimeError("Call failed")


    def process(self, out=None):

        data = self.get_input()

        if isinstance(self._geometry, ImageGeometry):
            binned_geometry = self._bin_image_geometry()
        elif isinstance(self._geometry, AcquisitionGeometry):
            binned_geometry = self._bin_acquisition_geometry()
        else:
            binned_geometry = None

        # return if just acting on geometry
        if not self._data_array:
            return binned_geometry

        # create output array or check size and shape of passed out
        if out is None:
            if binned_geometry is not None:
                data_out = binned_geometry.allocate(None)
                binned_array = data_out.as_array()
            else:
                binned_array = np.empty(self._shape_out,dtype=np.float32)
                data_out = DataContainer(binned_array,False, data.dimension_labels)
        else:
            try:
                out.array = np.asarray(out.array, dtype=np.float32, order='C').reshape(self._shape_out)
            except:
                raise ValueError("Array of `out` not compatible. Expected shape: {0}, data type: {1} Got shape: {2}, data type: {3}".format(self._shape_out, np.float32, out.array.shape, out.array.dtype))

            if binned_geometry is not None:
                if out.geometry != binned_geometry:
                    raise ValueError("Geometry of `out` not as expected. Got {0}, expected {1}".format(out.geometry, binned_geometry))
            
            binned_array = out.array

        # bin data
        if self._accelerated:
            self._bin_array_acc(data.array, binned_array)
        else:
            self._bin_array_numpy(data.array, binned_array)

        if out is None:
            return data_out
