#  Copyright 2021 United Kingdom Research and Innovation
#  Copyright 2021 The University of Manchester
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

from cil.framework import (DataProcessor, AcquisitionData, ImageData, DataContainer, ImageGeometry, VectorGeometry,
                           AcquisitionGeometry)
from cil.framework.labels import AcquisitionType
import numpy as np
import weakref
import logging

log = logging.getLogger(__name__)


# Note to developers: Binner and Slicer share a lot of common code
# so Binner has been implemented as a child of Slicer.  This makes use
# of commonality and redefines only the methods that differ. These methods
# dictate the style of slicer
class Slicer(DataProcessor):

    """This creates a Slicer processor.

    The processor will crop the data, and then return every n input pixels along a dimension from the starting index.

    The output will be a data container with the data, and geometry updated to reflect the operation.

    Parameters
    ----------

    roi : dict
        The region-of-interest to slice {'axis_name1':(start,stop,step), 'axis_name2':(start,stop,step)}
        The `key` being the axis name to apply the processor to, the `value` holding a tuple containing the ROI description

        Start: Starting index of input data. Must be an integer, or `None` defaults to index 0.
        Stop: Stopping index of input data. Must be an integer, or `None` defaults to index N.
        Step: Number of pixels to average together. Must be an integer or `None` defaults to 1.


    Example
    -------

    >>> from cil.processors import Slicer
    >>> roi = {'horizontal':(10,-10,2),'vertical':(10,-10,2)}
    >>> processor = Slicer(roi)
    >>> processor.set_input(data)
    >>> data_sliced= processor.get_output()


    Example
    -------
    >>> from cil.processors import Slicer
    >>> roi = {'horizontal':(None,None,2),'vertical':(None,None,2)}
    >>> processor = Slicer(roi)
    >>> processor.set_input(data.geometry)
    >>> geometry_sliced = processor.get_output()


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

        kwargs = {
            '_roi_input': roi,
            '_roi_ordered':None,
            '_data_array': False,
            '_geometry': None,
            '_processed_dims':None,
            '_shape_in':None,
            '_shape_out_full':None,
            '_shape_out':None,
            '_labels_out':None,
            '_labels_in':None,
            '_pixel_indices':None,
            '_accelerated': True
            }

        super(Slicer, self).__init__(**kwargs)


    def set_input(self, dataset):
        """
        Set the input data or geometry to the processor

        Parameters
        ----------
        dataset : DataContainer, Geometry
            The input DataContainer or Geometry
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

        self._set_up()

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
            if key not in ['angle', 'channel']:
                if isinstance(self._geometry , (AcquisitionGeometry)) and self._geometry.geom_type & AcquisitionType.CONE_FLEX:
                    raise NotImplementedError("Cone-Flex geometry is not supported by this processor for slicing along any dimension other than 'angles' or 'channels'")

        return True


    def _set_up(self):
        """
        This parses the input roi generically and then configures the processor according to its class.
        """
        #read input
        data = self.get_input()
        self._parse_roi(data.ndim, data.shape, data.dimension_labels)
        #processor specific configurations
        self._configure()
        # set boolean of dimensions to process
        self._processed_dims = [0 if self._shape_out_full[i] == self._shape_in[i] else 1 for i in range(4)]
        self._shape_out = tuple([i for i in self._shape_out_full if i > 1])
        self._labels_out = [self._labels_in[i] for i,x in enumerate(self._shape_out_full) if x > 1]

    def _parse_roi(self, ndim, shape, dimension_labels):
        '''
        Process the input roi
        '''
        offset = 4-ndim
        labels_in = [None]*4
        labels_in[offset::] = dimension_labels
        shape_in = [1]*4
        shape_in[offset::] = shape

        # defaults
        range_list = [range(0,x, 1) for x in shape_in]

        for i in range(ndim):

            roi = self._roi_input.get(dimension_labels[i],None)

            if roi == None or roi == -1:
                continue

            start = range_list[offset + i].start
            stop = range_list[offset + i].stop
            step = range_list[offset + i].step

            # accepts a tuple, range or slice
            try:
                roi = [roi.start, roi.stop, roi.step]
            except AttributeError:
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

            if stop <= 0:
                stop += shape_in[offset + i]

            if stop > shape_in[offset+i]:
                log.warning(f"ROI for axis {dimension_labels[i]} has 'stop' out of bounds. Using axis length as stop value."
                            f" Got stop index: {stop}, using {shape_in[offset+i]}")
                stop = shape_in[offset+i]

            if start > shape_in[offset+i]:
                raise ValueError(f"ROI for axis {dimension_labels[i]} has 'start' out of bounds."
                                 f" Got start index: {start} for axis length {shape_in[offset+i]}")

            if start >= stop:
                raise ValueError(f"ROI for axis {dimension_labels[i]} has 'start' out of bounds."
                                 f" Got start index: {start}, stop index {stop}")

            # set values
            range_list[offset+ i] = range(int(start), int(stop), int(step))

        # set values
        self._shape_in = shape_in
        self._labels_in = labels_in
        self._roi_ordered = range_list


    def _configure(self):
        """
        Once the ROI has been parsed this configure the input specifically for use with Slicer
        """
        self._shape_out_full = [len(x) for x in self._roi_ordered]
        self._pixel_indices = [(x[0],x[-1]) for x in self._roi_ordered]


    def _get_slice_position(self, roi):
        """
        Return the vertical position to extract a single slice for sliced geometry
        """
        return roi.start


    def _get_angles(self, roi):
        """
        Returns the sliced angles according to the roi
        """
        return self._geometry.angles[roi.start:roi.stop:roi.step]

    def _process_acquisition_geometry(self):
        """
        Creates the new acquisition geometry
        """
        geometry_new = self._geometry.copy()

        processed_dims = self._processed_dims.copy()

        # deal with vertical first as it may change the geometry type
        if 'vertical' in self._geometry.dimension_labels:
            vert_ind = self._labels_in.index('vertical')
            if processed_dims[vert_ind]:
                roi = self._roi_ordered[vert_ind]
                n_elements = len(roi)

                if n_elements > 1:
                    # difference in end indices, minus differences in start indices, divided by 2
                    pixel_offset = ((self._shape_in[vert_ind] -1 - self._pixel_indices[vert_ind][1]) - self._pixel_indices[vert_ind][0])*0.5
                    geometry_new.config.shift_detector_in_plane(pixel_offset, 'vertical')
                    geometry_new.config.panel.num_pixels[1] = n_elements
                else:
                    try:
                        position = self._get_slice_position(roi)
                        geometry_new = geometry_new.get_slice(vertical = position)
                    except ValueError:
                        log.warning("Unable to calculate the requested 2D geometry. Returning geometry=`None`")
                        return None

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
                if self._geometry.geom_type & AcquisitionType.CONE_FLEX:
                    geometry_new.config.system.num_positions = int(np.ceil((roi.stop - roi.start )/ roi.step))
                    print(roi.stop, roi.start, roi.step)
                    geometry_new.config.system.source = self._geometry.config.system.source[roi.start:roi.stop:roi.step]
                    geometry_new.config.system.detector = self._geometry.config.system.detector[roi.start:roi.stop:roi.step]
                    print("Number of positions: ", geometry_new.config.system.num_positions)
                    print("Source positions: ", len(geometry_new.config.system.source))
                    print("Detector positions: ", len(geometry_new.config.system.detector))
                else:
                    geometry_new.config.angles.angle_data = self._get_angles(roi)

            elif axis == 'horizontal':
                pixel_offset = ((self._shape_in[i] -1 - self._pixel_indices[i][1]) - self._pixel_indices[i][0])*0.5

                geometry_new.config.shift_detector_in_plane(pixel_offset, axis)
                geometry_new.config.panel.num_pixels[0] = n_elements
                geometry_new.config.panel.pixel_size[0] *= roi.step

        return geometry_new


    def _process_image_geometry(self):
        """
        Creates the new image geometry
        """

        if len(self._shape_out) == 0:
            return None
        elif len(self._shape_out) ==1:
            return VectorGeometry(self._shape_out[0], dimension_labels=self._labels_out[0])

        geometry_new = self._geometry.copy()
        for i, axis in enumerate(self._labels_in):

            if not self._processed_dims[i]:
                continue

            roi = self._roi_ordered[i]
            n_elements = len(roi)

            voxel_offset = (self._shape_in[i] -1 - self._pixel_indices[i][1] - self._pixel_indices[i][0])*0.5

            if axis == 'channel':
                geometry_new.channels = n_elements
                geometry_new.channel_spacing *= roi.step

            elif axis == 'vertical':
                geometry_new.center_z -= voxel_offset * geometry_new.voxel_size_z

                geometry_new.voxel_num_z = n_elements
                geometry_new.voxel_size_z *= roi.step

            elif axis == 'horizontal_x':
                geometry_new.center_x -= voxel_offset * geometry_new.voxel_size_x

                geometry_new.voxel_num_x = n_elements
                geometry_new.voxel_size_x *= roi.step

            elif axis == 'horizontal_y':
                geometry_new.center_y -= voxel_offset * geometry_new.voxel_size_y

                geometry_new.voxel_num_y = n_elements
                geometry_new.voxel_size_y *= roi.step

        return geometry_new


    def _process_data(self, dc_in, dc_out):
        """
        Slice the data array
        """
        slice_obj = tuple([slice(x.start, x.stop, x.step) for x in self._roi_ordered])
        arr_in = dc_in.array.reshape(self._shape_in)
        dc_out.fill(np.squeeze(arr_in[slice_obj]))

    def process(self, out=None):
        """
        Processes the input data

        Parameters
        ----------
        out : ImageData, AcquisitionData, DataContainer, optional
           Fills the referenced DataContainer with the processed output and suppresses the return

        Returns
        -------
        DataContainer
            The downsampled output is returned. Depending on the input type this may be:
            ImageData, AcquisitionData, DataContainer, ImageGeometry, AcquisitionGeometry
        """
        data = self.get_input()

        if isinstance(self._geometry, ImageGeometry):
            new_geometry = self._process_image_geometry()
        elif isinstance(self._geometry, AcquisitionGeometry):
            new_geometry = self._process_acquisition_geometry()
        else:
            new_geometry = None

        print("New geometry: ", new_geometry)
        print("Shape out: ", self._shape_out)

        # return if just acting on geometry
        if not self._data_array:
            return new_geometry

        # create output array or check size and shape of passed out
        if out is None:
            if new_geometry is not None:
                data_out = new_geometry.allocate(None)
                print("New geometry shape: ", data_out.shape)
            else:
                processed_array = np.empty(self._shape_out,dtype=np.float32)
                data_out = DataContainer(processed_array,False, self._labels_out)
        else:
            try:
                out.array = np.asarray(out.array, dtype=np.float32, order='C').reshape(self._shape_out)
            except:
                raise ValueError("Array of `out` not compatible. Expected shape: {0}, data type: {1} Got shape: {2}, data type: {3}".format(self._shape_out, np.float32, out.array.shape, out.array.dtype))

            if new_geometry is not None:
                if out.geometry != new_geometry:
                    raise ValueError("Geometry of `out` not as expected. Got {0}, expected {1}".format(out.geometry, new_geometry))

            data_out = out


        self._process_data(data, data_out)


        return data_out
