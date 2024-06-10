#  Copyright 2019 United Kingdom Research and Innovation
#  Copyright 2019 The University of Manchester
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
# Kyle Pidgeon (UKRI-STFC)

import numpy as np
import os
from cil.framework import AcquisitionData, AcquisitionGeometry, ImageData, ImageGeometry

h5pyAvailable = True
try:
    import h5py
except:
    h5pyAvailable = False


class NEXUSDataReader(object):

    """
    Create a reader for NeXus files.

    Parameters
    ----------
    file_name: str
        the full path to the NeXus file to read.
    """

    def __init__(self, file_name=None):

        self.file_name = file_name

        if self.file_name is not None:
            self.set_up(file_name = self.file_name)

    def set_up(self,
               file_name = None):
        """
        Initialise reader.

        Parameters
        ----------
        file_name : str
            Full path to NeXus file
        """

        self.file_name = os.path.abspath(file_name)

        # check that h5py library is installed
        if (h5pyAvailable == False):
            raise Exception('h5py is not available, cannot load NEXUS files.')

        if self.file_name == None:
            raise Exception('Path to nexus file is required.')

        # check if nexus file exists
        if not(os.path.isfile(self.file_name)):
            raise Exception('File\n {}\n does not exist.'.format(self.file_name))

        self._geometry = None

    def read_dimension_labels(self, attrs):
        dimension_labels = [None] * 4
        for k,v in attrs.items():
            if k in ['dim0', 'dim1', 'dim2' , 'dim3']:
                dimension_labels[int(k[3:])] = v

        # remove Nones
        dimension_labels = [i for i in dimension_labels if i]

        if len(dimension_labels) == 0:
            dimension_labels = None

        return dimension_labels

    def get_geometry(self):
        """
        Parse NEXUS file and return acquisition or reconstructed volume
        parameters, depending on file type.

        Returns
        -------
        AcquisitionGeometry or ImageGeometry
            Acquisition or reconstructed volume parameters. Exact type
            depends on file content.
        """

        with h5py.File(self.file_name,'r') as dfile:

            if np.string_(dfile.attrs['creator']) != np.string_('NEXUSDataWriter.py'):
                raise Exception('We can parse only files created by NEXUSDataWriter.py')

            ds_data = dfile['entry1/tomo_entry/data/data']

            if ds_data.attrs['data_type'] == 'ImageData':

                self._geometry = ImageGeometry(voxel_num_x = int(ds_data.attrs['voxel_num_x']),
                                                voxel_num_y = int(ds_data.attrs['voxel_num_y']),
                                                voxel_num_z = int(ds_data.attrs['voxel_num_z']),
                                                voxel_size_x = ds_data.attrs['voxel_size_x'],
                                                voxel_size_y = ds_data.attrs['voxel_size_y'],
                                                voxel_size_z = ds_data.attrs['voxel_size_z'],
                                                center_x = ds_data.attrs['center_x'],
                                                center_y = ds_data.attrs['center_y'],
                                                center_z = ds_data.attrs['center_z'],
                                                channels = ds_data.attrs['channels'])

                if ds_data.attrs.__contains__('channel_spacing') == True:
                    self._geometry.channel_spacing = ds_data.attrs['channel_spacing']

                # read the dimension_labels from dim{}
                dimension_labels = self.read_dimension_labels(ds_data.attrs)

            else:   # AcquisitionData
                if ds_data.attrs.__contains__('dist_source_center') or dfile['entry1/tomo_entry'].__contains__('config/source/position'):
                    geom_type = 'cone'
                else:
                    geom_type = 'parallel'

                if ds_data.attrs.__contains__('num_pixels_v'):
                    num_pixels_v = ds_data.attrs.get('num_pixels_v')
                elif ds_data.attrs.__contains__('pixel_num_v'):
                    num_pixels_v = ds_data.attrs.get('pixel_num_v')
                else:
                    num_pixels_v = 1

                if num_pixels_v > 1:
                    dim = 3
                else:
                    dim = 2


                if self.is_old_file_version():
                    num_pixels_h = ds_data.attrs.get('pixel_num_h', 1)
                    num_channels = ds_data.attrs['channels']
                    ds_angles = dfile['entry1/tomo_entry/data/rotation_angle']

                    if geom_type == 'cone' and dim == 3:
                        self._geometry = AcquisitionGeometry.create_Cone3D(source_position=[0, -ds_data.attrs['dist_source_center'], 0],
                                                                               detector_position=[0, ds_data.attrs['dist_center_detector'],0])
                    elif geom_type == 'cone' and dim == 2:
                        self._geometry = AcquisitionGeometry.create_Cone2D(source_position=[0, -ds_data.attrs['dist_source_center']],
                                                        detector_position=[0, ds_data.attrs['dist_center_detector']])
                    elif geom_type == 'parallel' and dim == 3:
                        self._geometry = AcquisitionGeometry.create_Parallel3D()
                    elif geom_type == 'parallel' and dim == 2:
                        self._geometry = AcquisitionGeometry.create_Parallel2D()


                else:
                    num_pixels_h = ds_data.attrs.get('num_pixels_h', 1)
                    num_channels = ds_data.attrs['num_channels']
                    ds_angles = dfile['entry1/tomo_entry/config/angles']

                    rotation_axis_position = list(dfile['entry1/tomo_entry/config/rotation_axis/position'])
                    detector_position = list(dfile['entry1/tomo_entry/config/detector/position'])

                    ds_detector = dfile['entry1/tomo_entry/config/detector']
                    if ds_detector.__contains__('direction_x'):
                        detector_direction_x = list(dfile['entry1/tomo_entry/config/detector/direction_x'])
                    else:
                        detector_direction_x = list(dfile['entry1/tomo_entry/config/detector/direction_row'])

                    if ds_detector.__contains__('direction_y'):
                        detector_direction_y = list(dfile['entry1/tomo_entry/config/detector/direction_y'])
                    elif ds_detector.__contains__('direction_col'):
                        detector_direction_y = list(dfile['entry1/tomo_entry/config/detector/direction_col'])

                    ds_rotate = dfile['entry1/tomo_entry/config/rotation_axis']
                    if ds_rotate.__contains__('direction'):
                        rotation_axis_direction = list(dfile['entry1/tomo_entry/config/rotation_axis/direction'])

                    if geom_type == 'cone':
                        source_position = list(dfile['entry1/tomo_entry/config/source/position'])

                        if dim == 2:
                            self._geometry = AcquisitionGeometry.create_Cone2D(source_position, detector_position, detector_direction_x, rotation_axis_position)
                        else:
                            self._geometry = AcquisitionGeometry.create_Cone3D(source_position,\
                                                detector_position, detector_direction_x, detector_direction_y,\
                                                rotation_axis_position, rotation_axis_direction)
                    else:
                        ray_direction = list(dfile['entry1/tomo_entry/config/ray/direction'])

                        if dim == 2:
                            self._geometry = AcquisitionGeometry.create_Parallel2D(ray_direction, detector_position, detector_direction_x, rotation_axis_position)
                        else:
                            self._geometry = AcquisitionGeometry.create_Parallel3D(ray_direction,\
                                                detector_position, detector_direction_x, detector_direction_y,\
                                                rotation_axis_position, rotation_axis_direction)

                # for all Aquisition data
                #set angles
                angles = list(ds_angles)
                angle_unit = ds_angles.attrs.get('angle_unit','degree')
                initial_angle = ds_angles.attrs.get('initial_angle',0)
                self._geometry.set_angles(angles, initial_angle=initial_angle, angle_unit=angle_unit)

                #set panel
                pixel_size_v = ds_data.attrs.get('pixel_size_v', ds_data.attrs['pixel_size_h'])
                origin = ds_data.attrs.get('panel_origin','bottom-left')
                self._geometry.set_panel((num_pixels_h, num_pixels_v),\
                                        pixel_size=(ds_data.attrs['pixel_size_h'], pixel_size_v),\
                                        origin=origin)

                # set channels
                self._geometry.set_channels(num_channels)

                dimension_labels = []
                dimension_labels = self.read_dimension_labels(ds_data.attrs)

        #set labels
        self._geometry.set_labels(dimension_labels)

        return self._geometry

    def get_data_scale(self):
        """
        Parse NEXUS file and return the scale factor applied to compress
        the dataset.

        Returns
        -------
        scale : float
            The scale factor applied to compress the dataset
        """

        with h5py.File(self.file_name,'r') as dfile:
            ds_data = dfile['entry1/tomo_entry/data/data']
            try:
                scale = ds_data.attrs['scale']
            except:
                scale = 1.0

        return scale

    def get_data_offset(self):
        """
        Parse NEXUS file and return the offset factor applied to compress
        the dataset.

        Returns
        -------
        offset : float
            The offset factor applied to compress the dataset
        """

        with h5py.File(self.file_name,'r') as dfile:
            ds_data = dfile['entry1/tomo_entry/data/data']
            try:
                offset = ds_data.attrs['offset']
            except:
                offset = 0.0

        return offset

    def __read_as(self, dtype=np.float32):
        """
        Parse NEXUS file and return raw file content.

        Parameters
        ----------
        dtype : data-type
            The data type used for storing the parsed data.

        Returns
        -------
        output : ImageData or AcquisitionData
            The parsed raw data. Exact type depends on file content.
        """

        if self._geometry is None:
            self.get_geometry()

        #allocate data container as requested type
        output = self._geometry.allocate(None, dtype=dtype)

        with h5py.File(self.file_name,'r') as dfile:

            ds_data = dfile['entry1/tomo_entry/data/data']
            ds_data.read_direct(output.array)

        return output

    def read_as_original(self):
        """
        Returns the compressed data from the file.

        Returns
        -------
        output : ImageData or AcquisitionData
            The raw, compressed data. Exact type depends on file content.
        """

        with h5py.File(self.file_name,'r') as dfile:
            ds_data = dfile['entry1/tomo_entry/data/data']
            dtype = ds_data.dtype

        return self.__read_as(dtype)


    def read(self):
        """
        Returns the uncompressed data as numpy.float32.

        Returns
        -------
        output : ImageData or AcquisitionData
            The uncompressed data. Exact type depends on file content.
        """

        output = self.__read_as(np.float32)
        scale = self.get_data_scale()
        offset = self.get_data_offset()

        if offset != 0:
            output -= offset
        if scale != 1:
            output /= scale

        return output


    def load_data(self):
        """
        Alias of `read`.

        See Also
        --------
        read
        """

        return self.read()

    def is_old_file_version(self):
        #return ds_data.attrs.__contains__('geom_type')
        with h5py.File(self.file_name,'r') as dfile:

            if np.string_(dfile.attrs['creator']) != np.string_('NEXUSDataWriter.py'):
                raise Exception('We can parse only files created by NEXUSDataWriter.py')

            ds_data = dfile['entry1/tomo_entry/data/data']

            return 'geom_type' in ds_data.attrs.keys()
            # return True
