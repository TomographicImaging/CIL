#  Copyright 2022 United Kingdom Research and Innovation
#  Copyright 2022 The University of Manchester
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
# Andrew Shartis (UES, Inc.)
from cil.framework import AcquisitionData, AcquisitionGeometry, ImageData, ImageGeometry
from cil.framework.labels import AngleUnit, AcquisitionDimension, ImageDimension
import numpy as np
import os
import logging
import warnings


class ZEISSDataReader:

    '''
    Create a reader for ZEISS files

    Parameters
    ----------
    file_name: str
        file name to read
    roi: dict, default None
        dictionary with roi to load for each axis:
        ``{'axis_labels_1': (start, end, step),'axis_labels_2': (start, end, step)}``.
        ``axis_labels`` are defined by ImageGeometry and AcquisitionGeometry dimension labels.

    Notes
    -----
    `roi` behaviour:
        For ImageData to skip files or to change number of files to load,
        adjust ``vertical``. E.g. ``'vertical': (100, 300)`` will skip first 100 files
        and will load 200 files.

        ``'axis_label': -1`` is a shortcut to load all elements along axis.

        ``start`` and ``end`` can be specified as ``None`` which is equivalent
        to ``start = 0`` and ``end = load everything to the end``, respectively.
    '''

    def __init__(self, file_name=None, roi=None):

        self.file_name = file_name

        # Set logging level for dxchange reader.py
        logger_dxchange = logging.getLogger(name='dxchange.reader')
        if logger_dxchange is not None:
            logger_dxchange.setLevel(logging.ERROR)

        if file_name is not None:
            self.set_up(file_name, roi = roi)


    def set_up(self,
               file_name,
               roi = None):
        '''Set up the reader


        Parameters
        ----------
        file_name: str
            file name to read
        roi: dict, default None
            dictionary with roi to load for each axis:
            ``{'axis_labels_1': (start, end, step),'axis_labels_2': (start, end, step)}``.
            ``axis_labels`` are defined by ImageGeometry and AcquisitionGeometry dimension labels.

        Notes
        -----
        `roi` behaviour:
            ``'axis_label': -1`` is a shortcut to load all elements along axis.

            ``start`` and ``end`` can be specified as ``None`` which is equivalent
            to ``start = 0`` and ``end = load everything to the end``, respectively.

            **Acquisition Data**

            The axis labels in the `roi` dict for `AcquisitionData` will be:
            ``{'angle':(...),'vertical':(...),'horizontal':(...)}``

            **Image Data**

            The axis labels in the `roi` dict for `ImageData` will be:
            ``{'angle':(...),'vertical':(...),'horizontal':(...)}``

            To skip files or to change number of files to load,
            adjust ``vertical``. E.g. ``'vertical': (100, 300)`` will skip first 100 files
            and will load 200 files.
        '''

        # check if file exists
        file_name = os.path.abspath(file_name)
        if not(os.path.isfile(file_name)):
            raise FileNotFoundError('{}'.format(file_name))

        file_type = os.path.basename(file_name).split('.')[-1].lower()
        if file_type not in ['txrm', 'txm']:
            raise TypeError('This reader can only process TXRM or TXM files. Got {}'.format(os.path.basename(file_name)))

        self.file_name = file_name


        metadata = self.read_metadata()
        default_roi = [ [0,metadata['number_of_images'],1],
                        [0,metadata['image_height'],1],
                        [0,metadata['image_width'],1]]

        if roi is not None:
            if metadata['data geometry'] == 'acquisition':
                zeiss_data_order = {AcquisitionDimension.ANGLE: 0,
                                    AcquisitionDimension.VERTICAL: 1,
                                    AcquisitionDimension.HORIZONTAL: 2}
            else:
                zeiss_data_order = {ImageDimension.VERTICAL: 0,
                                    ImageDimension.HORIZONTAL_Y: 1,
                                    ImageDimension.HORIZONTAL_X: 2}

            # check roi labels and create tuple for slicing
            for key in roi.keys():
                idx = zeiss_data_order[key]
                if roi[key] != -1:
                    print(roi[key])
                    if key == AcquisitionDimension.ANGLE:
                        if roi[key][1] > default_roi[0][1]:
                            raise ValueError('Requested angle range {} exceeds available range [0, {}]'.format(roi[key], default_roi[0][1]))
                    elif key == ImageDimension.VERTICAL or key == AcquisitionDimension.VERTICAL:
                        if roi[key][1] > default_roi[1][1]:
                            raise ValueError('Requested vertical range {} exceeds available range [0, {}]'.format(roi[key], default_roi[1][1]))
                    elif key == ImageDimension.HORIZONTAL_X or key == ImageDimension.HORIZONTAL_Y or key == AcquisitionDimension.HORIZONTAL:
                        if roi[key][1] > default_roi[2][1]:
                            raise ValueError('Requested horizontal range {} exceeds available range [0, {}]'.format(roi[key], default_roi[2][1]))

                    for i, x in enumerate(roi[key]):
                        if x is None:
                            continue

                        if i != 2: #start and stop
                            default_roi[idx][i] = x if x >= 0 else default_roi[idx][1] - x
                        else: #step
                            default_roi[idx][i] =  x if x > 0 else 1

            self._roi = default_roi
            self._metadata = self.slice_metadata(metadata)
        else:
            self._roi = False
            self._metadata = metadata

        #setup geometry using metadata
        if metadata['data geometry'] == 'acquisition':
            self._setup_acq_geometry()
        else:
            self._setup_image_geometry()

    def read_metadata(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "pkg_resources is deprecated", UserWarning)
            import dxchange
        import olefile
        # Read one image to get the metadata
        _,metadata = dxchange.read_txrm(self.file_name,((0,1),(None),(None)))

        with olefile.OleFileIO(self.file_name) as ole:
            #Configure beam geometry
            xray_geometry = dxchange.reader._read_ole_value(ole, 'ImageInfo/XrayGeometry', '<i')

            if xray_geometry == 1:
                metadata['beam geometry'] ='cone'
            else:
                metadata['beam geometry'] = 'parallel'

            #Configure data geometry
            file_type = dxchange.reader._read_ole_value(ole, 'ImageInfo/AcquisitionMode', '<i')

            if file_type == 0:
                metadata['data geometry'] = 'acquisition'

                # Read source to center and detector to center distances
                StoRADistance = dxchange.reader._read_ole_arr(ole, \
                        'ImageInfo/StoRADistance', "<{0}f".format(metadata['number_of_images']))
                DtoRADistance = dxchange.reader._read_ole_arr(ole, \
                        'ImageInfo/DtoRADistance', "<{0}f".format(metadata['number_of_images']))

                dist_source_center = np.abs(StoRADistance[0])
                dist_center_detector = np.abs(DtoRADistance[0])

                # Pixelsize loaded in metadata is really the voxel size in um.
                # We can compute the effective detector pixel size as the geometric
                # magnification times the voxel size.
                metadata['dist_source_center'] = dist_source_center
                metadata['dist_center_detector'] = dist_center_detector
                metadata['detector_pixel_size'] = ((dist_source_center+dist_center_detector)/dist_source_center)*metadata['pixel_size']
            else:
                metadata['data geometry'] = 'image'

        return metadata

    def slice_metadata(self,metadata):
        '''
        Slices metadata to configure geometry before reading data
        '''
        image_slc = range(*self._roi[0])
        height_slc = range(*self._roi[1])
        width_slc = range(*self._roi[2])
        #These values are 0 or do not exist in TXM files and can be skipped
        if metadata['data geometry'] == 'acquisition':
            metadata['thetas'] = metadata['thetas'][image_slc]
            metadata['x_positions'] = metadata['x_positions'][image_slc]
            metadata['y_positions'] = metadata['y_positions'][image_slc]
            metadata['z_positions'] = metadata['z_positions'][image_slc]
            metadata['x-shifts'] = metadata['x-shifts'][image_slc]
            metadata['y-shifts'] = metadata['y-shifts'][image_slc]
            metadata['reference'] = metadata['reference'][height_slc.start:height_slc.stop:height_slc.step,
                                                          width_slc.start:width_slc.stop:width_slc.step]
        metadata['number_of_images'] = len(image_slc)
        metadata['image_width'] = len(width_slc)
        metadata['image_height'] = len(height_slc)
        return metadata

    def _setup_acq_geometry(self):
        '''
        Setup AcquisitionData container
        '''
        if self._metadata['beam geometry'] == 'cone':
            self._geometry = AcquisitionGeometry.create_Cone3D(
                [0,-self._metadata['dist_source_center'],0],[0,self._metadata['dist_center_detector'],0] \
                ) \
                    .set_panel([self._metadata['image_width'], self._metadata['image_height']],\
                        pixel_size=[self._metadata['detector_pixel_size']/1000,self._metadata['detector_pixel_size']/1000])\
                    .set_angles(self._metadata['thetas'],angle_unit=AngleUnit.RADIAN)
        else:
            self._geometry = AcquisitionGeometry.create_Parallel3D()\
                    .set_panel([self._metadata['image_width'], self._metadata['image_height']])\
                    .set_angles(self._metadata['thetas'],angle_unit=AngleUnit.RADIAN)
        self._geometry.dimension_labels =  ['angle', 'vertical', 'horizontal']

    def _setup_image_geometry(self):
        '''
        Setup ImageData container
        '''
        slices = self._metadata['number_of_images']
        width = self._metadata['image_width']
        height = self._metadata['image_height']
        voxel_size = self._metadata['pixel_size']
        self._geometry = ImageGeometry(voxel_num_x=width,
                                    voxel_size_x=voxel_size,
                                    voxel_num_y=height,
                                    voxel_size_y=voxel_size,
                                    voxel_num_z=slices,
                                    voxel_size_z=voxel_size)

    def read(self):
        '''
        Reads projections and return Acquisition (TXRM) or Image (TXM) Data container
        '''
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "pkg_resources is deprecated", UserWarning)
            import dxchange
        # Load projections or slices from file
        slice_range = None
        if self._roi:
            slice_range = tuple(self._roi)
        data, _ = dxchange.read_txrm(self.file_name,slice_range)

        if isinstance(self._geometry,AcquisitionGeometry):
            # Normalise data by flatfield
            data = data / self._metadata['reference']

            for num in range(self._metadata['number_of_images']):
                data[num,:,:] = np.roll(data[num,:,:], \
                    (int(self._metadata['x-shifts'][num]),int(self._metadata['y-shifts'][num])), \
                    axis=(1,0))

            acq_data = AcquisitionData(array=data, deep_copy=False, geometry=self._geometry.copy())
            return acq_data
        else:
            ig_data = ImageData(array=data, deep_copy=False, geometry=self._geometry.copy())
            return ig_data


    def get_geometry(self):
        '''
        Return Acquisition (TXRM) or Image (TXM) Geometry object
        '''
        return self._geometry

    def get_metadata(self):
        '''return the metadata of the file'''
        return self._metadata
