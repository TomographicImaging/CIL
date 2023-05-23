# -*- coding: utf-8 -*-
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

from cil.framework import AcquisitionGeometry
from cil.io.TIFF import TIFFStackReader
from cil.io.ReaderABC import ReaderExtendedABC
from cil.processors import Normaliser
import numpy as np
import os
import logging

class NikonDataReader(ReaderExtendedABC):

    '''Reader for xtekct files

    This will read xtekct files and allow you create a CIL `AcquisitionData`.

    If configured the geometry will respect any centre of rotation offset.
    
    Parameters
    ----------
    file_name: str 
        full path to .xtekct file

    Example
    -------
    Read in your Nikon dataset
    >>> reader = NikonDataReader('path/to/my/dataset.xtekct')
    >>> data = reader.read()

    Example
    -------
    Read in a region of interest (ROI). This is done by using `set_panel_roi`.
    `preview` can then be used to preview 2 angles of the normalised data with the requested ROI.
    Reset the configured ROI with `reset`

    >>> reader = NikonDataReader('path/to/my/dataset.xtekct')
    >>> reader.set_panel_roi(vertical=(100,100), horizontal=(50,50))
    >>> reader.preview()
    >>> data = reader.read()

    '''
    
    def __init__(self, file_name = None, **deprecated_kwargs):

        super().__init__(file_name)
        #care about order of inteface inheritance?


        if deprecated_kwargs.get('roi', None) is not None:
            logging.warning("Input argument `roi` has been deprecated. Please use methods 'set_panel_roi()' and 'set_angles()' instead")
            
            roi = deprecated_kwargs.pop('roi')
            self.set_angles(roi.get('angle')) 
            self.set_panel_roi(vertical=roi.get('vertical'), horizontal=roi.get('horizontal'))

        if deprecated_kwargs.pop('normalise', None) is not None:
            raise DeprecationWarning("Input argument `normalise` has been deprecated.\
                The 'read' method will return the normalised data.\
                The 'get_data_array' method will return the data array without processing")

        if deprecated_kwargs.pop('mode', None) is not None:
            raise DeprecationWarning("Input argument `mode` has been deprecated.\
                Please use methods 'set_panel_roi()' to bin on the spatial domain \
                and 'set_angles()' to slice on the angular domain")

        if deprecated_kwargs.pop('fliplr', None) is not None:
            raise PendingDeprecationWarning("Input argument `fliplr` has been deprecated.")
            
        if deprecated_kwargs:
            logging.warning("Additional keyworded arguments passed but not used: {}".format(deprecated_kwargs))


    @property
    def _supported_extensions(self):
        """A list of file extensions supported by this reader"""
        return ['xtekct']
     

    def _read_metadata(self):
        """
        Constructs a dictionary `self._metadata` of the values used from the metadata. 
        """
        self._metadata = {}

        # parse xtekct file
        with open(self.file_name, 'r', errors='replace') as f:
            content = f.readlines()    
                
        content = [x.strip() for x in content]
        
        #initialise parameters
        self._metadata['ObjectTilt'] = 0
        self._metadata['ObjectRoll'] = None
        self._metadata['ObjectOffsetX'] = None
        self._metadata['CentreOfRotationTop'] = 0
        self._metadata['CentreOfRotationBottom'] = 0

        # list of params to read and type to store as
        params  = [('Projections',int),('WhiteLevel',float),('DetectorPixelsY',int),('DetectorPixelsX',int),
        ('DetectorPixelSizeX',float),('DetectorPixelSizeY',float),('SrcToObject',float),('SrcToDetector',float),('InitialAngle',float),
        ('AngularStep',float),('ObjectOffsetX',float),('ObjectRoll',float),('ObjectTilt',float),
        ('CentreOfRotationTop',float),('CentreOfRotationBottom',float),('InputFolderName',str)]

        for line in content:
            for param in params:
                if line.startswith(param[0]):
                    self._metadata[param[0]] = param[1](line.split('=')[1])

        self._data_path = os.path.join(os.path.dirname(self.file_name), self.metadata['InputFolderName'])


    def _create_full_geometry(self):
        """
        Create the `AcquisitionGeometry` `self._acquisition_geometry` that describes the full dataset
        """

        #angles from xtekct ignore *.ang and _ctdata.txt as not correct
        angles = np.asarray( [ self.metadata['AngularStep'] * proj for proj in range(self.metadata['Projections']) ] , dtype=np.float32)

        #convert NikonGeometry to CIL geometry
        angles = -angles - self.metadata['InitialAngle'] + 180

        magnification = self.metadata['SrcToObject'] /self.metadata['SrcToDetector']


        if self.metadata['ObjectOffsetX'] == None:
            object_offset_x = (self.metadata['CentreOfRotationBottom']+self.metadata['CentreOfRotationTop'])* 0.5 * magnification
        else:
            object_offset_x=self.metadata['ObjectOffsetX']

        if self.metadata['ObjectRoll'] == None:
            x = (self.metadata['CentreOfRotationTop']-self.metadata['CentreOfRotationBottom'])
            y = self.metadata['DetectorPixelsY'] * self.metadata['DetectorPixelSizeY']
            object_roll = -np.arctan2(x, y)
        else:
            object_roll = self.metadata['ObjectRoll'] * np.pi /180.

        object_tilt = -self.metadata['ObjectTilt'] * np.pi /180.

        tilt_matrix = np.eye(3)
        tilt_matrix[1][1] = tilt_matrix[2][2] = np.cos(object_tilt)
        tilt_matrix[1][2] = -np.sin(object_tilt)
        tilt_matrix[2][1] = np.sin(object_tilt)

        roll_matrix = np.eye(3)
        roll_matrix[0][0] = roll_matrix[2][2] = np.cos(object_roll)
        roll_matrix[0][2] = np.sin(object_roll)
        roll_matrix[2][0] = -np.sin(object_roll)

        # order of construction may be reversed, but unlikely to have both in a dataset
        rot_matrix = np.matmul(tilt_matrix,roll_matrix)
        rotation_axis_direction = rot_matrix.dot([0,0,1])


        origin = 'top-right'

        if ['DetectorPixelsY'] == 1:
            self._acquisition_geometry = AcquisitionGeometry.create_Cone2D(source_position=[0, 0 ],
                                                     rotation_axis_position=[-object_offset_x, self.metadata['SrcToObject']],
                                                     detector_position=[0, self.metadata['SrcToDetector'] ])
    

            self._acquisition_geometry.set_angles(angles, angle_unit='degree')
            
            self._acquisition_geometry.set_panel(self.metadata['DetectorPixelsX'], pixel_size=self.metadata['DetectorPixelSizeX'], origin=origin)

            self._acquisition_geometry.set_labels(labels=['angle', 'horizontal'])
        else:
            self._acquisition_geometry = AcquisitionGeometry.create_Cone3D(source_position=[0, 0, 0],
                                                         rotation_axis_position=[-object_offset_x, self.metadata['SrcToObject'], 0],
                                                         rotation_axis_direction=rotation_axis_direction,
                                                         detector_position=[0, self.metadata['SrcToDetector'], 0])
            self._acquisition_geometry.set_angles(angles, angle_unit='degree')
            
            self._acquisition_geometry.set_panel((self.metadata['DetectorPixelsX'], self.metadata['DetectorPixelsY']),
                               pixel_size=(self.metadata['DetectorPixelSizeX'], self.metadata['DetectorPixelSizeY']),
                               origin=origin)
        
            self._acquisition_geometry.set_labels(labels=['angle', 'vertical', 'horizontal'])


    def get_raw_data(self):
        """
        Returns a `numpy.ndarray` with the raw data in the format they are stored.
        """
        data_reader = TIFFStackReader(file_name = self._data_path, dtype=None)
        return data_reader.read()


    def get_raw_flatfield(self):
        """
        Returns a `numpy.ndarray` with the raw flat-field images in the format they are stored.
        """
        return self.metadata['WhiteLevel']



    def _set_up_normaliser(self):
        """
        Set up the Normaliser
        """
        flat_field = self.get_raw_flatfield()
        #needs to crop the normalisation images to match the cropped data.... what's the best way?
        self._normaliser = Normaliser(flat_field, None, method='default')


    def _apply_normalisation(self, data_array):
        """
        Method to apply the normalisation accessed from self._normalisation to the cropped data as a `numpy.ndarray`

        Can be overwritten if normaliser doesn't have functionality needed
        """
        self._normaliser(data_array, out = data_array)


    def _set_data_reader(self):
        """
        create the data reader
        """

        if not hasattr(self,'_data_reader'):
            self._data_reader = TIFFStackReader(self._data_path, dtype=np.float32)


    def _get_data_chunk(self, selection):
        """
        Method to read an roi of the data from disk and return an `numpy.ndarray`.

        selection is a tuple of slice objects for each dimension
        """
        
        angles = (selection[0].start, selection[0].stop, selection[0].step)
        vertical= (selection[1].start, selection[1].stop, selection[1].step)
        horizontal=(selection[2].start, selection[2].stop, selection[2].step)
        
        self._set_data_reader()
        self._data_reader.set_image_roi(vertical=vertical, horizontal=horizontal, mode='slice')
        self._data_reader.set_image_indices(angles)

        return self._data_reader.read()


