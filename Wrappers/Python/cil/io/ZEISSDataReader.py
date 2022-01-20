from cil.framework import AcquisitionData, AcquisitionGeometry, ImageData, ImageGeometry
import numpy as np
import os
import olefile
import logging
import dxchange

class ZEISSDataReader(object):
    
    def __init__(self, 
                 **kwargs):
        '''
        Constructor
        
        :param file_name: file name to read
        :type file_name: os.path or string, default None
        :param angle_unit: describe what the unit is, angle or degree
        :type angle_unit: string, default degree
        :param logging_level: Logging messages which are less severe than level will be ignored.
        :type logging_level: int, default 40 i.e. ERROR, possible values are 0 10 20 30 40 50 https://docs.python.org/3/library/logging.html#levels
        :param roi: dictionary with roi to load 
                {'axis_0': (start, end, step), 
                 'axis_1': (start, end, step), 
                 'axis_2': (start, end, step)}
                Files are stacked along axis_0. axis_1 and axis_2 correspond
                to row and column dimensions, respectively.
                Files are stacked in alphabetic order. 
                To skip files or to change number of files to load, 
                adjust axis_0. For instance, 'axis_0': (100, 300)
                will skip first 100 files and will load 200 files.
                'axis_0': -1 is a shortcut to load all elements along axis.
                Start and end can be specified as None which is equivalent 
                to start = 0 and end = load everything to the end, respectively.
                Start and end also can be negative.
        :type roi: dictionary, default None
                    
        '''
        
        self.file_name = kwargs.get('file_name', None)
        angle_unit = kwargs.get('angle_unit', AcquisitionGeometry.DEGREE)
        level = kwargs.get('logging_level', 40)
        self.roi = kwargs.get('roi', None)
        if self.file_name is not None:
            self.set_up(file_name = self.file_name, roi = self.roi, angle_unit=angle_unit, logging_level=level)

    def set_up(self, 
               file_name = None,
               angle_unit = AcquisitionGeometry.DEGREE,
               logging_level=40,
               roi = None):
        '''Set up the reader
        
        :param file_name: file name to read
        :type file_name: os.path or string, default None
        :param slice_range: list with range to slice data, [start,stop,step(optional)]
        :type slice_range: list, default None
        :param angle_unit: describe what the unit is, angle or degree
        :type angle_unit: string, default degree
        :param logging_level: Logging messages which are less severe than level will be ignored.
        :type logging_level: int, default 40 i.e. ERROR, possible values are 0 10 20 30 40 50 https://docs.python.org/3/library/logging.html#levels
                :param roi: dictionary with roi to load 
                {'axis_0': (start, end, step), 
                 'axis_1': (start, end, step), 
                 'axis_2': (start, end, step)}
                Files are stacked along axis_0. axis_1 and axis_2 correspond
                to row and column dimensions, respectively.
                Files are stacked in alphabetic order. 
                To skip files or to change number of files to load, 
                adjust axis_0. For instance, 'axis_0': (100, 300)
                will skip first 100 files and will load 200 files.
                'axis_0': -1 is a shortcut to load all elements along axis.
                Start and end can be specified as None which is equivalent 
                to start = 0 and end = load everything to the end, respectively.
                Start and end also can be negative.
        :type roi: dictionary, default None'''

        # Set logging level for dxchange reader.py
        self.logging_level = logging_level
        logger = logging.getLogger(name='dxchange.reader')
        if logger is not None:
            logger.setLevel(self.logging_level)
        
        self.file_name = os.path.abspath(file_name)
        
        # Check if file path and supsequent file exists
        if self.file_name == None:
            raise ValueError('Path to txrm file is required.')
        
        if not(os.path.isfile(self.file_name)):
            raise FileNotFoundError('{}'.format(self.file_name))
        
        # Set type of units for theta/angle
        possible_units = [AcquisitionGeometry.DEGREE, AcquisitionGeometry.RADIAN]
        if angle_unit in possible_units:
            self.angle_unit = angle_unit
        else:
            raise ValueError('angle_unit should be one of {}'.format(possible_units))

        metadata = self.read_metadata()

        self.roi = roi
        # check roi labels and create tuple for slicing
        default_roi = {'axis_0': (0,metadata['number_of_images'],1), 
               'axis_1': (0,metadata['image_height'],1),
               'axis_2': (0,metadata['image_width'],1)}
        if self.roi:       
            for key in self.roi.keys():
                if key not in ['axis_0', 'axis_1', 'axis_2']:
                    raise Exception("Wrong label. axis_0, axis_1 and axis_2 are expected")
                elif key in default_roi.keys():
                    default_roi[key] = roi[key]
            self._roi = default_roi
            self._metadata = self.slice_metadata(metadata)
        else:
            self._roi = default_roi
            self._metadata = metadata
        
        #setup geometry using metadata
        if metadata['data geometry'] == 'acquisition':
            self._setup_acq_geometry()
        else:
            self._setup_image_geometry()

    def read_metadata(self):
        # Read one image to get the metadata
        _,metadata = dxchange.read_txrm(self.file_name,((0,1),(None),(None)))

        # convert angles to requested unit measure, Zeiss stores in radians
        if self.angle_unit == AcquisitionGeometry.DEGREE:
            metadata['thetas'] = np.degrees(metadata['thetas'])

        # Read extra metadata
        with olefile.OleFileIO(self.file_name) as ole:
            # Read source to center and detector to center distances
            StoRADistance = dxchange.reader._read_ole_arr(ole, \
                    'ImageInfo/StoRADistance', "<{0}f".format(metadata['number_of_images']))
            DtoRADistance = dxchange.reader._read_ole_arr(ole, \
                    'ImageInfo/DtoRADistance', "<{0}f".format(metadata['number_of_images']))
            
            dist_source_center = np.abs(StoRADistance[0])
            dist_center_detector = np.abs(DtoRADistance[0])

            # Read xray geometry (cone or parallel beam) and file type (TXRM or TXM)
            xray_geometry = dxchange.reader._read_ole_value(ole, 'ImageInfo/XrayGeometry', '<i')
            file_type = dxchange.reader._read_ole_value(ole, 'ImageInfo/AcquisitionMode', '<i')

            # Pixelsize loaded in metadata is really the voxel size in um.
            # We can compute the effective detector pixel size as the geometric
            # magnification times the voxel size.
            metadata['dist_source_center'] = dist_source_center
            metadata['dist_center_detector'] = dist_center_detector
            metadata['detector_pixel_size'] = ((dist_source_center+dist_center_detector)/dist_source_center)*metadata['pixel_size']

            #Configure beam and data geometries
            if xray_geometry == 1:
                print('setting up cone beam geometry')
                metadata['beam geometry'] ='cone'
            else:
                print('setting up parallel beam geometry')
                metadata['beam geometry'] = 'parallel'
            if file_type == 0:
                metadata['data geometry'] = 'acquisition'
            else:
                metadata['data geometry'] = 'image'
        return metadata
    
    def slice_metadata(self,metadata):
        '''
        Slices metadata to configure geometry before reading data
        '''
        image_slc = range(*self._roi['axis_0'])
        height_slc = range(*self._roi['axis_1'])
        width_slc = range(*self._roi['axis_2'])
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
                    .set_angles(self._metadata['thetas'],angle_unit=self.angle_unit)
        else:
            self._geometry = AcquisitionGeometry.create_Parallel3D()\
                    .set_panel([self._metadata['image_width'], self._metadata['image_height']])\
                    .set_angles(self._metadata['thetas'],angle_unit=self.angle_unit)
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
        # Load projections or slices from file
        slice_range = None
        if self.roi:
            slice_range = tuple(self._roi.values())
        print(slice_range)
        data, _ = dxchange.read_txrm(self.file_name,slice_range)
        
        if isinstance(self._geometry,AcquisitionGeometry):
            # Normalise data by flatfield
            data = data / self._metadata['reference']

            for num in range(self._metadata['number_of_images']):
                data[num,:,:] = np.roll(data[num,:,:], \
                    (int(self._metadata['x-shifts'][num]),int(self._metadata['y-shifts'][num])), \
                    axis=(1,0))
                #data = np.flip(data, 1)
            acq_data = AcquisitionData(array=data, deep_copy=False, geometry=self._geometry.copy(),suppress_warning=True)
            return acq_data
        else:
            #data = np.flip(data, 1)
            ig_data = ImageData(array=data, deep_copy=False, geometry=self._geometry.copy())
            return ig_data

    def load_projections(self):
        '''alias of read for backward compatibility'''
        return self.read()
    
    def get_geometry(self):
        '''
        Return Acquisition (TXRM) or Image (TXM) Geometry object
        '''
        return self._geometry

    def get_metadata(self):
        '''return the metadata of the file'''
        return self._metadata