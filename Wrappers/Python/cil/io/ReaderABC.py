from cil.framework import AcquisitionGeometry,AcquisitionData,DataContainer
import os
from abc import ABC, abstractmethod
from cil.processors import Binner
import numpy as np
from cil.utilities.display import show2D

class Reader(ABC): 
    """
    This is an abstract base class for a data reader.

    Abstract methods to be defined in child classes:
    supported_extensions=[]
    def _read_metadata(self)
    def _create_geometry(self)
    def get_flatfield_array(self)
    def get_darkfield_array(self):
    def get_data_array(self)
    def _create_normalisation_correction(self)
    def _apply_normalisation(self, data_array)
    def _get_data(self, proj_slice=None)

    """

    supported_extensions=[]

    @property
    def file_name(self):
        return self._file_name

    def __init__(self, file_name):

        # check file
        file_name_abs = os.path.abspath(file_name)
        
        if not(os.path.isfile(file_name_abs)):
            raise FileNotFoundError('{}'.format(file_name_abs))
        
        file_extension = os.path.basename(file_name_abs).split('.')[-1].lower()
        if file_extension not in self.supported_extensions:
            raise TypeError('This reader can only process files with extensions: {0}. Got {1}'.format(self.supported_extensions, file_extension))

        self._file_name = file_name_abs
        self._normalisation = False

        # create the full geometry and configure the ROIs
        self._read_metadata()

        # These should only be called once but maybe init isn't the best place
        # should still be able to access the raw data?
        self._create_geometry()
        self.reset()

    @property
    def metadata(self):
        return self._metadata

    @property
    def geometry(self):
        return self._acquisition_geometry


    @abstractmethod
    def _read_metadata(self):
        """
        Constructs a dictionary `self._metadata` of the values used from the metadata. 
        """
        self._metadata = {}
        self._metadata['fieldA'] = 'example'


    @abstractmethod
    def _create_geometry(self):
        """
        Create the `AcquisitionGeometry` `self._acquisition_geometry` that describes the full dataset
        """
        self._acquisition_geometry = AcquisitionGeometry.create_Parallel3D()
        self._acquisition_geometry.set_angles([0])
        self._acquisition_geometry.set_panel([1,1])
        self._acquisition_geometry.set_channels(1)
        self._acquisition_geometry.set_labels(labels='cil')


    @abstractmethod
    def get_raw_flatfield(self):
        """
        Returns a `numpy.ndarray` with the raw flat-field images in the format they are stored.
        """
        return None


    @abstractmethod
    def get_raw_darkfield(self):
        """
        Returns a `numpy.ndarray` with the raw dark-field images in the format they are stored.
        """
        return None


    @abstractmethod
    def get_raw_data(self):
        """
        Returns a `numpy.ndarray` with the raw data in the format they are stored.
        """
        return None


    @abstractmethod
    def _create_normalisation_correction(self):
        """
        Process the normalisation images as required and store the full scale versions 
        in self._normalisation for future use by `_apply_normalisation`  
        """
        darkfield = self.get_raw_darkfield()
        darkfield = np.mean(darkfield, axis=0)

        flatfield = self.get_raw_flatfield()
        flatfield = np.mean(flatfield, axis=0)

        self._normalisation = (darkfield, 1/(flatfield-darkfield))


    @abstractmethod
    def _apply_normalisation(self, data_array):
        """
        Method to apply the normalisation accessed from self._normalisation to the cropped data as a `numpy.ndarray`
        """
        data_array -= self._normalisation[0][self._panel_crop]
        data_array *= self._normalisation[1][self._panel_crop]


    @abstractmethod
    def _get_data(self, proj_slice=None):
        """
        Method to read the data from disk and return an `numpy.ndarray` of the cropped image dimensions.

        should handle proj as a slice, range, list or index
        """

        datareader = None
        path = None

        if proj_slice is None:
            selection = (slice(None),*self._panel_crop)
            data = datareader.read(path, source_sel=selection)

        elif isinstance(proj_slice,(range,slice)):   
            selection = (slice(*proj_slice),*self._panel_crop)
            data = datareader.read(path, source_sel=selection)
        
        elif isinstance(proj_slice, int):
            selection = (slice(proj_slice, proj_slice+1),*self._panel_crop)
            data = datareader.read(path, source_sel=selection)

        elif isinstance(proj_slice,(list,np.ndarray)):
            data = np.empty(shape=(len(proj_slice),len(self._panel_crop[0]),len(self._panel_crop[0]) ))
            for i, proj in enumerate(proj_slice):
                selection = (slice(i, i+1),*self._panel_crop)
                data[i,:,:] = datareader.read(path, source_sel=selection)
        else:
            raise ValueError("Nope")

        return data


    def _get_normalised_data(self, projs=None):
        """
        Method to read the data from disk, normalise and bin as requested. Returns an `numpy.ndarray`

        projs is None, a range or a list
        """

        # if normalisation images don't exist yet create them
        if not self._normalisation:
            self._create_normalisation_correction()
      
        output_array = self._get_data(projs)
        self._apply_normalisation(output_array)

        if self._bin:
            binner = Binner(roi={'vertical':(None,None,self._bin_roi[0]),'horizontal':(None,None,self._bin_roi[1])})

            proj_unbinned=DataContainer(output_array,False,['angle','vertical','horizontal'])
            binner.set_input(proj_unbinned) 
            output_array = binner.get_output().array

        return output_array.squeeze()


    def _parse_crop_bin(self, arg, length):
        """
        Method to parse the input roi as a int or tuple (start, stop, step) perform checks and return values
        """
        crop = slice(None,None)
        step = 1

        if arg is not None:
            if isinstance(arg,int):
                crop = slice(arg, arg+1)
                step = 1
            elif isinstance(arg,tuple):
                slice_obj = slice(*arg)
                crop = slice(slice_obj.start, slice_obj.stop)

                if slice_obj.step is None:
                    step = 1
                else:
                    step = slice_obj.step
            else:
                raise TypeError("Expected input to be an int or tuple. Got {}".format(arg))
        
        
        range_new = range(0, length)[crop]

        if len(range_new)//step < 1:
            raise ValueError("Invalid ROI selection. Cannot")  
        
        return crop, step


    def set_panel_roi(self, vertical=None, horizontal=None):
        """
        Method to configure the ROI of data to be returned as a CIL object.

        horizontal: takes an integer for a single slice, a tuple of (start, stop, step)
        vertical: tuple of (start, stop, step), or `vertical='centre'` for the centre slice

        If step is greater than 1 pixels will be averaged together.
        """

        if vertical == 'centre':
            dim = self.geometry.dimension_labels.index('vertical')
            
            centre_slice_pos = (self.geometry.shape[dim]-1) / 2.
            ind0 = int(np.floor(centre_slice_pos))

            w2 = centre_slice_pos - ind0
            if w2 == 0:
                vertical=(ind0, ind0+1, 1)
            else:
                vertical=(ind0, ind0+2, 2)

        crop_v, step_v = self._parse_crop_bin(vertical, self.geometry.pixel_num_v)
        crop_h, step_h = self._parse_crop_bin(horizontal, self.geometry.pixel_num_h)

        if step_v > 1 or step_h > 1:
            self._bin = True
        else:
            self._bin = False

        self._bin_roi = (step_v, step_h)
        self._panel_crop = (crop_v, crop_h)


    def set_projections(self, angle_indices=None):
        """
        Method to configure the angular indices to be returned as a CIL object.

        angle_indices: takes an integer for a single projections, a tuple of (start, stop, step), 
        or a list of indices.

        If step is greater than 1 pixels the data will be sliced. i.e. a step of 10 returns 1 in 10 projections.
        """      

        if angle_indices is not None:
            if isinstance(angle_indices,tuple):
                angle_indices = slice(*angle_indices)
            elif isinstance(angle_indices,(list,np.ndarray)):
                angle_indices = angle_indices
            elif isinstance(angle_indices,int):
                angle_indices = [angle_indices]
            else:
                raise ValueError("Nope")
        
            try:
                angles = self.geometry.angles[(angle_indices)]

            except IndexError:
                raise ValueError("Out of range")
            
            if angles.size < 1:
                raise ValueError(") projections selected. Please select at least 1 angle")
        self._angle_indices = angle_indices
        

    def reset(self):
        """
        Resets the configured ROI and angular indices to the full dataset
        """
        # range or list object for angles to process, defaults to None
        self._angle_indices = None

        # slice in each dimension, initialised to none
        self._panel_crop = (slice(None),slice(None))

        # number of pixels to bin in each dimension
        self._bin_roi = (1,1)

        # boolean if binned
        self._bin = False


    def preview(self, initial_angle=0):
        """
        Displays two normalised projections approximately 90 degrees apart.

        This respects the configured ROI and angular indices.

        Parameters
        ----------
        initial_angle: float
            Set the angle of the 1st projection in degrees
        """

        ag = self.get_geometry()
        angles = ag.angles.copy()


        if ag.config.angles.angle_unit == 'degree':
            ang1 = initial_angle
            ang2 = ang1+90

            #angles in range 0->360
            for i, a in enumerate(angles):
                while a < 0:
                    a += 360
                while a >= 360:
                    a -= 360
                angles[i] = a

        if ag.config.angles.angle_unit == 'radian':
            ang1 = initial_angle
            ang2 = ang1+np.pi

            #angles in range 0->2*pi
            for i, a in enumerate(angles):
                while a < 0:
                    a += 2 * np.pi
                while a >= 2*np.pi:
                    a -= 2 * np.pi
                angles[i] = a


        idx_1 = np.argmin(np.abs(angles-ang1))
        idx_2 = np.argmin(np.abs(angles-ang2))

        ag.set_angles([angles[idx_1], angles[idx_2]])
        
        data = self._get_normalised_data(projs=[idx_1,idx_2])
        show2D(data, slice_list=[0,1], title= [str(angles[idx_1])+ ag.config.angles.angle_unit, str(angles[idx_2]) +ag.config.angles.angle_unit],origin='upper-left')


    def get_geometry(self):
        """
        Method to retrieve the geometry describing your data.

        This respects the configured ROI and angular indices.

        Returns
        -------
        AcquisitionGeometry
            Returns an AcquisitionGeometry describing your system.
        """

        ag = self._acquisition_geometry.copy()

        if isinstance(self._angle_indices,slice):
            ag.config.angles.angle_data = ag.angles[(self._angle_indices)]
        elif isinstance(self._angle_indices,list):
            ag.config.angles.angle_data = np.take(ag.angles, list(self._angle_indices))

        #slice and bin geometry
        roi = { 'horizontal':(self._panel_crop[1].start, self._panel_crop[1].stop, self._bin_roi[1]),
                'vertical':(self._panel_crop[0].start, self._panel_crop[0].stop, self._bin_roi[0]),
        }

        return Binner(roi)(ag)


    def read(self):
        """
        Method to retrieve the data .

        This respects the configured ROI and angular indices.

        Returns
        -------
        AcquisitionData
            Returns an AcquisitionData containing your data and AcquisitionGeometry.
        """

        geometry = self.get_geometry()
        data = self._get_normalised_data(projs=self._angle_indices)
        return AcquisitionData(data, False, geometry)
