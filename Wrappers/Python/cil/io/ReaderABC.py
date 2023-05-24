from cil.framework import AcquisitionGeometry,AcquisitionData,DataContainer
import os
from abc import ABC, abstractmethod
from cil.processors import Binner
import numpy as np
from cil.utilities.display import show2D
from cil.processors import Normaliser
from copy import deepcopy
import weakref
import importlib

class ReaderABC(ABC): 
    """
    This is an abstract base class for a data reader that reads data, metadata and returns a CIL AcquisitionData.

    If you derive from this class you can build a reader that returns a CIL AcquisitionData object.

    All abstract methods must be defined.

    Abstract methods to be defined in child classes:

    def _supported_extensions(self):
    def _read_metadata(self):
    def _create_full_geometry(self):
    def read_data(self, dtype=np.float32, roi=(slice(None),slice(None),slice(None))):

    Methods that may be reimplemented in a child. These return a default that may not be suitable for all use cases:
    def get_raw_flatfield(self):
    def get_raw_darkfield(self):
    def _set_up_normaliser(self):
    """

    class data_handler(object):
        """
        This class controls the reading, casting, normalising and caching of the dataset.
        It should not need modification as long as it is configured with 'read_data_method' and 
        'apply_normalisation_method'.
        """

        def __init__(self, read_data_method, apply_normalisation_method):
            self.read_data = read_data_method
            self.apply_normalisation = apply_normalisation_method

            self._array = None
            self.dtype = None
            self.roi = None
            self.normalised = False


        @property
        def array(self):
            if self._array is None:
                return None
            else:
                return self._array()
        
        @array.setter
        def array(self, val):
            self._array = weakref.ref(val)


        def get_data(self, dtype=np.float32, roi=None, normalise=True):
            """
            Caches the previous read when possible
            """
            # cached must be same roi, but can be not normalised and castable dtype
            if self.array is not None and ((normalise == self.normalised) or normalise)  and \
                self.roi==roi and np.can_cast(self.dtype,dtype,casting='safe'):

                self.array.astype(dtype, casting='safe',copy=False)
                self.dtype = self.array.dtype

                if not self.normalised and normalise:
                    self.apply_normalisation(self.array)
                    self.normalised = True

                array = self.array

            else:
                array = self.read_data(dtype, roi)
                array = np.asarray(array, dtype=dtype)
                self.array = array
                self.dtype = array.dtype
                self.roi = roi
                self.normalised = normalise

                if normalise:
                    self.apply_normalisation(self.array)

            return array


    def __init__(self, file_name):

        self._data_handle = self.data_handler(self.read_data, self._apply_normalisation)
        self.file_name = file_name
        self._normalise = True
        self.reset()


    @property
    def file_name(self):
        return self._file_name
    

    @file_name.setter
    def file_name(self, val):
        file_name_abs = os.path.abspath(val)
        
        if not(os.path.isfile(file_name_abs)):
            raise FileNotFoundError('{}'.format(file_name_abs))
        
        file_extension = os.path.basename(file_name_abs).split('.')[-1].lower()
        if file_extension not in self._supported_extensions:
            raise TypeError('This reader can only process files with extensions: {0}. Got {1}'.format(self._supported_extensions, file_extension))

        self._file_name = val

        #if filename changes then reset the file dependent members
        self._acquisition_geometry = False
        self._normaliser = False
        self._metadata = False
        self._data_handle._array = None


    @property
    def full_geometry(self):
        if not self._acquisition_geometry:
            self._create_full_geometry()
        return self._acquisition_geometry


    @property
    def metadata(self):
        if not self._metadata:
            self._read_metadata()
        return self._metadata
    

    @property
    @abstractmethod
    def _supported_extensions(self):
        """A list of file extensions supported by this reader"""
        return []

    
    @abstractmethod
    def _read_metadata(self):
        """
        Constructs a dictionary `self._metadata` of the values used from the dataset meta data. 
        """
        self._metadata = {}
        self._metadata['fieldA'] = 'example'


    @abstractmethod
    def _create_full_geometry(self):
        """
        Create the `AcquisitionGeometry` `self._acquisition_geometry` that describes the full dataset.

        This should use the values from `self._metadata` where possible.
        """
        self._acquisition_geometry = AcquisitionGeometry.create_Parallel3D()
        self._acquisition_geometry.set_angles(self._metadata['angles'])
        self._acquisition_geometry.set_panel([1,1])
        self._acquisition_geometry.set_channels(1)
        self._acquisition_geometry.set_labels(labels='cil')


    @abstractmethod
    def read_data(self, dtype=np.float32, roi=(slice(None),slice(None),slice(None))):
        """
        Method to read the data in. Can use a CIL reader or another library custom version.

        Should be able to read in an roi of the data.        
        """
        if not hasattr(self,'_data_reader'):
            self._data_reader = TIFFStackReader(file_name=self._data_path)

        self._data_reader.dtype = dtype
        self._data_reader.set_roi(roi)
        return self._data_reader.read()
        
  
    def get_raw_flatfield(self):
        """
        Returns a `numpy.ndarray` with the raw flat-field images in the format they are stored.
        """
        return None


    def get_raw_darkfield(self):
        """
        Returns a `numpy.ndarray` with the raw dark-field images in the format they are stored.
        """
        return None


    def _set_up_normaliser(self):
        """
        Set up the Normaliser
        """
        self._normaliser = Normaliser(self.get_raw_flatfield(), self.get_raw_darkfield(), method='default')


    def set_normalisation(self, normalise=True):
        """
        Toggle return of normalised/un-normalised data
        """
        self._normalise = normalise


    def _apply_normalisation(self, data_array):
        """
        Method to apply the normalisation accessed from self._normalisation to the cropped data as a `numpy.ndarray`

        Can be overwritten if normaliser doesn't have functionality needed
        """
        self._normaliser(data_array, out = data_array)


    def get_raw_data(self):
        """
        Get the raw data array if not already in memory
        """
        return self._data_handle.get_data(dtype=None, roi=None, normalise=False)


    def read(self):
        """
        Method to retrieve the data .

        This respects the configured ROI and angular indices.

        Returns
        -------
        AcquisitionData
            Returns an AcquisitionData containing your data and AcquisitionGeometry.
        """

        data = self._get_data()
        return AcquisitionData(data, False, self.full_geometry)



    def _get_data_array(self, selection):
        """
        Method to read an roi of the data from disk and return an `numpy.ndarray`.

        selection is a tuple of slice objects for each dimension
        """
        return self._data_handle.get_data(dtype=np.float32, roi=selection, normalise=self._normalise)


    def _set_normaliser_roi(self):
        self._normaliser_roi = deepcopy(self._normaliser)

        # this is going to break if dimensions aren't 3
        if isinstance(self._normaliser_roi._scale, np.ndarray):
            self._normaliser_roi._scale = self._normaliser_roi._scale[self._panel_crop]

        if isinstance(self._normaliser_roi._offset, np.ndarray):
            self._normaliser_roi._offset = self._normaliser_roi._offset[self._panel_crop]


    def _get_data(self, projection_indices=None):
        """
        Method to read the data from disk, normalise and bin as requested. Returns an `numpy.ndarray`

        if projection_indices is None will use based on set_angles
        """

        # if normaliser doesn't exist yet create it
        if self._normalise:
            if not self._normaliser:
                self._set_up_normaliser()

            # update normaliser for new roi
            self._set_normaliser_roi()

        # if override default (this is used by preview)
        if projection_indices is None:
            indices = self._indices
        else:
            indices = projection_indices

        if indices is None: 
            selection = (slice(0, self.full_geometry.num_projections), *self._panel_crop)
            output_array = self._get_data_array(selection)

        elif isinstance(indices,(range,slice)):   
            selection = (slice(indices.start, indices.stop, indices.step),*self._panel_crop)
            output_array = self._get_data_array(selection)

        elif isinstance(indices, int):
            selection = (slice(indices, indices+1),*self._panel_crop)
            output_array = self._get_data_array(selection)

        elif isinstance(indices,(list,np.ndarray)):

            # need to make this shape robust
            output_array = np.empty((len(indices), *self.full_geometry.shape[1::]), dtype=np.float32)

            i = 0
            while i < len(indices):

                ind_start = i
                while ((i+1) < len(indices)) and (indices[i] + 1 == indices[i+1]):
                    i+= 1

                i+=1
                selection = (slice(indices[ind_start], indices[ind_start] + i-ind_start),*self._panel_crop)
                output_array[ind_start:ind_start+i-ind_start,:,:] = self._get_data_array(selection)


        else:
            raise ValueError("Nope")

        #what if sliced and reduced dimensions?
        proj_unbinned = DataContainer(output_array,False, self.full_geometry.dimension_labels)

        if self._bin:
            binner = Binner(roi={'vertical':(None,None,self._bin_roi[0]),'horizontal':(None,None,self._bin_roi[1])})
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
            dim = self.full_geometry.dimension_labels.index('vertical')
            
            centre_slice_pos = (self.full_geometry.shape[dim]-1) / 2.
            ind0 = int(np.floor(centre_slice_pos))

            w2 = centre_slice_pos - ind0
            if w2 == 0:
                vertical=(ind0, ind0+1, 1)
            else:
                vertical=(ind0, ind0+2, 2)

        crop_v, step_v = self._parse_crop_bin(vertical, self.full_geometry.pixel_num_v)
        crop_h, step_h = self._parse_crop_bin(horizontal, self.full_geometry.pixel_num_h)

        if step_v > 1 or step_h > 1:
            self._bin = True
        else:
            self._bin = False

        self._bin_roi = (step_v, step_h)
        self._panel_crop = (crop_v, crop_h)


    def set_angles(self, indices=None):
        """
        Method to configure the angular indices to be returned as a CIL object.

        indices: takes an integer for a single projections, a tuple of (start, stop, step), 
        or a list of indices.

        If step is greater than 1 pixels the data will be sliced. i.e. a step of 10 returns 1 in 10 projections.
        """      

        if indices is not None:
            if isinstance(indices,tuple):
                indices = slice(*indices)
            elif isinstance(indices,(list,np.ndarray)):
                indices = indices
            elif isinstance(indices,int):
                indices = [indices]
            else:
                raise ValueError("Nope")
        
            try:
                angles = self.full_geometry.angles[(indices)]

            except IndexError:
                raise ValueError("Out of range")
            
            if angles.size < 1:
                raise ValueError(") projections selected. Please select at least 1 angle")
        self._indices = indices
        

    def reset(self):
        """
        Resets the configured ROI and angular indices to the full dataset
        """
        # range or list object for angles to process, defaults to None
        self._indices = None

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
        
        # overide projectsions to be read
        data = self._get_data(projection_indices=[idx_1,idx_2])
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

        ag = self.full_geometry.copy()

        if isinstance(self._indices,slice):
            ag.config.angles.angle_data = ag.angles[(self._indices)]
        elif isinstance(self._indices,list):
            ag.config.angles.angle_data = np.take(ag.angles, list(self._indices))

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
        data = self._get_data()
        return AcquisitionData(data, False, geometry)
