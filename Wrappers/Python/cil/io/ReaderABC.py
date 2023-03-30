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

        # create the full geometry and cofigure the ROIs
        self._read_metadata()
        self._create_geometry()
        self._create_normalisation_correction()
        self.reset()



    @property
    def metadata(self):
        return self._metadata


    @abstractmethod
    def _read_metadata(self):
        """
        Populate a dictionary self._metadata = {} of used fields and values in original dataset
        
        """
        self._metadata = {}

        # read meta data and create a dictionary
        #metadata['fieldA'] = 'example'


    @abstractmethod
    def _create_geometry(self):
        """
        Create the AcquisitionGeometry self._acquisition_geometry for the full dataset using the values in self.metadata
        """

        # self._acquisition_geometry = AcquisitionGeometry.create_Parallel3D()
        # self._acquisition_geometry.set_angles([0])
        # self._acquisition_geometry.set_panel([1,1])
        # self._acquisition_geometry.set_channels(1)
        # self._acquisition_geometry.set_labels(labels='cil')


    @abstractmethod
    def get_flatfield_array(self):
        """
        return numpy array with the raw flatfield image if applicable. 
        """
        return None


    @abstractmethod
    def get_darkfield_array(self):
        """
        return numpy array with the raw darkfield image if applicable. 
        """
        return None


    @abstractmethod
    def get_data_array(self):
        """
        return numpy array with the full raw data.
        """
        return None


    @abstractmethod
    def _create_normalisation_correction(self):
        """
        Process the normalisation images to be stored for future use in self._normalisation
        """
        darkfield = self.get_darkfield_array()
        darkfield = np.mean(darkfield, axis=0)

        flatfield = self.get_flatfield_array()
        flatfield = np.mean(flatfield, axis=0)

        self._normalisation = (darkfield, 1/(flatfield-darkfield))


    @abstractmethod
    def _apply_normalisation(self, data_array):
        """
        Apply normalisation to the data respecting roi
        """
        data_array -= self._normalisation[0][tuple(self._slice_list)]
        data_array *= self._normalisation[1][tuple(self._slice_list)]


    @abstractmethod
    def _get_data(self, proj_slice=None):
        """
        The methods to access the data as a numpy array and return the roi and projections requested

        proj as a slice oject
        
        datareader - tiff, raw, dxchange, matlab
        """

        selection = self._slice_list.copy()

        if proj_slice is not None:
            selection[0] = slice(*proj_slice)

        return datareader.read(path, source_sel=tuple(selection))



    def _get_normalised_data(self, shape_out, projs=None):
        """
        The full method to read and bin the data as requested
        """

        # projection indices to iterate over
        if projs is None: 
            projs_indices = range(shape_out[0])
        elif isinstance(projs, (list,np.ndarray)):
            projs_indices = projs
        elif isinstance(projs, slice):
            projs_indices = range(shape_out[0])[slice(projs)]
        else:
            raise ValueError("Nope")            
        
        # if binning read and bin a projection at a time to reduce memory use, normalise and then and bin
        if self._bin:
            binner = Binner(self._bin_roi)

            output_array = np.empty(shape_out, dtype=np.float32)

            for count, ind in enumerate(projs_indices):
                arr = self._get_data(proj_slice=(ind,ind+1,None))
                self._apply_normalisation(arr)

                proj_unbinned=DataContainer(arr,False,['vertical','horizontal'])
                binner.set_input(proj_unbinned) 
                proj_binned = binner.get_output() 

                output_array[count:count+1,:,:]= proj_binned.array

            return output_array
        else:
            # read a single projection at a time to output array
            if isinstance(projs, (list,np.ndarray)):
                output_array = np.empty(shape_out, dtype=np.float32)
                for count, ind in enumerate(projs_indices):
                    output_array[count:count+1,:,:] = self._get_data((ind,ind+1,None))
            # read all
            else:
                output_array = self._get_data(projs)
            # normalise data
            self._apply_normalisation(output_array)

            return output_array


    def set_panel_roi(self, vertical=None, horizontal=None):
        """
        Set cropping and binning of panel
        can take an int for a single slice
        or tuple (start,stop,bin)
        """
        self._bin = False
        self._slice_list = [slice(None),slice(None),slice(None)]

        step = 1
        if vertical is not None:
            if isinstance(vertical,int):
                start = int(vertical)
                self._slice_list[1] = slice(start,start+1,1)
            elif isinstance(vertical,tuple):
                slice_obj = slice(*vertical)
                step = slice_obj.step
                self._slice_list[1] = slice(slice_obj.start,slice_obj.stop, 1)
            else:
                raise ValueError("Nope")

        if step is not None and step > 1:
            self._bin_roi['vertical'] = (None,None,step)
            self._bin = True
        else:
            self._bin_roi['vertical'] = (None,None,None)
            

        step = 1
        if horizontal is not None:
            if isinstance(horizontal,int):
                start = int(horizontal)
                self._slice_list[2] = slice(start,start+1,1)
            elif isinstance(horizontal,tuple):
                slice_obj = slice(*horizontal)
                step = slice_obj.step
                self._slice_list[2] = slice(slice_obj.start,slice_obj.stop, 1)
            else:
                raise ValueError("Nope")

        if step is not None and step > 1:
            self._bin_roi['horizontal'] = (None,None,step)
            self._bin = True
        else:
            self._bin_roi['horizontal'] = (None,None,None)


    def set_projections(self, angle_indices=None):
        """
        slice
        list
        None
        """

        if angle_indices == None:
            self._angle_indices = None

        else:
            if isinstance(angle_indices,int):
                start = int(angle_indices)
                self._angle_indices = slice(start,start+1,1)
            elif isinstance(angle_indices,tuple):
                self._angle_indices = slice(*angle_indices)
            elif isinstance(angle_indices,(list,np.ndarray)):
                self._angle_indices = angle_indices
            else:
                raise ValueError("Nope")
        

    def reset(self):
        """
        resets panel_roi and projections to default
        """
        self._angle_indices = None
        self._slice_list = [slice(None),slice(None),slice(None)]
        self._bin_roi ={'horizontal':(None,None,None),'vertical':(None,None,None)}
        self._bin = False


    def preview(self, initial_angle=0):
        """
        Display 2 projections approximately 90 degrees apart with panel_roi configuration.
        Set the initial angle in degrees
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
        
        data = self._get_normalised_data(ag.shape, projs=[idx_1,idx_2])
        show2D(data, slice_list=[0,1], title= [str(angles[idx_1])+ ag.config.angles.angle_unit, str(angles[idx_2]) +ag.config.angles.angle_unit],origin='upper-left')


    def get_geometry(self):
        """
        Returns
        -------
        AcquisitonGeometry
            Returns an AcquisitonGeometry describing your system.
            
            This respects the panel_roi and selected projections 
        """

        ag = self._acquisition_geometry.copy()

        if isinstance(self._angle_indices,slice):
            ag.config.angles.angle_data = ag.angles[(self._angle_indices)]
        elif isinstance(self._angle_indices,list):
            ag.config.angles.angle_data = np.take(ag.angles, list(self._angle_indices))

        #slice and bin geometry
        roi = { 'horizontal':(self._slice_list[2].start, self._slice_list[2].stop, self._bin_roi['horizontal'][2]),
                'vertical':(self._slice_list[1].start, self._slice_list[1].stop, self._bin_roi['vertical'][2]),
        }

        return Binner(roi)(ag)


    def read(self):
        """
        Returns
        -------
        AcquisitonData
            Returns an AcquisitonData containing your data and AcquisitionGeometry.
            
            This respects the panel_roi and selected projections 
        """

        geometry = self.get_geometry()
        data = self._get_normalised_data(geometry.shape, projs=self._angle_indices)
        return AcquisitionData(data, False, geometry)




