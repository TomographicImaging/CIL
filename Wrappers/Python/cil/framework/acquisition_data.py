#  Copyright 2018 United Kingdom Research and Innovation
#  Copyright 2018 The University of Manchester
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
# Joshua DM Hellier (University of Manchester) [refactorer]
import numpy
import warnings

from .labels import AcquisitionDimension, Backend, AcquisitionType
from .data_container import DataContainer
from .partitioner import Partitioner


class AcquisitionData(DataContainer, Partitioner):
    """
    DataContainer for holding 2D or 3D sinogram
    
    Parameters
    ----------
    array : numpy.ndarray or DataContainer
        The data array. Default None creates an empty array of size determined by the geometry.
    deep_copy : bool, default is True
        If True, the array will be deep copied. If False, the array will be shallow copied.
    geometry : AcquisitionGeometry
        The geometry of the data. If the dtype of the array and geometry are different, the geometry dtype will be overridden.
    
    **kwargs:
        dtype : numpy.dtype
            Specify the data type of the AcquisitionData array, this is useful if you pass None to array and want to over-ride the dtype of the geometry. 
            If an array is passed, dtype must match the dtype of the array.

    """
    __container_priority__ = 1

    @property
    def geometry(self):
        return self._geometry

    @geometry.setter
    def geometry(self, val):
        self._geometry = val

    @property
    def dimension_labels(self):
        return self.geometry.dimension_labels

    @dimension_labels.setter
    def dimension_labels(self, val):
        if val is not None:
            raise ValueError("Unable to set the dimension_labels directly. Use geometry.set_labels() instead")

    def __init__(self,
                 array = None,
                 deep_copy=True,
                 geometry = None,
                 **kwargs):

        dtype = kwargs.pop('dtype', None)
        if dtype is not None and array is not None:
            if dtype != array.dtype:
                    raise TypeError('dtype must match the array dtype got {} expected {}'.format(dtype, array.dtype))

        if geometry is None:
            raise AttributeError("AcquisitionData requires a geometry")

        labels = kwargs.pop('dimension_labels', None)
        if labels is not None and labels != geometry.dimension_labels:
                raise ValueError("Deprecated: 'dimension_labels' cannot be set with 'allocate()'. Use 'geometry.set_labels()' to modify the geometry before using allocate.")

        if array is None:
            if dtype is None:
                dtype = geometry.dtype
            array = numpy.empty(geometry.shape, dtype)
    
        elif issubclass(type(array) , DataContainer):
            array = array.as_array()

        elif issubclass(type(array) , numpy.ndarray):
            # remove singleton dimensions
            array = numpy.squeeze(array)

        else:
            raise TypeError('array must be a CIL type DataContainer or numpy.ndarray got {}'.format(type(array)))

        if array.shape != geometry.shape:
            raise ValueError('Shape mismatch got {} expected {}'.format(array.shape, geometry.shape))

        super(AcquisitionData, self).__init__(array, deep_copy, geometry=geometry,**kwargs)

        if kwargs:
            warnings.warn(f"Unused keyword arguments: {kwargs}", stacklevel=2)

    def __eq__(self, other):
        '''
        Check if two AcquisitionData objects are equal. This is done by checking if the geometry, data and dtype are equal.
        Also, if the other object is a numpy.ndarray, it will check if the data and dtype are equal.
        
        Parameters
        ----------
        other: AcquisitionData or numpy.ndarray
            The object to compare with.
        
        Returns
        -------
        bool
            True if the two objects are equal, False otherwise.
        '''

        if isinstance(other, AcquisitionData):
            if numpy.array_equal(self.as_array(), other.as_array()) \
                and self.geometry == other.geometry \
                and self.dtype == other.dtype:
                return True 
        elif numpy.array_equal(self.as_array(), other) and self.dtype==other.dtype:
            return True
        else:
            return False
        
    def _get_slice(self, **kwargs):
        '''
        Functionality of get_slice
        '''

        force = kwargs.pop('force', False)
        try:
            geometry_new = self.geometry.get_slice(**kwargs)
        except ValueError:
            if force:
                geometry_new = None
            else:
                raise ValueError ("Unable to return slice of requested AcquisitionData. Use 'force=True' to return DataContainer instead.")

        #get new data
        #if vertical = 'centre' slice convert to index and subset, this will interpolate 2 rows to get the center slice value
        if kwargs.get('vertical') == 'centre':
            dim = self.geometry.dimension_labels.index('vertical')

            centre_slice_pos = (self.geometry.shape[dim]-1) / 2.
            ind0 = int(numpy.floor(centre_slice_pos))
            w2 = centre_slice_pos - ind0
            kwargs['vertical'] = ind0
            out = DataContainer.get_slice(self, **kwargs)

            if w2 > 0:
                kwargs['vertical'] = ind0 +  1
                out2 = DataContainer.get_slice(self, **kwargs)
                out = out * (1 - w2) + out2 * w2
        else:
            out = DataContainer.get_slice(self, **kwargs)

        if len(out.shape) == 1 or geometry_new is None:
            return out
        else:
            return AcquisitionData(out.array, deep_copy=False, geometry=geometry_new)
        
    
    def get_slice(self, *args, **kwargs):
        '''
        Returns a new AcquisitionData of a single slice in the requested direction.

        Parameters
        ----------
        channel: int, optional
            index on channel dimension to slice on. If None, does not slice on this dimension.
        angle/projection: int, optional
            index on angle or projection dimension to slice on. Dimension label depends on the geometry type:
            For CONE_FLEX geometry, use 'projection'.
            For all other geometries, use 'angle'.
        vertical: int, str, optional
            If int, index on vertical dimension to slice on. If str, can be 'centre' to return the slice at the center of the vertical dimension.
        horizontal: int, optional
            index on horizontal dimension to slice on. If None, does not slice on this dimension.
        force: bool, default False
            If True, will return a DataContainer instead of an AcquisitionData when the requested slice is not reconstructable.

        Returns
        -------
        AcquisitionData
            A new AcquisitionData object containing the requested slice.
        DataContainer
            If `force=True` and the slice is not reconstructable, a DataContainer will be returned instead.

        Notes
        -----
        If there is an even number of slices in the vertical dimension and 'centre' is requested, the slice returned will be the average of the two middle slices.
        '''

        for key in kwargs:
            if self.geometry.geom_type == AcquisitionType.CONE_FLEX and key == 'angle':
                raise TypeError(f"Cannot use 'angle' for Cone3D_Flex geometry. Use 'projection' instead.")
            elif self.geometry.geom_type != AcquisitionType.CONE_FLEX and key == 'projection':
                raise TypeError(f"Cannot use 'projection' for geometries that use angles. Use 'angle' instead.")
            elif key not in AcquisitionDimension and key != 'force':
                raise TypeError(f"'{key}' not in allowed labels {AcquisitionDimension}.")
            
        if args:
            warnings.warn("Positional arguments for get_slice are deprecated. Use keyword arguments instead.", DeprecationWarning, stacklevel=2)
            
            num_args = len(args)

            if num_args > 0:
                kwargs['channel'] = args[0]

            if num_args > 1:
                if self.geometry.geom_type & AcquisitionType.CONE_FLEX:
                    kwargs['projection'] = args[1]
                else:
                    kwargs['angle'] = args[1]

            if num_args > 2:
                kwargs['vertical'] = args[2]

            if num_args > 3:
                kwargs['horizontal'] = args[3]

            if num_args > 4:
                kwargs['force'] = args[4]

        return self._get_slice(**kwargs)
    
    def get_centre_slice(self):
        '''
        Returns a new AcquisitionData of the centre slice in the vertical direction.
        '''
        return self.get_slice(vertical='centre')

    def reorder(self, order):
        '''
        Reorders the data in memory as requested. This is an in-place operation.

        Parameters
        ----------
        order: list or str
            Ordered list of labels from self.dimension_labels, or string 'astra' or 'tigre'.
        '''
        if order in Backend:
            order = AcquisitionDimension.get_order_for_engine(order, self.geometry)

        super().reorder(order)
