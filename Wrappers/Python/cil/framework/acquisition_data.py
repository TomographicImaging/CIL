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
import numpy

from .labels import AcquisitionDimension, Backend
from .data_container import DataContainer
from .partitioner import Partitioner


class AcquisitionData(DataContainer, Partitioner):
    ''''''
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
        """
        DataContainer for holding 2D or 3D sinogram
        
        Parameters
        ----------
        array : numpy.ndarray or DataContainer
            The data array.
        deep_copy : bool, default is True
            If True, the array will be deep copied. If False, the array will be shallow copied.
        geometry : AcquisitionGeometry
            The geometry of the data. If the dtype of the array and geometry are different, the geometry dtype will be overridden.
        """

        dtype = kwargs.get('dtype', numpy.float32)

        if geometry is None:
            raise AttributeError("AcquisitionData requires a geometry")

        labels = kwargs.get('dimension_labels', None)
        if labels is not None and labels != geometry.dimension_labels:
                raise ValueError("Deprecated: 'dimension_labels' cannot be set with 'allocate()'. Use 'geometry.set_labels()' to modify the geometry before using allocate.")

        if array is None:
            array = numpy.empty(geometry.shape, dtype=dtype)
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

    def get_slice(self,channel=None, angle=None, vertical=None, horizontal=None, force=False):
        '''Returns a new dataset of a single slice in the requested direction.'''
        try:
            geometry_new = self.geometry.get_slice(channel=channel, angle=angle, vertical=vertical, horizontal=horizontal)
        except ValueError:
            if force:
                geometry_new = None
            else:
                raise ValueError ("Unable to return slice of requested AcquisitionData. Use 'force=True' to return DataContainer instead.")

        #get new data
        #if vertical = 'centre' slice convert to index and subset, this will interpolate 2 rows to get the center slice value
        if vertical == 'centre':
            dim = self.geometry.dimension_labels.index('vertical')

            centre_slice_pos = (self.geometry.shape[dim]-1) / 2.
            ind0 = int(numpy.floor(centre_slice_pos))
            w2 = centre_slice_pos - ind0
            out = DataContainer.get_slice(self, channel=channel, angle=angle, vertical=ind0, horizontal=horizontal)

            if w2 > 0:
                out2 = DataContainer.get_slice(self, channel=channel, angle=angle, vertical=ind0 + 1, horizontal=horizontal)
                out = out * (1 - w2) + out2 * w2
        else:
            out = DataContainer.get_slice(self, channel=channel, angle=angle, vertical=vertical, horizontal=horizontal)

        if len(out.shape) == 1 or geometry_new is None:
            return out
        else:
            return AcquisitionData(out.array, deep_copy=False, geometry=geometry_new, suppress_warning=True)

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
