import numpy

from .DataContainer import DataContainer
from .partitioner import Partitioner

class AcquisitionData(DataContainer, Partitioner):
    '''DataContainer for holding 2D or 3D sinogram'''
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
            pass
        else:
            raise TypeError('array must be a CIL type DataContainer or numpy.ndarray got {}'.format(type(array)))

        if array.shape != geometry.shape:
            raise ValueError('Shape mismatch got {} expected {}'.format(array.shape, geometry.shape))

        super(AcquisitionData, self).__init__(array, deep_copy, geometry=geometry,**kwargs)


    def get_slice(self,channel=None, angle=None, vertical=None, horizontal=None, force=False):
        '''
        Returns a new dataset of a single slice of in the requested direction. \
        '''
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
