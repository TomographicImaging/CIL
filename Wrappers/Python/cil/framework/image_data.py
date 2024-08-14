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

from .data_container import DataContainer


class ImageData(DataContainer):
    '''DataContainer for holding 2D or 3D DataContainer'''
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
                 deep_copy=False,
                 geometry=None,
                 **kwargs):

        dtype = kwargs.get('dtype', numpy.float32)


        if geometry is None:
            raise AttributeError("ImageData requires a geometry")


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
            raise ValueError('Shape mismatch {} {}'.format(array.shape, geometry.shape))

        if array.ndim not in [2,3,4]:
            raise ValueError('Number of dimensions are not 2 or 3 or 4 : {0}'.format(array.ndim))

        super(ImageData, self).__init__(array, deep_copy, geometry=geometry, **kwargs)


    def get_slice(self,channel=None, vertical=None, horizontal_x=None, horizontal_y=None, force=False):
        '''
        Returns a new ImageData of a single slice of in the requested direction.
        '''
        try:
            geometry_new = self.geometry.get_slice(channel=channel, vertical=vertical, horizontal_x=horizontal_x, horizontal_y=horizontal_y)
        except ValueError:
            if force:
                geometry_new = None
            else:
                raise ValueError ("Unable to return slice of requested ImageData. Use 'force=True' to return DataContainer instead.")

        #if vertical = 'centre' slice convert to index and subset, this will interpolate 2 rows to get the center slice value
        if vertical == 'centre':
            dim = self.geometry.dimension_labels.index('vertical')
            centre_slice_pos = (self.geometry.shape[dim]-1) / 2.
            ind0 = int(numpy.floor(centre_slice_pos))

            w2 = centre_slice_pos - ind0
            out = DataContainer.get_slice(self, channel=channel, vertical=ind0, horizontal_x=horizontal_x, horizontal_y=horizontal_y)

            if w2 > 0:
                out2 = DataContainer.get_slice(self, channel=channel, vertical=ind0 + 1, horizontal_x=horizontal_x, horizontal_y=horizontal_y)
                out = out * (1 - w2) + out2 * w2
        else:
            out = DataContainer.get_slice(self, channel=channel, vertical=vertical, horizontal_x=horizontal_x, horizontal_y=horizontal_y)

        if len(out.shape) == 1 or geometry_new is None:
            return out
        else:
            return ImageData(out.array, deep_copy=False, geometry=geometry_new, suppress_warning=True)


    def apply_circular_mask(self, radius=0.99, in_place=True):
        """

        Apply a circular mask to the horizontal_x and horizontal_y slices. Values outside this mask will be set to zero.

        This will most commonly be used to mask edge artefacts from standard CT reconstructions with FBP.

        Parameters
        ----------
        radius : float, default 0.99
            radius of mask by percentage of size of horizontal_x or horizontal_y, whichever is greater

        in_place : boolean, default True
            If `True` masks the current data, if `False` returns a new `ImageData` object.


        Returns
        -------
        ImageData
            If `in_place = False` returns a new ImageData object with the masked data

        """
        ig = self.geometry

        # grid
        y_range = (ig.voxel_num_y-1)/2
        x_range = (ig.voxel_num_x-1)/2

        Y, X = numpy.ogrid[-y_range:y_range+1,-x_range:x_range+1]

        # use centre from geometry in units distance to account for aspect ratio of pixels
        dist_from_center = numpy.sqrt((X*ig.voxel_size_x+ ig.center_x)**2 + (Y*ig.voxel_size_y+ig.center_y)**2)

        size_x = ig.voxel_num_x * ig.voxel_size_x
        size_y = ig.voxel_num_y * ig.voxel_size_y

        if size_x > size_y:
            radius_applied =radius * size_x/2
        else:
            radius_applied =radius * size_y/2

        # approximate the voxel as a circle and get the radius
        # ie voxel area = 1, circle of area=1 has r = 0.56
        r=((ig.voxel_size_x * ig.voxel_size_y )/numpy.pi)**(1/2)

        # we have the voxel centre distance to mask. voxels with distance greater than |r| are fully inside or outside.
        # values on the border region between -r and r are preserved
        mask =(radius_applied-dist_from_center).clip(-r,r)

        #  rescale to -pi/2->+pi/2
        mask *= (0.5*numpy.pi)/r

        # the sin of the linear distance gives us an approximation of area of the circle to include in the mask
        numpy.sin(mask, out = mask)

        # rescale the data 0 - 1
        mask = 0.5 + mask * 0.5

        # reorder dataset so 'horizontal_y' and 'horizontal_x' are the final dimensions
        labels_orig = self.dimension_labels
        labels = list(labels_orig)

        labels.remove('horizontal_y')
        labels.remove('horizontal_x')
        labels.append('horizontal_y')
        labels.append('horizontal_x')


        if in_place == True:
            self.reorder(labels)
            numpy.multiply(self.array, mask, out=self.array)
            self.reorder(labels_orig)

        else:
            image_data_out = self.copy()
            image_data_out.reorder(labels)
            numpy.multiply(image_data_out.array, mask, out=image_data_out.array)
            image_data_out.reorder(labels_orig)

            return image_data_out
