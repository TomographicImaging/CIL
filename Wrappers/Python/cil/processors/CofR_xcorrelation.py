#  Copyright 2020 United Kingdom Research and Innovation
#  Copyright 2020 The University of Manchester
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

from cil.framework import Processor, AcquisitionData
import numpy as np

import logging

log = logging.getLogger(__name__)

class CofR_xcorrelation(Processor):

    r'''CofR_xcorrelation processor uses the cross-correlation algorithm on a single slice between two projections at 180 degrees inteval.

    For use on parallel-beam geometry it requires two projections 180 degree apart.

    Parameters
    ----------
    slice_index: int or str, optional
        An integer defining the vertical slice to run the algorithm on or string='centre' specifying the central slice should be used (default is 'centre')
    projection_index: int or list/tuple of ints, optional
        The index of the first projection the cross correlation algorithm will use, where the second projection is chosen as the projection closest to 180 degrees from this.
        Or a list/tuple of ints specifying the two indices to be used for cross correlation (default is 0)
    ang_tol: float, optional
        The angular tolerance in degrees between the two input projections 180 degree gap (default is 0.1)

    Returns
    -------
    AcquisitionData
        object with an AcquisitionGeometry with updated centre of rotation

    '''

    def __init__(self, slice_index='centre', projection_index=0, ang_tol=0.1):

        kwargs = {
                    'slice_index': slice_index,
                    'ang_tol': ang_tol,
                    'projection_index': projection_index,
                    '_indices' : None,
                 }

        super(CofR_xcorrelation, self).__init__(**kwargs)

    def check_input(self, data):
        if not isinstance(data, AcquisitionData):
            raise Exception('Processor supports only AcquisitionData')

        if data.geometry == None:
            raise Exception('Geometry is not defined.')

        if data.geometry.geom_type == 'cone':
            raise ValueError("Only parallel-beam data is supported with this algorithm")

        if data.geometry.channels > 1:
            raise ValueError("Only single channel data is supported with this algorithm")

        if self.slice_index != 'centre':
            try:
                int(self.slice_index)
            except:
                raise ValueError("slice_index expected to be a positive integer or the string 'centre'. Got {0}".format(self.slice_index))

            if self.slice_index < 0 or self.slice_index >= data.get_dimension_size('vertical'):
                raise ValueError('slice_index is out of range. Must be in range 0-{0}. Got {1}'.format(data.get_dimension_size('vertical'), self.slice_index))

        # check if projection_index is either a tuple or list of length 2
        try:
            len_check = ( len(self.projection_index) == 2 )
            index = list(self.projection_index)
        # if projection_index does not have a length, put the object in a list
        except:
            len_check = True
            index = [self.projection_index]

        if not len_check:
            raise ValueError('Expected projection_index to be an int or list/tuple of 2 ints, got {0}'.format(self.projection_index))

        for angle in index:
            # check if all the indices are int
            if not isinstance(angle, int):
                raise ValueError('Expected projection_index to be an int, got {0}'.format(type(angle)))

            # check if all the indices are in range 0 to the number of angles
            if angle< 0  or angle>=data.geometry.config.angles.num_positions:
                raise ValueError('projection_index is out of range. Must be between 0 and {0}. Got {1}'.format(data.geometry.config.angles.num_positions-1, self.projection_index))

        angles_deg = data.geometry.config.angles.angle_data.copy()
        if data.geometry.config.angles.angle_unit == "radian":
                angles_deg *= 180/np.pi

        # if there is only 1 index specified, find the angle in the list closest to 180 degrees away from the index
        if len(index) == 1:
            new_index = CofR_xcorrelation._return_180_index(angles_deg, self.projection_index)
            index.append(new_index)

        # check the two angles are 180 degrees apart within the specified tolerance
        ang_diff = (angles_deg[index[0]] - angles_deg[index[1]]) % 360
        if abs(ang_diff-180) > self.ang_tol:
            if isinstance(self.projection_index, (list, tuple)):
                raise ValueError('Method requires projections 180+/-{0} degrees apart. The projection angle {1} and angle {2} provided are not within this tolerance.\
                            \nPick different projections or increase the angular tolerance ang_tol.'.format(self.ang_tol, data.geometry.angles[index[0]], data.geometry.angles[index[1]]))
            else:
                raise ValueError('Method requires projections 180+/-{0} degrees apart, for chosen projection angle {1} found closest angle {2}.\
                            \nPick a different initial projection or increase the angular tolerance `ang_tol`.'.format(self.ang_tol, data.geometry.angles[index[0]], data.geometry.angles[index[1]]))

        self._indices = index

        return True

    @staticmethod
    def _return_180_index(angles_deg, initial_index):
        '''
        Finds the index of the angle closest to 180 degrees from a specified initial angle, from a list of angles in degrees

        '''
        # set the target projection to 0
        angles_deg -= angles_deg[initial_index]

        #keep angles in range 0 to 360
        angles_deg = angles_deg % 360

        return np.abs(angles_deg - 180).argmin()

    def process(self, out=None):

        data_full = self.get_input()

        if data_full.geometry.dimension == '3D':

            data = data_full.get_slice(vertical=self.slice_index)
        else:
            data = data_full

        geometry = data.geometry

        #cross correlate single slice with the 180deg one reversed
        data1 = data.get_slice(angle=self._indices[0]).as_array()
        data2 = np.flip(data.get_slice(angle=self._indices[1]).as_array())

        border = int(data1.size * 0.05)
        lag = np.correlate(data1[border:-border],data2[border:-border],"full")

        ind = lag.argmax()

        #fit quadratic to 3 centre points
        a = (lag[ind+1] + lag[ind-1] - 2*lag[ind]) * 0.5
        b = a + lag[ind] - lag[ind-1]
        quad_max = -b / (2*a) + ind

        shift = (quad_max - (lag.size-1)/2)/2
        shift = np.floor(shift *100 +0.5)/100

        new_geometry = data_full.geometry.copy()

        #set up new geometry
        new_geometry.config.system.rotation_axis.position[0] = shift * geometry.config.panel.pixel_size[0]

        log.info("Centre of rotation correction found using cross-correlation")
        log.info("Calculated from slice: %s", str(self.slice_index))
        log.info("Centre of rotation shift = %f pixels", shift)
        log.info("Centre of rotation shift = %f units at the object", shift * geometry.config.panel.pixel_size[0])
        log.info("Return new dataset with centred geometry")

        if out is None:
            return AcquisitionData(array = data_full, deep_copy = True, geometry = new_geometry, supress_warning=True)
        else:
            out.geometry = new_geometry
            return out
