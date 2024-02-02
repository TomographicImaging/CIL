# -*- coding: utf-8 -*-
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

logger = logging.getLogger(__name__)

class CofR_xcorrelation(Processor):

    r'''CofR_xcorrelation processor uses the cross-correlation algorithm on a single slice between two projections at 180 degrees inteval.

    For use on parallel-beam geometry it requires two projections 180 degree apart.

    :param slice_index: An integer defining the vertical slice to run the algorithm on.
    :type slice_index: int, str='centre', optional
    :param projection_index: An integer defining the first projection the algorithm will use. The second projection at 180 degrees will be located automatically.
    :type projection_index: int, optional
    :param ang_tol: The angular tolerance in degrees between the two input projections 180degree gap
    :type ang_tol: float, optional
    :return: returns an AcquisitionData object with an updated AcquisitionGeometry
    :rtype: AcquisitionData
    '''

    def __init__(self, slice_index='centre', projection_index=0, ang_tol=0.1):
        
        kwargs = {
                    'slice_index': slice_index,
                    'ang_tol': ang_tol,
                    'projection_index': projection_index
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

        if self.projection_index >= data.geometry.config.angles.num_positions:
                raise ValueError('projection_index is out of range. Must be less than {0}. Got {1}'.format(data.geometry.config.angles.num_positions, self.projection_index))

        return True
    
    def _find_xcor_angle(self, geometry):

        angles_deg = geometry.config.angles.angle_data.copy()
        
        if geometry.config.angles.angle_unit == "radian":
            angles_deg *= 180/np.pi 

        # set the target projection to 0
        angles_deg -= angles_deg[self.projection_index]

        #keep angles in range 0 to 360
        while angles_deg.min() <0:
            angles_deg[angles_deg<0] += 360
        while angles_deg.max() >= 360:
            angles_deg[angles_deg>=360] -= 360
   
        ind = np.abs(angles_deg - 180).argmin()
        ang_diff = abs(angles_deg[ind] - angles_deg[self.projection_index])

        if abs(ang_diff-180) > self.ang_tol:
            raise ValueError('Method requires projections 180+/-{0} degrees apart, for chosen projection angle {1} found closest angle {2}.\
                             \nPick a different initial projection or increase the angular tolerance `ang_tol`.'.format(self.ang_tol, geometry.angles[self.projection_index], geometry.angles[ind]))
        else:
            return ind

    def process(self, out=None):

        data_full = self.get_input()

        if data_full.geometry.dimension == '3D':

            data = data_full.get_slice(vertical=self.slice_index)
        else:
            data = data_full

        geometry = data.geometry
        
        ind = self._find_xcor_angle(geometry)
        
        #cross correlate single slice with the 180deg one reversed
        data1 = data.get_slice(angle=self.projection_index).as_array()
        data2 = np.flip(data.get_slice(angle=ind).as_array())

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
        
        logger.info("Centre of rotation correction found using cross-correlation")
        logger.info("Calculated from slice: %s", str(self.slice_index))
        logger.info("Centre of rotation shift = %f pixels", shift)
        logger.info("Centre of rotation shift = %f units at the object", shift * geometry.config.panel.pixel_size[0])
        logger.info("Return new dataset with centred geometry")

        if out is None:
            return AcquisitionData(array = data_full, deep_copy = True, geometry = new_geometry, supress_warning=True)
        else:
            out.geometry = new_geometry
