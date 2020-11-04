from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ccpi.framework import DataProcessor, AcquisitionData
import numpy as np
from scipy import signal

class CofR_xcorr(DataProcessor):

    def __init__(self, slice_index='centre', projection_index=0, ang_tol=0.1):
        
        kwargs = {
                    'slice_index': slice_index,
                    'ang_tol': ang_tol,
                    'projection_index': 0
                 }

        super(CofR_xcorr, self).__init__(**kwargs)

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

            if self.slice_index >= data.get_dimension_size('vertical'):
                raise ValueError('slice_index is out of range. Must be less than {0}. Got {1}'.format(data.get_dimension_size('vertical'), self.slice_index))

        if self.projection_index >= data.geometry.config.angles.num_positions:
                raise ValueError('projection_index is out of range. Must be less than {0}. Got {1}'.format(data.geometry.config.angles.num_positions, self.projection_index))

        return True
    
    def process(self, out=None):

        data = self.get_input()
        geometry = data.geometry

        angles_deg = geometry.config.angles.angle_data.copy()

        if geometry.config.angles.angle_unit == "radian":
            angles_deg *= 180/np.pi 

        #keep angles in range -180 to 180
        while angles_deg.min() <-180:
            angles_deg[angles_deg<-180] += 360

        while angles_deg.max() >= 180:
            angles_deg[angles_deg>=180] -= 360

        target = angles_deg[self.projection_index] + 180
        
        if target <-180:
            target += 360
        elif target >= 180:
            target -= 360     

        ind = np.abs(angles_deg - target).argmin()
        
        if abs(angles_deg[ind] - angles_deg[0])-180 > self.ang_tol:
            raise ValueError('Method requires projections at 180 degrees interval')

        #cross correlate single slice with the 180deg one reveresed
        data_slice = data.subset(vertical=self.slice_index)

        data1 = data_slice.subset(angle=0).as_array()
        data2 = np.flip(data_slice.subset(angle=ind).as_array())
    
        border = int(data1.size * 0.05)
        lag = np.correlate(data1[border:-border],data2[border:-border],"full")

        ind = lag.argmax()
        
        #fit quadratic to 3 centre points
        a = (lag[ind+1] + lag[ind-1] - 2*lag[ind]) * 0.5
        b = a + lag[ind] - lag[ind-1]
        quad_max = -b / (2*a) + ind

        shift = (quad_max - (lag.size-1)/2)/2
        shift = np.floor(shift *100 +0.5)/100

        new_geometry = data.geometry.copy()

        #set up new geometry
        new_geometry.config.system.rotation_axis.position[0] = shift * geometry.config.panel.pixel_size[0]
        
        print("Centre of rotation correction using cross-correlation")
        print("\tCalculated from slice: ", self.slice_index)
        print("\tApplied centre of rotation shift = ", shift, "pixels at the detector.")

        if out is None:
            return AcquisitionData(array = data, deep_copy = True, dimension_labels = new_geometry.dimension_labels, geometry = new_geometry, supress_warning=True)
        else:
            out.geometry = new_geometry
