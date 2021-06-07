# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library (CIL) developed by CCPi 
#   (Collaborative Computational Project in Tomographic Imaging), with 
#   substantial contributions by UKRI-STFC and University of Manchester.

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from cil.framework import Processor, AcquisitionData
from cil.processors.Binner import Binner
import matplotlib.pyplot as plt
import scipy
import numpy as np
import inspect
import logging
import math

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class CofR_sobel(Processor):

    r'''CofR_sobel processor maximises the sharpness of a reconstructed slice.

    For use on data-sets that can be reconstructed with FBP.

    :param slice_index: An integer defining the vertical slice to run the algorithm on.
    :type slice_index: int, str='centre', optional
    :param FBP: A CIL FBP class imported from cil.plugins.tigre or cil.plugins.astra  
    :type FBP: class
    :param tolerance: The tolerance of the fit in pixels, the default is 1/200 of a pixel. Note this is a stopping critera, not a statement of accuracy of the algorithm.
    :type tolerance: float, default = 0.001    
    :param search_range: The range in pixels to search either side of the panel centre. If `None` the width of the panel/4 is used. 
    :type search_range: int
    :param initial_binning: The size of the bins for the initial grid. If `None` will bin the image to a step corresponding to <128 pixels. Note the fine search will be on unbinned data.
    :type initial_binning: int
    :return: returns an AcquisitionData object with an updated AcquisitionGeometry
    :rtype: AcquisitionData
    '''

    def __init__(self, slice_index='centre', FBP=None, tolerance=0.005, search_range=None, initial_binning=None):
        
        if not inspect.isclass(FBP):
            raise ValueError("Please pass a CIL FBP class from cil.plugins.tigre or cil.plugins.astra")

        kwargs = {
                    'slice_index': slice_index,
                    'FBP': FBP,
                    'tolerance': tolerance,
                    'search_range': search_range,
                    'initial_binning': initial_binning
                 }

        super(CofR_sobel, self).__init__(**kwargs)

    def check_input(self, data):
        if not isinstance(data, AcquisitionData):
            raise Exception('Processor supports only AcquisitionData')
        
        if data.geometry == None:
            raise Exception('Geometry is not defined.')

        if data.geometry.geom_type == 'cone' and self.slice_index != 'centre':
            raise ValueError("Only the centre slice is supported with this alogrithm")

        if data.geometry.channels > 1:
            raise ValueError("Only single channel data is supported with this algorithm")

        if self.slice_index != 'centre':
            try:
                int(self.slice_index)
            except:
                raise ValueError("slice_index expected to be a positive integer or the string 'centre'. Got {0}".format(self.slice_index))

            if self.slice_index >= data.get_dimension_size('vertical'):
                raise ValueError('slice_index is out of range. Must be less than {0}. Got {1}'.format(data.get_dimension_size('vertical'), self.slice_index))

        return True

    def gss(self, data, ig, search_range, tolerance):
        '''Golden section search'''
        # intervals c:cr:c where r = φ − 1=0.619... and c = 1 − r = 0.381..., φ

        phi = (1 + math.sqrt(5))*0.5
        r = phi - 1
        #1/(r+2)
        r2inv = 1/ (r+2)
        #c = 1 - r

        all_data = {}
        #set up
        sample_points = [np.nan]*4
        evaluation = [np.nan]*4

        sample_points[0] = search_range[0]
        sample_points[3] = search_range[1]

        interval = sample_points[-1] - sample_points[0]
        step_c = interval *r2inv
        sample_points[1] = search_range[0] + step_c
        sample_points[2] = search_range[1] - step_c

        for i in range(4):
            evaluation[i] = self.calculate(data, ig, sample_points[i])
            all_data[sample_points[i]] = evaluation[i]

        count = 0
        while(count < 30):
            ind = np.argmin(evaluation)
            if ind == 1:
                del sample_points[-1]
                del evaluation[-1]

                interval = sample_points[-1] - sample_points[0]
                step_c = interval *r2inv
                new_point = sample_points[0] + step_c

            elif ind == 2:
                del sample_points[0]
                del evaluation[0]

                interval = sample_points[-1] - sample_points[0]
                step_c = interval *r2inv
                new_point = sample_points[-1]- step_c

            if interval < tolerance:
                break

            sample_points.insert(ind, new_point)      
            obj = self.calculate(data, ig, new_point)
            evaluation.insert(ind, obj)
            all_data[new_point] = obj

            count +=1

        if logger.isEnabledFor(logging.DEBUG):
            print("evaluated {} points".format(len(all_data)))
            keys, values = zip(*all_data.items())
            self.plot(keys, values, ig.voxel_size_x)

        z = np.polyfit(sample_points, evaluation, 2)
        min_point = -z[1] / (2*z[0])

        if np.sign(z[0]) == 1 and min_point < sample_points[2] and min_point > sample_points[0]:
            return min_point
        else:
            ind = np.argmin(evaluation)
            return sample_points[ind]

    def calculate(self, data, ig, offset):
        ag_shift = data.geometry.copy()
        ag_shift.config.system.rotation_axis.position = [offset, 0]

        reco = self.FBP(ig, ag_shift)(data)
        return (reco*reco).sum()

    def plot(self, offsets,values, vox_size):
        x=[x / vox_size for x in offsets] 
        y=values
                 
        plt.figure()
        plt.scatter(x,y)
        plt.show()

    def get_min(self, offsets, values, ind):
        #calculate quadratic from 3 points around ind  (-1,0,1)
        a = (values[ind+1] + values[ind-1] - 2*values[ind]) * 0.5
        b = a + values[ind] - values[ind-1]
        ind_centre = -b / (2*a)+ind

        ind0 = int(ind_centre)
        w1 = ind_centre - ind0
        return (1.0 - w1) * offsets[ind0] + w1 * offsets[ind0+1]


    def process(self, out=None):

        #get slice
        data_full = self.get_input()

        if data_full.geometry.dimension == '3D':
            data = data_full.get_slice(vertical=self.slice_index)
        else:
            data = data_full

        #prepare data
        data.geometry.config.system.update_reference_frame()
        data_filtered = data.copy()
        data_filtered.fill(scipy.ndimage.sobel(data.as_array(), axis=1, mode='reflect', cval=0.0)) 

        #initial grid search
        width = data.geometry.config.panel.num_pixels[0]
        if self.search_range is None:
            self.search_range = width //4

        if self.initial_binning is None:
            self.initial_binning = min(int(np.ceil(width / 128)),8)

        if self.initial_binning > 1:
            data_temp = data.copy()
            data_temp.fill(scipy.ndimage.gaussian_filter(data.as_array(), [0,self.initial_binning//2]))
            data_temp = Binner({'horizontal':(None,None,self.initial_binning)})(data_temp)
                
            #%% sobel filter 
            data_binned_filtered = data_temp.copy()
            data_binned_filtered.fill(scipy.ndimage.sobel(data_temp.as_array(), axis=1, mode='reflect', cval=0.0))
            data_processed = data_binned_filtered
            ig = data_processed.geometry.get_ImageGeometry()
        else:
            data_processed = data_filtered
            ig = data_processed.geometry.get_ImageGeometry()

        vox_rad = self.search_range //self.initial_binning
        offsets = np.linspace(-vox_rad, vox_rad, int(2*vox_rad + 1)) * ig.voxel_size_x
        obj_vals = []

        for offset in offsets:
            obj_vals.append(self.calculate(data_processed, ig, offset))

        if logger.isEnabledFor(logging.DEBUG):
            self.plot(offsets,obj_vals,ig.voxel_size_x / self.initial_binning)

        ind = np.argmin(obj_vals)
        if ind == 0 or ind == len(obj_vals)-1:
            raise ValueError ("Unable to minimise function within set search_range")
        else:
            centre = self.get_min(offsets, obj_vals, ind)

        #fine search
        data_processed = data_filtered
        ig = data.geometry.get_ImageGeometry()  

        a = centre - ig.voxel_size_x * self.initial_binning
        b = centre + ig.voxel_size_x * self.initial_binning

        centre = self.gss(data_processed,ig, (a, b), self.tolerance *ig.voxel_size_x )

        new_geometry = data_full.geometry.copy()
        new_geometry.config.system.rotation_axis.position[0] = centre
        
        voxel_offset = centre/ig.voxel_size_x

        logger.info("Centre of rotation correction using sobel filtering with FBP")
        logger.info("Calculated from slice: %s", str(self.slice_index))
        logger.info("Applied centre of rotation shift = %f pixels", centre/ig.voxel_size_x)
        logger.info("Applied centre of rotation shift = %f units at the object", centre)

        if out is None:
            return AcquisitionData(array=data_full, deep_copy=True, geometry=new_geometry, supress_warning=True)
        else:
            out.geometry = new_geometry
