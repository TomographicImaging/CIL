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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class CofR_sobel(Processor):

    r'''CofR_sobel processor maximises the sharpness of a reconstructed slice.

    For use on data-sets that can be reconstructed with FBP.

    :param slice_index: An integer defining the vertical slice to run the algorithm on.
    :type slice_index: int, str='centre', optional
    :param FBP: A CIL FBP class imported from cil.plugins.tigre or cil.plugins.astra  
    :type FBP: class
    :param search_range: The range in pixels to search accross. If `None` the width of the panel/2 is used. 
    :type search_range: int
    :param binning: The size of the bins for the initial grid. If `None` will bin the image to a step corresponding to <256 pixels.
    :type binning: int
    :return: returns an AcquisitionData object with an updated AcquisitionGeometry
    :rtype: AcquisitionData
    '''

    def __init__(self, slice_index='centre', FBP=None, search_range=None, binning=None):
        
        if not inspect.isclass(FBP):
            ValueError("Please pass a CIL FBP class from cil.plugins.tigre or cil.plugins.astra")

        kwargs = {
                    'slice_index': slice_index,
                    'FBP': FBP,
                    'search_range': search_range,
                    'binning': binning
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
    
    def process(self, out=None):

        #%% get slice
        data_full = self.get_input()

        if data_full.geometry.dimension == '3D':
            data = data_full.get_slice(vertical=self.slice_index)
        else:
            data = data_full

        data.geometry.config.system.update_reference_frame()
        centre = data.geometry.config.system.rotation_axis.position[0]
           
        width = data.geometry.config.panel.num_pixels[0]
        if self.search_range is None:
            self.search_range = width //4

        if self.binning is None:
            binning = int(np.ceil(width / 256))
        else:
            binning = self.binning

        count = 0
        while count < 3:

            if count == 0:
                #start with pre-calculated binning
                start = -self.search_range //binning
                stop = self.search_range //binning
                total = stop - start + 1

            if count == 1:
                binning = 1
                start = -5
                stop =  +5 
                total = (stop - start)*2 + 1

            if count == 2:
                binning = 1
                start = -0.5
                stop =  +0.5
                total = (stop - start)*10 + 1

            data_processed = data.copy()
            if binning > 1:
                data_temp = data_processed.copy()
                data_processed.fill(scipy.ndimage.gaussian_filter(data_temp.as_array(), [0,binning//2]))
                data_processed = Binner({'horizontal':(None,None,binning)})(data_processed) #check behaviour
                
            #%% sobel filter 
            data_filtered = data_processed.copy()
            data_filtered.fill(scipy.ndimage.sobel(data_processed.as_array(), axis=1, mode='reflect', cval=0.0))  

            ag = data_filtered.geometry
            ig = ag.get_ImageGeometry()

            #search
            obj_vals = []
            offsets = np.linspace(start, stop, total) * ig.voxel_size_x + centre
            for offset in offsets:
                ag_shift = data_filtered.geometry.copy()
                ag_shift.config.system.rotation_axis.position = [offset, 0]

                reco = self.FBP(ig, ag_shift)(data_filtered)
                obj_val = (reco*reco).sum()
                obj_vals.append(obj_val)

            output_cor = zip(offsets, obj_vals)

            ind = np.argmin(obj_vals)
            centre = offsets[ind]

            if ind != 0 and ind != len(offsets)-1:
                #fit quadratic to 3 centre points (-1,0,1)
                a = (obj_vals[ind+1] + obj_vals[ind-1] - 2*obj_vals[ind]) * 0.5
                b = a + obj_vals[ind] - obj_vals[ind-1]
                ind_centre = -b / (2*a)+ind

                ind0 = int(ind_centre)
                w1 = ind_centre - ind0
                centre = (1.0 - w1) * offsets[ind0] + w1 * offsets[ind0+1]
                count += 1
            else:
                raise ValueError ("Unable to minimise function within set search_range")

            logger.debug("iteration: %f\nbinning: %f\ncor at: %f",count, binning, centre)

            if logger.isEnabledFor(logging.DEBUG):
                plt.figure()
                plt.scatter(binning *offsets/ig.voxel_size_x, obj_vals)
                plt.show()
        new_geometry = data_full.geometry.copy()
        new_geometry.config.system.rotation_axis.position[0] = centre
        
        logger.info("Centre of rotation correction using sobel filtering with FBP")
        logger.info("Calculated from slice: %s", str(self.slice_index))
        logger.info("Applied centre of rotation shift = %f pixels", centre/ig.voxel_size_x)
        logger.info("Applied centre of rotation shift = %f units at the object.", centre)

        if out is None:
            return AcquisitionData(array=data_full, deep_copy=True, geometry=new_geometry, supress_warning=True)
        else:
            out.geometry = new_geometry
