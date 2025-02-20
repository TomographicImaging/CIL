# -*- coding: utf-8 -*-
#  Copyright 2024 United Kingdom Research and Innovation
#  Copyright 2024 The University of Manchester
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


from cil.framework import DataProcessor, AcquisitionData, ImageData, ImageGeometry, DataContainer, AcquisitionGeometry
import numpy
from cil.utilities.display import show2D
import numpy as np
import time
import math

def xy_to_index(matrix, x, y):
    """
    Transform (x, y) coordinates in a 2D array to their corresponding index in the flattened array.

    Args:
        matrix (list): The input 2D array.
        x (int): The row index of the element.
        y (int): The column index of the element.

    Returns:
        int: The index of the element in the flattened array.
    """
    return x * matrix.shape[1] + y

class BadPixelCorrector(DataProcessor):
    r'''
    
    '''

    def __init__(self, mask):
        
        r'''Processor to correct bad pixels in an AcquisitionData, by replacing with the weighted mean value of unmasked nearest 
        neighbours (including diagonals) in the projection
        Requires the data to be ordered so that horizontal is the last dimension and vertical (if present) is the second last dimension.

        Parameters
        ----------
        mask : DataContainer, numpy.ndarray
            A boolean array with the same dimensions as the size of a projection in the input data (i.e. 1D or 2D), 
            where 'False' represents masked values.
            Mask can be generated using 'MaskGenerator' processor.
            Axis labels must contain 'horizontal' and optionally 'vertical'.
        '''

        kwargs = {'mask': mask}

        super(BadPixelCorrector, self).__init__(**kwargs)
    
    def check_input(self, data):
        if self.mask is None:
            raise ValueError('Please, provide a mask.')
        
        if not isinstance(self.mask, (DataContainer, numpy.ndarray)):
            raise TypeError('Mask must be a DataContainer or a numpy array.')
        
        if not isinstance(data, (AcquisitionData)):
            raise TypeError('Input data must be an AcquisitionData')
        
        # Check that horizontal and vertical (if present) are the final two dimensions:

        labels = data.geometry.dimension_labels
        if 'horizontal' not in labels:
            raise ValueError('Data must have a horizontal dimension')
        if labels[-1] != 'horizontal':
            raise ValueError('Horizontal dimension must be the last dimension')
        if 'vertical' in labels:
            if labels[-2] != 'vertical':
                raise ValueError('Vertical dimension must be the second last dimension')


        if isinstance(self.mask, DataContainer):
            mask_labels = self.mask.dimension_labels
            # do not allow anything but horizontal and vertical as the axis labels for the mask:
            for mask_label in mask_labels:
                if mask_label not in ['horizontal', 'vertical']:
                    raise ValueError('Mask must have only horizontal and vertical dimensions')
            if 'horizontal' not in mask_labels:
                raise ValueError('Mask must have a horizontal dimension')
            if mask_labels[-1] != 'horizontal':
                raise ValueError('Horizontal dimension must be the last dimension')
            if 'vertical' in mask_labels:
                if mask_labels[-2] != 'vertical':
                    raise ValueError('Vertical dimension must be the second last dimension')
        
        # Check that the shapes match:
        proj_shape = self._get_proj_shape(data)
   
        if proj_shape != self.mask.shape:
            raise ValueError(f"Projection and Mask shapes do not match: {proj_shape} != {self.mask.shape}")

        return True
    
    def _get_proj_shape(self, data):
        '''
        Get the shape of a single projection from the data.
        This is the shape of the data with the channel and angle dimensions removed.

        Parameters
        ----------
        data : AcquisitionData
            The input data.
        
        Returns
        -------
        tuple
            The shape of a single projection.
        '''

        channel = True if 'channel' in data.dimension_labels else None
        angle = True if 'angle' in data.dimension_labels else None

        try:
            proj_shape = data.get_slice(channel=channel, angle=angle).shape
        except Exception:
            # if we have only one angle and one channel, we can't use get_slice  
            proj_shape = data.shape

        return proj_shape

    def _get_neighbours_and_weights(self, mask_arr):
        """
        Get the neighbours (including diagonal) and weights for each masked pixel in the mask array.
        
        Parameters
        ----------
        mask_arr : numpy.ndarray
            The mask array with masked pixels as 'False'.

        Returns
        -------
        dict
            A dictionary with masked pixel coordinates as keys and a dictionary with 'neighbours' and 'weights' as values.
            'neighbours' and 'weights' are lists of the coordinates of the unmasked neighbours and their weights respectively.
        """
        diag_weight = 1/np.sqrt(2) # to be used in weighting mean of diagonal neighbours
        masked_pixel_neighbours = {}
        for coord in masked_pixels: 
            # Get all neighbours
            neighbours = []
            weights = []
            if len(mask_arr.shape)== 1:
                if coord > 0:
                    neighbours.append(coord-1)
                    weights.append(1)
                if coord < mask_arr.shape[0]-1:
                    neighbours.append(coord+1)
                    weights.append(1)
            else:
                if coord % mask_arr.shape[1]>0:
                    # Left neighbour
                    neighbours.append(coord - 1)
                    weights.append(1)
                if coord > (mask_arr.shape[1]-1):
                    # Upper neighbour
                    neighbours.append(coord - mask_arr.shape[1])
                    weights.append(1)
                if coord < (mask_arr.shape[0]*mask_arr.shape[1] - mask_arr.shape[1]):
                    # Lower neighbour
                    neighbours.append(coord + mask_arr.shape[1])
                    weights.append(1)
                if (coord + 1) % mask_arr.shape[1] >0:
                    # Right neighbour
                    neighbours.append(coord + 1)
                    weights.append(1)
                if coord < (mask_arr.shape[0]*mask_arr.shape[1] - mask_arr.shape[1]) and coord % mask_arr.shape[1]>0:
                    # Diagonal lower left neighbour
                    neighbours.append(coord + mask_arr.shape[1] - 1)
                    weights.append(diag_weight)
                if coord > (mask_arr.shape[1]-1) and (coord + 1) % mask_arr.shape[1] >0:
                    # Diagonal Upper right neighbour
                    neighbours.append(coord - mask_arr.shape[1]+1)
                    weights.append(diag_weight)
                if coord > mask_arr.shape[1] and coord % mask_arr.shape[1]>0:
                    #Diagonal upper left neighbour
                    neighbours.append(coord - mask_arr.shape[1] -1)
                    weights.append(diag_weight)
                if coord < (mask_arr.shape[0]*mask_arr.shape[1] - mask_arr.shape[1]) and (coord + 1) % mask_arr.shape[1] >0:
                    # Diagonal lower right neighbour
                    neighbours.append(coord + mask_arr.shape[1] + 1)
                    weights.append(diag_weight)
                
            masked_pixel_neighbours[coord] = {'neighbours': neighbours, 'weights': weights}
        return masked_pixel_neighbours


    def process(self, out=None):
        
        data = self.get_input()
        
        return_arr = False
        if out is None:
            out = data.copy()
            return_arr = True
        try:
            mask_arr = self.mask.as_array()
        except:
            mask_arr = self.mask

        mask_arr = numpy.array(mask_arr, dtype=bool)
        mask_invert = ~mask_arr

        # Coordinates of all masked pixels:
        masked_pixels = numpy.transpose((mask_invert).nonzero())
        
        proj_size = math.prod(self.mask.shape)

        num_proj = int(data.size / proj_size) 

        # flat view of full array:
        out_flat = out.array.ravel()

        try:
            masked_pixels = [x * mask_arr.shape[1] + y for (x,y) in masked_pixels]
        except:
            masked_pixels = [x for (x,) in masked_pixels]

        # Dict of masked pixel coordinates and their unmasked neighbour coordinates and weights:
        masked_pixel_neighbours = _get_masked_pixel_neighbours(mask_arr)

        for k in range(num_proj):
            print("Processing projection %d of %d" %(k+1, num_proj))
            projection_out = out_flat[k*proj_size:(k+1)*proj_size]
        
            projection_temp = projection_out.copy()
            masked_pixels_in_proj = masked_pixels.copy()
            masked_pixels_temp = masked_pixels.copy()
            
            # Loop through masked pixel coordinates until all have been corrected:
            while (len(masked_pixels_in_proj)>0): 
                for coord in masked_pixels_in_proj:
                    neighbour_values = []
                    weights = []
                    for i, neighbour in enumerate(masked_pixel_neighbours[coord]['neighbours']):
                        if neighbour not in masked_pixels_in_proj:
                            neighbour_values.append(projection_out[neighbour])
                            weights.append(masked_pixel_neighbours[coord]['weights'][i])
                    if len(neighbour_values) > 0:
                        # Until we have looped through all masked pixels, we will only update the temporary projection
                        # and temporary copy of the list of masked pixels to be corrected:
                        projection_temp[coord] = numpy.average(neighbour_values, weights=weights)
                        masked_pixels_temp.remove(coord)

                # Update projection_out and remaining masked pixels to be corrected in projection
                # now that all masked pixels which have unmasked neighbours have been addressed in this iteration:
                projection_out = projection_temp.copy()
                masked_pixels_in_proj = masked_pixels_temp.copy()

            # Update the projection in out:
            out_flat[k*proj_size:(k+1)*proj_size] = projection_out

        if return_arr is True:
            return out
        
