# -*- coding: utf-8 -*-
#  Copyright 2025 United Kingdom Research and Innovation
#  Copyright 2025 The University of Manchester
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


class BadPixelCorrector(DataProcessor):   
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

    def __init__(self, mask, accelerated=False):
        kwargs = {'mask': mask, 'accelerated': accelerated}

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

    def _get_neighbours_and_weights(self, masked_pixels):
        """
        Get the neighbours (including diagonal) and weights for each masked pixel in the mask array.
        
        Parameters
        ----------

        masked_pixels : list
            A list of the coordinates of the masked pixels in the mask array.

        Returns
        -------
        dict
            A dictionary with masked pixel coordinates as keys and a dictionary with 'neighbours' and 'weights' as values.
            'neighbours' and 'weights' are lists of the coordinates of the unmasked neighbours and their weights respectively.
        """
        diag_weight = 1/np.sqrt(2) # to be used in weighting mean of diagonal neighbours
        masked_pixel_neighbours = {}
        proj_shape = self._get_proj_shape(self.get_input())
        for coord in masked_pixels: 
            # Get all neighbours
            neighbours = []
            weights = []
            if len(proj_shape)== 1:
                if coord > 0:
                    neighbours.append(coord-1)
                    weights.append(1)
                if coord < proj_shape[0]-1:
                    neighbours.append(coord+1)
                    weights.append(1)
            else:
                if coord % proj_shape[1]>0:
                    # Left neighbour
                    neighbours.append(coord - 1)
                    weights.append(1)
                if coord > (proj_shape[1]-1):
                    # Upper neighbour
                    neighbours.append(coord - proj_shape[1])
                    weights.append(1)
                if coord < (proj_shape[0]*proj_shape[1] - proj_shape[1]):
                    # Lower neighbour
                    neighbours.append(coord + proj_shape[1])
                    weights.append(1)
                if (coord + 1) % proj_shape[1] >0:
                    # Right neighbour
                    neighbours.append(coord + 1)
                    weights.append(1)
                if coord < (proj_shape[0]*proj_shape[1] - proj_shape[1]) and coord % proj_shape[1]>0:
                    # Diagonal lower left neighbour
                    neighbours.append(coord + proj_shape[1] - 1)
                    weights.append(diag_weight)
                if coord > (proj_shape[1]-1) and (coord + 1) % proj_shape[1] >0:
                    # Diagonal Upper right neighbour
                    neighbours.append(coord - proj_shape[1]+1)
                    weights.append(diag_weight)
                if coord > proj_shape[1] and coord % proj_shape[1]>0:
                    #Diagonal upper left neighbour
                    neighbours.append(coord - proj_shape[1] -1)
                    weights.append(diag_weight)
                if coord < (proj_shape[0]*proj_shape[1] - proj_shape[1]) and (coord + 1) % proj_shape[1] >0:
                    # Diagonal lower right neighbour
                    neighbours.append(coord + proj_shape[1] + 1)
                    weights.append(diag_weight)
                
            masked_pixel_neighbours[coord] = {'neighbours': neighbours, 'weights': weights}
        return masked_pixel_neighbours
        # alternative would be padding a border

    def _get_masked_pixels(self):
        try:
            mask_arr = self.mask.as_array()
        except:
            mask_arr = self.mask

        mask_arr = numpy.array(mask_arr, dtype=bool)

        # Coordinates of all masked pixels:
        masked_pixels = numpy.transpose(numpy.where(mask_arr == False))

        try:
            masked_pixels = [x * self._get_proj_shape(self.get_input())[1] + y for (x,y) in masked_pixels]
        except:
            masked_pixels = [x for (x,) in masked_pixels]

        return masked_pixels

    def process(self, out=None):
        data = self.get_input()
        return_arr = False
        if out is None:
            out = data.copy()
            return_arr = True

        if self.accelerated:
            pass
        else:
            self._process_numpy(data, out)

        if return_arr:
            return out

    def _process_numpy(self, data, out):
        try:
            mask_arr = self.mask.as_array()
        except:
            mask_arr = self.mask
        kernels, x, y = self._get_details_of_passes(mask_arr, out)

        print("Number of passes: ", len(kernels))

        self._bad_pix_correct_pixelwise(out, x, y, kernels)


    def _get_details_of_passes(self, mask, proj):
        print(mask)

        from scipy.ndimage import binary_erosion, binary_dilation

        #eroded_mask = ~binary_erosion(~mask)

        kernels = []
        coords_x = []
        coords_y = []
        proj_shape = np.shape(mask)
        num_proj = proj.shape[0]


        diag_weight = 1/np.sqrt(2)

        if len(proj_shape)== 1:
            pix_v = 1
            pix_h = proj_shape[0]
            weight_array = np.array([ 1,0,1])

        else:
            pix_v = proj_shape[0]
            pix_h = proj_shape[1]
            weight_array = np.array([diag_weight, 1, diag_weight, 1,0,1,diag_weight,1,diag_weight])


        prev_erosion = mask

        # proj_size = proj.size

        # eroded_mask = binary_dilation(prev_erosion)
        # print("Eroded mask: ",eroded_mask )
        # current_pixel_map = ~prev_erosion & eroded_mask
        current_pixel_map=True
        eroded_mask=False

        while not np.all(eroded_mask):
            #eroded_mask = ~binary_erosion(~prev_erosion)
            eroded_mask = binary_dilation(prev_erosion)
            #print("Eroded mask: ", eroded_mask )
            current_pixel_map = ~prev_erosion & eroded_mask

            #print("Current pixel map: ", current_pixel_map)
            #print("we'll check neighbours against: ", prev_erosion)

            

            # Coordinates of pixels to address in this iteration:
            #if we raveled:
            #masked_pixels = np.where(current_pixel_map.ravel() == True)[0]
            #         y = int(np.floor(coord / pix_h))
            #         x = int(coord  - y * pix_h)


            y_coords, x_coords = np.where(current_pixel_map)

            current_kernels = np.zeros(len(x_coords), dtype=np.uint8)

            
            for i, (x,y)  in enumerate(zip(x_coords,y_coords)):
                bit_mask = 255

                if x == 0:
                    bit_mask = bit_mask & 107
                if x == pix_h - 1:
                    bit_mask = bit_mask & 214

                if y == 0:
                    bit_mask = bit_mask & 31
                if y == pix_v -1:
                    bit_mask = bit_mask & 248


                # remove pixels if they're bad pixels that haven't been corrected in a previous iteration:


                # print("Previous erosion: ", prev_erosion)

                # could change this to not ravel:
                prev_erosion_flat = prev_erosion
                if (bit_mask & 1 << 7):
                    if not (prev_erosion_flat[y - 1][x - 1]): # [y-1][x-1]
                        bit_mask = bit_mask & ~(1 << 7)
                if (bit_mask & 1 << 6):
                    if not (prev_erosion_flat[y - 1][x]):
                        bit_mask = bit_mask & ~(1 << 6)
                if (bit_mask & 1 << 5):
                    if not (prev_erosion_flat[y - 1][x + 1]):
                        bit_mask = bit_mask & ~(1 << 5)
                if (bit_mask & 1 << 4):
                    if not (prev_erosion_flat[y][x - 1]):
                        bit_mask = bit_mask & ~(1 << 4)
                if (bit_mask & 1 << 3):
                    if not (prev_erosion_flat[y][x + 1]):
                        bit_mask = bit_mask & ~(1 << 3)
                if (bit_mask & 1 << 2):
                    if not (prev_erosion_flat[y + 1][x - 1]):
                        bit_mask = bit_mask & ~(1 << 2)
                if (bit_mask & 1 << 1):
                    if not (prev_erosion_flat[y + 1][x]):
                        bit_mask = bit_mask & ~(1 << 1)
                if (bit_mask & 1):
                    if not (prev_erosion_flat[y + 1][x + 1]):
                        bit_mask = bit_mask & ~(1)


                current_kernels[i] = bit_mask
            coords_x.append(x_coords)
            coords_y.append(y_coords)
            kernels.append(current_kernels) # replace the appends

            # now make the new current pixel map
            prev_erosion = eroded_mask

        #     print("the eroded mask: ", eroded_mask)
            

        # print("the kernels", kernels)

        # print("k: ", [bin(x) for x in kernels[0]])

        return kernels, coords_x, coords_y


    def _bad_pix_correct_pixelwise(self, proj, coords_x, coords_y, kernels):
        diag_weight = 1/np.sqrt(2)
        num_proj = len(proj.geometry.angles)
        num_channels = proj.geometry.channels
        print(proj)
        print(proj.dimension_labels)
        # For each channel:
        for l in range(0, num_channels):
            if num_channels == 1:
                current_channel = proj
            else:
                current_channel = proj.get_slice(channel=l)
        # For each projection:
            for k in range(0, num_proj):
                if num_proj == 1:
                    current_proj = current_channel
                else:
                    current_proj = proj.get_slice(angle=k)
                current_proj = current_proj.array
                # For each pass:
                for j in range(0, len(kernels)):
                    # For each bad pixel:
                    for i in range(0, len(kernels[j])):
                        kernel = kernels[j][i]
                        x = coords_x[j][i]
                        y = coords_y[j][i]

                        accum_num = 0
                        accum_denom = 0

                        if (kernel & 1 << 7):
                            accum_num += current_proj[y - 1][x - 1]
                            accum_denom +=1

                        if (kernel & 1 << 5):
                            accum_num += current_proj[y - 1][x + 1]
                            accum_denom +=1

                        if (kernel & 1 << 2):
                            accum_num += current_proj[y + 1][x - 1]
                            accum_denom +=1

                        if (kernel & 1 ):
                            accum_num += current_proj[y + 1][x + 1]
                            accum_denom +=1

                        accum_denom*=diag_weight
                        accum_num*=diag_weight


                        if (kernel & 1 << 6):
                            accum_num += current_proj[y - 1][x]
                            accum_denom +=1

                        if (kernel & 1 << 4):
                            accum_num += current_proj[y][x - 1]
                            accum_denom +=1

                        if (kernel & 1 << 3):
                            accum_num += current_proj[y][x + 1]
                            accum_denom +=1

                        if (kernel & 1 << 1):
                            accum_num += current_proj[y + 1][x]
                            accum_denom +=1

                        mean = accum_num/accum_denom

                        if num_channels>1 and num_proj>1:
                            proj.array[l][k][y][x] = mean
                        elif num_proj==1 and num_channels==1:
                            proj.array[y][x] = mean
                        elif num_proj==1:
                            proj.array[l][y][x] = mean
                        else:
                            proj.array[k][y][x] = mean


    def _process_numpy_old(self, data, out):

        # flat view of full array:
        out_flat = out.array.ravel()

        proj_size = math.prod(self._get_proj_shape(data))

        num_proj = int(data.size / proj_size) 

        masked_pixels = self._get_masked_pixels()

        # Dict of masked pixel coordinates and their unmasked neighbour coordinates and weights:
        masked_pixel_neighbours = self._get_neighbours_and_weights(masked_pixels)
        # get rid and pad instead and then we know location of neighbours
        # could potentially store first iteration or more's map


        for k in range(num_proj):
            print("Processing projection %d of %d" %(k+1, num_proj))
            projection_out = out_flat[k*proj_size:(k+1)*proj_size]
        
            projection_temp = projection_out.copy() #pad with zeros
            masked_pixels_in_proj = masked_pixels.copy()
            masked_pixels_temp = masked_pixels.copy()
            
            # Loop through masked pixel coordinates until all have been corrected:
            while (len(masked_pixels_in_proj)>0): 
                for coord in masked_pixels_in_proj:
                    neighbour_values = []
                    weights = []
                    for i, neighbour in enumerate(masked_pixel_neighbours[coord]['neighbours']): # could instead for each iteration run through all projs?
                        if neighbour not in masked_pixels_in_proj:
                            neighbour_values.append(projection_out[neighbour])
                            weights.append(masked_pixel_neighbours[coord]['weights'][i])
                            # accumulate the mean as identify pixels? bitwise and?
                    if len(neighbour_values) > 0:
                        # Until we have looped through all masked pixels, we will only update the temporary projection
                        # and temporary copy of the list of masked pixels to be corrected:
                        projection_temp[coord] = numpy.average(neighbour_values, weights=weights) # check if where arg exists
                        masked_pixels_temp.remove(coord)

                # Update projection_out and remaining masked pixels to be corrected in projection
                # now that all masked pixels which have unmasked neighbours have been addressed in this iteration:
                projection_out = projection_temp.copy()
                masked_pixels_in_proj = masked_pixels_temp.copy()

            # Update the projection in out: CHECK IF REDUNDANT:
            out_flat[k*proj_size:(k+1)*proj_size] = projection_out

        
