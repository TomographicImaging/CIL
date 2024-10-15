# -*- coding: utf-8 -*-
#  Copyright 2021 United Kingdom Research and Innovation
#  Copyright 2021 The University of Manchester
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

#%%

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

        Parameters
        ----------
        mask : DataContainer, numpy.ndarray
            A boolean array with the same dimensions as the size of a projection in the input data (i.e. 1D or 2D), 
            where 'False' represents masked values.
            Mask can be generated using 'MaskGenerator' processor.
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
            # do not allow anything but horizontal and vertical:
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
        channel = True if 'channel' in data.dimension_labels else None
        angle = True if 'angle' in data.dimension_labels else None

        try:
            proj_shape = data.get_slice(channel=channel, angle=angle).shape
        except Exception:
            # if we have only one angle and one channel, we can't use get_slice  
            proj_shape = data.shape

        return proj_shape
        

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
        
        proj_size = math.prod(self._get_proj_shape(data))

        num_proj = int(data.size / proj_size) # product of remaining dimensions

        # flat view of full array
        out_flat = out.array.ravel()

        try:
            masked_pixels = [x * mask_arr.shape[1] + y for (x,y) in masked_pixels]
        except:
            masked_pixels = [x for (x,) in masked_pixels]
        
        #%%
        for k in range(num_proj):
            print("Processing projection %d of %d" %(k+1, num_proj))
            # this will create a copy the data
            projection_out = out_flat[k*proj_size:(k+1)*proj_size]
        
            projection_temp = projection_out.copy()
                
            # Stores coordinates of all bad pixels and whether they have been corrected:
            masked_pixels_status = {}
            
            for coord in masked_pixels:
                masked_pixels_status[coord] = False

            masked_pixels_status_temp = masked_pixels_status.copy()

            # Loop through masked pixel coordinates until all have been corrected:
            while not all (masked_pixels_status.values()): # later have len >0
                for coord, corrected in masked_pixels_status.items():
                    if not corrected:
                        # Get all neighbours
                        neighbours = []
                        diag_neighbours = []
                        weights = []
                        if len(mask_arr.shape)== 1:
                            if coord > 0:
                                neighbours.append(coord-1)
                            if coord < mask_arr.shape[0]-1:
                                neighbours.append(coord+1)
                        else:
        
                            if coord % mask_arr.shape[1]>0:
                                # Left neighbour
                                neighbours.append(coord - 1)
                            if coord > (mask_arr.shape[1]-1):
                                # Upper neighbour
                                neighbours.append(coord - mask_arr.shape[1])
                            if coord < (mask_arr.shape[0]*mask_arr.shape[1] - mask_arr.shape[1]):
                                # Lower neighbour
                                neighbours.append(coord + mask_arr.shape[1])
                            if (coord + 1) % mask_arr.shape[1] >0:
                                # Right neighbour
                                neighbours.append(coord + 1)
                            if coord < (mask_arr.shape[0]*mask_arr.shape[1] - mask_arr.shape[1]) and coord % mask_arr.shape[1]>0:
                                # Diagonal lower left neighbour
                                diag_neighbours.append(coord + mask_arr.shape[1] - 1)
                            if coord > (mask_arr.shape[1]-1) and (coord + 1) % mask_arr.shape[1] >0:
                                # Diagonal Upper right neighbour
                                diag_neighbours.append(coord - mask_arr.shape[1]+1)
                            if coord > mask_arr.shape[1] and coord % mask_arr.shape[1]>0:
                                #Diagonal upper left neighbour
                                diag_neighbours.append(coord - mask_arr.shape[1] -1)
                            if coord < (mask_arr.shape[0]*mask_arr.shape[1] - mask_arr.shape[1]) and (coord + 1) % mask_arr.shape[1] >0:
                                # Diagonal lower right neighbour
                                diag_neighbours.append(coord + mask_arr.shape[1] + 1)

                        # Save coord of unmasked neighbours, should include neighbours and diag_neighbours:
                        neighbour_values = []
                        weights = []
                        projection_array = projection_out
                        diag_weight = 1/np.sqrt(2)
                        for neighbour in neighbours + diag_neighbours:
                            if neighbour in masked_pixels_status.keys() and not masked_pixels_status.get(neighbour):
                                continue
                            neighbour_values.append(projection_array[neighbour])
                            weights.append(1 if neighbour in neighbours else diag_weight)

                        if len(neighbour_values) > 0:
                            projection_temp[coord] = numpy.average(neighbour_values, weights=weights)
                            masked_pixels_status_temp[coord] = True

                # Update projection_out now that all masked pixels have been addressed in this iteration:
                projection_out = projection_temp.copy()
                masked_pixels_status = masked_pixels_status_temp.copy()

            # Update the projection:
            # #fill the data back
            out_flat[k*proj_size:(k+1)*proj_size] = projection_out


        if return_arr is True:
            return out

        

# # THE FOLLOWING ARE FOR QUICK TESTING ONLY - WILL LATER BE MOVED TO UNIT TESTS
# #%%
# # 1D data

# print( "EXAMPLES -----------------------------------")

# print("1D Example")
        
# start_time = time.time()

# a = np.array([6.0,0.0,4.0])
# mask = np.array([True, False, True])

# ag = AcquisitionGeometry.create_Cone2D(source_position=[0, -1000], detector_position=[0, 1000]).set_panel(3).set_angles([0])
# ad = AcquisitionData(array=a, geometry=ag)

# print("Input:")
# print(ad.as_array())

# # Create a BadPixelCorrector processor
# bad_pixel_corrector = BadPixelCorrector(mask)
# # Apply the processor to the data
# corrected_data = bad_pixel_corrector(ad)

# end_time = time.time()
# # Check the result

# # Calculate the time taken
# time_taken = end_time - start_time

# # Print the time taken
# print(f"Time taken: {time_taken} seconds")


# #%%
# print("2D Example")
# a = np.array([[6.0,4.0,6.0], [4,0,4], [6,4,6]])
# mask = np.array([[True, True, True], [True, False, True], [True, True, True]])

# ag = AcquisitionGeometry.create_Cone3D(source_position=[0,0, -1000], detector_position=[0,0, 1000]).set_panel([3,3]).set_angles([0])
# ad = AcquisitionData(array=a, geometry=ag)

# print("Input:")
# print(ad.as_array())

# # Create a BadPixelCorrector processor
# bad_pixel_corrector = BadPixelCorrector(mask)
# # Apply the processor to the data
# corrected_data = bad_pixel_corrector(ad)
# # Check the result

# # print("Result: ")
# # print(corrected_data.array)


# #%%
# print("Note the ordering of how we loop through the pixels does not affect the results.")
# print("In the following 2 examples we have the same input data but flipped:")
# a = np.array([[3.,0.,3.0], [0.,0.,0], [1.,0.,1.]])
# mask = np.array([[True, False, True], [False, False, False], [True, False, True]])

# ag = AcquisitionGeometry.create_Cone3D(source_position=[0,0, -1000], detector_position=[0,0, 1000]).set_panel([3,3]).set_angles([0])
# ad = AcquisitionData(array=a, geometry=ag)

# print("Input:")
# print(ad.as_array())

# # Create a BadPixelCorrector processor
# bad_pixel_corrector = BadPixelCorrector(mask)
# # Apply the processor to the data
# corrected_data = bad_pixel_corrector(ad)


# #%%
# a = np.array([[1.,0.,1.0], [0.,0.,0], [3.,0.,3.]])
# mask = np.array([[True, False, True], [False, False, False], [True, False, True]])

# ag = AcquisitionGeometry.create_Cone3D(source_position=[0,0, -1000], detector_position=[0,0, 1000]).set_panel([3,3]).set_angles([0])
# ad = AcquisitionData(array=a, geometry=ag)

# # Check the result
# print("Input:")
# print(ad.as_array())

# # Create a BadPixelCorrector processor
# bad_pixel_corrector = BadPixelCorrector(mask)
# # Apply the processor to the data
# corrected_data = bad_pixel_corrector(ad)



# #%%
# print("This example shows that the method still corrects all of the pixels even if some of the starting masked pixels begin with no unmasked neighbours:")
# print("Input:")
# print(ad.as_array())
# a = np.array([[0,0,1.,1.], [0.,0.,2.,2.], [4.,3.,0.,0.], [4.,3.,0.,0.]])
# mask = np.array([[False,False,True,True], [False, False, True, True], [True, True, False, False], [True, True, False, False]])

# ag = AcquisitionGeometry.create_Cone3D(source_position=[0,0, -1000], detector_position=[0,0, 1000]).set_panel([4,4]).set_angles([0])
# ad = AcquisitionData(array=a, geometry=ag)

# # Create a BadPixelCorrector processor
# bad_pixel_corrector = BadPixelCorrector(mask)
# # Apply the processor to the data
# corrected_data = bad_pixel_corrector(ad)








# #%% 1D data with channels

# a = np.array([[6.0,0.0,4.0], [5.0,0,3]])
# mask = np.array([True, False, True])

# ag = AcquisitionGeometry.create_Cone2D(source_position=[0, -1000], detector_position=[0, 1000]).set_panel(3).set_angles([0]).set_channels(2)
# ad = AcquisitionData(array=a, geometry=ag)

# print("Input:")
# print(ad.as_array())

# # Create a BadPixelCorrector processor
# bad_pixel_corrector = BadPixelCorrector(mask)
# # Apply the processor to the data
# corrected_data = bad_pixel_corrector(ad)
# # Check the result
# print("Input:")
# print(ad.as_array())
# # print("Result: ")
# # print(corrected_data.array)



# #%%

# a = np.array([[3.,3.0,3.0], [3,0,3], [3,3,3]])
# mask = np.array([[True, True, True], [True, False, True], [True, True, True]])

# ag = AcquisitionGeometry.create_Cone3D(source_position=[0,0, -1000], detector_position=[0,0, 1000]).set_panel([3,3]).set_angles([0])
# ad = AcquisitionData(array=a, geometry=ag)

# # Create a BadPixelCorrector processor
# bad_pixel_corrector = BadPixelCorrector(mask)
# # Apply the processor to the data
# corrected_data = bad_pixel_corrector(ad)
# # Check the result
# print("Result: ")
# print(corrected_data.array)

# #%%

# a = np.array([[0.,0.,3], [0,2,2], [1,1,1]])
# mask = np.array([[False, False, True], [False, True, True], [True, True, True]])

# ag = AcquisitionGeometry.create_Cone3D(source_position=[0,0, -1000], detector_position=[0,0, 1000]).set_panel([3,3]).set_angles([0])
# ad = AcquisitionData(array=a, geometry=ag)

# # Create a BadPixelCorrector processor
# bad_pixel_corrector = BadPixelCorrector(mask)
# # Apply the processor to the data
# corrected_data = bad_pixel_corrector(ad)
# # Check the result
# print("Result: ")
# print(corrected_data.array)


# #%%
# a = np.array([[1.,1.,1], [0,0,0], [3,3,3]])
# expected_a = np.array([[1,1,1], [2,2,2], [3,3,3]])
# print(a.shape)

# ag = AcquisitionGeometry.create_Cone3D(source_position=[0, 0, -1000], detector_position=[0, 0, 1000]).set_panel([3,3]).set_angles([0])
# ad = AcquisitionData(array=a, geometry=ag) 

# mask = np.array([[True, True, True], [False, False, False], [True, True, True]])

# print("Data: ")
# print(ad.array)
# print("Mask: ")
# print(mask)

# # Create a BadPixelCorrector processor
# bad_pixel_corrector = BadPixelCorrector(mask)
# # Apply the processor to the data
# corrected_data = bad_pixel_corrector(ad)
# # Check the result
# print("Result: ")
# print(corrected_data.array)

# assert np.allclose(corrected_data.array, expected_a)



# #%%

# a_x = np.array([[1.,1.,1.], [0.,0.,0.], [3.,3.,3.]])
# a_y = np.array([[2,2,2], [0,0,0], [3,3,3]])
# a_z = np.array([[3,3,3], [0,0,0], [3,3,3]])
# a = np.array([a_x, a_y, a_z])

# e_a_x = np.array([[1,1,1], [2,2,2], [3,3,3]])
# e_a_y = np.array([[2,2,2], [2.5,2.5,2.5], [3,3,3]])
# e_a_z = np.array([[3,3,3], [3,3,3], [3,3,3]])
# print(a.shape)
             
# expected_a = np.array([e_a_x, e_a_y, e_a_z])
# print(a.shape)

# ag = AcquisitionGeometry.create_Cone3D(source_position=[0, 0, -1000], detector_position=[0, 0, 1000]).set_panel([3,3]).set_angles([0]).set_channels(3)
# ad = AcquisitionData(array=a, geometry=ag) 

# print(ag.dimension_labels)


# print(ad)

# mask = np.array([[True, True, True], [False, False, False], [True, True, True]])

# print("Data: ")
# print(ad.array)
# print("Mask: ")
# print(mask)

# # Create a BadPixelCorrector processor
# bad_pixel_corrector = BadPixelCorrector(mask)
# # Apply the processor to the data
# corrected_data = bad_pixel_corrector(ad)
# # Check the result
# print("Result: ")
# print(corrected_data.array)

# assert np.allclose(corrected_data.array, expected_a)


# #%%
# # if __name__ == '__main__':

# print('BadPixelCorrector: main')
# # Create a 2D image
# ig = AcquisitionGeometry.create_Cone3D(source_position=[0, 0, -1000], detector_position=[0, 0, 1000]).set_panel([10,10]).set_angles([0,1])
# data = ig.allocate()
# print(data.shape)

# # make array that's one for first row, two for second row:
# # a = 
# # data[0].fill(1)
# # data[1].fill(2)

# # print(data.array[:, 2,4])

# mask_coord = [(2,4), (6,8), (0,0)]

# #%%
# print(data.array[:].shape)

# for coord in mask_coord:
#     data.array[0][coord] = np.inf
#     data.array[1][coord] = np.inf

# print(data.array)

# # Create a mask
# mask = ig.allocate()
# mask.fill(True)
# mask = mask.array[0]

# for coord in mask_coord:
#     mask[coord] = False

# # convert to bool:
# # mask = np.array(mask, dtype=bool)
# print(mask.shape)

# show2D(data.as_array(), title='Original data')
# show2D(mask, title='Mask')


# # Create a BadPixelCorrector processor
# bad_pixel_corrector = BadPixelCorrector(mask=mask)
# # Apply the processor to the data
# corrected_data = bad_pixel_corrector(data)
# # Check the result
# #print(corrected_data.as_array())
# print('BadPixelCorrector: main: done')
# show2D(corrected_data.as_array(), title='Corrected data')
# # %%
