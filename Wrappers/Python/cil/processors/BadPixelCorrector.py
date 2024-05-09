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
import warnings
import numpy
from scipy import interpolate
from cil.utilities.display import show2D
import numpy as np

class BadPixelCorrector(DataProcessor):
    r'''
    
    '''

    def __init__(self, mask):
        
        r'''Processor to correct bad pixels in an image, by replacing with the mean value of unmasked nearest neighbours.

        Parameters
        ----------
        mask : DataContainer, numpy.ndarray
            A boolean array with the same dimensions as the size of a projection in the input data, where 'False' represents masked values.
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

        try:
            proj_shape = data.get_slice(channel=0 if 'channel' in data.dimension_labels else None,
                                angle=0 if 'angle' in data.dimension_labels else None).shape
        except:
            proj_shape = data.shape
   
        if proj_shape != self.mask.shape:
            raise ValueError(f"Projection and Mask shapes do not match: {proj_shape} != {self.mask.shape}")
        
        return True

    def process(self, out=None):

        # Main question: can I allow only AcquisitionData?
        # Can I allow only 1D or 2D mask - don't think we would have 3D?

        # Cases:
        # 1D data, 1D mask -> apply mean in all (i.e. 1) direction
        # 2D data: i.e. 1D plus multiple angles, 1D mask -> apply mean in just 1 (not angles) direction
        # 2D data: i.e. single projection, with horizontal and vertical dimensions, 2D mask -> apply mean in both directions
        # 3D data: i.e. multiple projections, with horizontal and vertical dimensions, 2D mask -> apply mean in 2 directions

        # But we won't be passing in the mask, we will be passing in coordinates.
        # So we need to know which coordinates correspond to which dimensions.
        # We can do this by looking at the dimension labels of the data.
        # We can then use this to work out which dimensions to apply the mean in.


        # Could there be a case where the coords dict and data dimension labels are mismatched? e.g. run mask generator on a different dataset to the one we are correcting?
        # How do we avoid this?
        # We could save the info in the dict somehow

        # So if we have the coords and the dimension labels
        # We can work out which dimensions to apply the mean in
        # We can then apply the mean in those dimensions
        # Look for horizontal and vertical dimensions
        # Do we need to accommodate any others? Expect acquisitiondata.

        # Maybe we only ever take 2D mask which has horiz and vert?
        
        data = self.get_input()
        
        return_arr = False
        if out is None:
            out = data.copy()
            return_arr = True

        #assumes mask has 'as_array' method, i.e. is a DataContainer or is a numpy array
        try:
            mask_arr = self.mask.as_array()
        except:
            mask_arr = self.mask

        mask_arr = numpy.array(mask_arr, dtype=bool)
        mask_invert = ~mask_arr

        # Replace masked pixels with mean of unmasked neighbours, starting
        # with masked pixels with unmasked neighbours and iterating.

        # Coordinates of all masked pixels:
        masked_pixels = numpy.transpose((mask_invert).nonzero())


        # loop over angles:

        try:
            angles = data.geometry.angles
        except:
            angles = [0]

        channels = data.geometry.channels


        
        # need to check order of input array

        for j in range(channels):
            try: 
                projections = data.get_slice(channel=j)
            except:
                projections = data
            
            channel_out = projections.copy()
            print("First channel: ", channel_out)
            for i in range(len(angles)):
                # If only have one angle, can't use get_slice
                try:
                    projection = projections.get_slice(angle=i)
                except:
                    projection = projections

                projection_out = projection.copy()

                # Stores coordinates of all bad pixels and whether they have been corrected:
                masked_pixels_status = {}
                for coords in masked_pixels:
                    masked_pixels_status[tuple(coords)] = False

                # Loop through masked pixel coordinates until all have been corrected:
                while not all (masked_pixels_status.values()): # later have len >0
                    for coords, corrected in masked_pixels_status.items():
                        if not corrected: 
                            # Get all neighbours
                            neighbours = []
                            diag_neighbours = []
                            if len(list(coords))== 1:
                                if list(coords)[0] > 0:
                                    neighbours.append([coords[0]-1])
                                if list(coords)[0] < mask_arr.shape[0]-1:
                                    neighbours.append([coords[0]+1])
                            else:
                                if coords[0]>0 and coords[1]>0:
                                    diag_neighbours.append(tuple([coords[0]-1, coords[1]-1]))
                                if coords[0]>0:
                                    neighbours.append(tuple([coords[0]-1, coords[1]]))
                                if coords[1]>0:
                                    neighbours.append(tuple([coords[0], coords[1]-1]))
                                if coords[0]<mask_arr.shape[0]-1 and coords[1]<mask_arr.shape[1]-1:
                                    diag_neighbours.append(tuple([coords[0]+1, coords[1]+1]))
                                if coords[0]<mask_arr.shape[0]-1:
                                    neighbours.append(tuple([coords[0]+1, coords[1]]))
                                if coords[1]<mask_arr.shape[1]-1:
                                    neighbours.append(tuple([coords[0], coords[1]+1]))
                                if coords[0]<mask_arr.shape[0]-1 and coords[1]>0:
                                    diag_neighbours.append(tuple([coords[0]+1, coords[1]-1]))
                                if coords[0]>0 and coords[1]<mask_arr.shape[1]-1:
                                    diag_neighbours.append(tuple([coords[0]-1, coords[1]+1]))

                            # Save coords of unmasked neighbours, should include neighbours and diag_neighbours:
                            neighbour_values = []
                            weights = []
                            for neighbour in neighbours:
                                if neighbour in masked_pixels_status.keys() and not masked_pixels_status.get(neighbour):
                                    continue
                                neighbour_values.append(projection_out.as_array()[neighbour])
                                weights.append(1)
                            
                            for neighbour in diag_neighbours:
                                if neighbour in masked_pixels_status.keys() and not masked_pixels_status.get(neighbour):
                                    continue
                                neighbour_values.append(projection_out.as_array()[neighbour])
                                weights.append(1/np.sqrt(2))

                            print("neighbs: ", neighbour_values)
                            print("weights: ", weights)

                            if len(neighbour_values) > 0:
                                projection_out.array[coords] = numpy.average(neighbour_values, weights=weights)
                                masked_pixels_status[coords] = True # later remove from masked pixels array
                            print(projection_out.array)
                    # If data is a single projection, this will be the entire output:
                    try:
                        channel_out[i] = projection_out.array
                    except:
                        channel_out = projection_out.array

                    print("Channel out: ", channel_out)
                
                print("Out: ", out)
                try:
                    out.array[j] = channel_out
                except:
                    out.array = channel_out
                print("Out: ", out)

        if return_arr is True:
            return out
        



#%%

a = np.array([[6.0,4.0,6.0], [4,0,4], [6,4,6]])
mask = np.array([[True, True, True], [True, False, True], [True, True, True]])

ag = AcquisitionGeometry.create_Cone3D(source_position=[0,0, -1000], detector_position=[0,0, 1000]).set_panel([3,3]).set_angles([0])
ad = AcquisitionData(array=a, geometry=ag)

# Create a BadPixelCorrector processor
bad_pixel_corrector = BadPixelCorrector(mask)
# Apply the processor to the data
corrected_data = bad_pixel_corrector(ad)
# Check the result
print("Result: ")
print(corrected_data.array)


#%%

a = np.array([[3.,3.0,3.0], [3,0,3], [3,3,3]])
mask = np.array([[True, True, True], [True, False, True], [True, True, True]])

ag = AcquisitionGeometry.create_Cone3D(source_position=[0,0, -1000], detector_position=[0,0, 1000]).set_panel([3,3]).set_angles([0])
ad = AcquisitionData(array=a, geometry=ag)

# Create a BadPixelCorrector processor
bad_pixel_corrector = BadPixelCorrector(mask)
# Apply the processor to the data
corrected_data = bad_pixel_corrector(ad)
# Check the result
print("Result: ")
print(corrected_data.array)

#%%

a = np.array([[0.,0.,3], [0,2,2], [1,1,1]])
mask = np.array([[False, False, True], [False, True, True], [True, True, True]])

ag = AcquisitionGeometry.create_Cone3D(source_position=[0,0, -1000], detector_position=[0,0, 1000]).set_panel([3,3]).set_angles([0])
ad = AcquisitionData(array=a, geometry=ag)

# Create a BadPixelCorrector processor
bad_pixel_corrector = BadPixelCorrector(mask)
# Apply the processor to the data
corrected_data = bad_pixel_corrector(ad)
# Check the result
print("Result: ")
print(corrected_data.array)

#%%

a = np.array([[3.,0.,3.0], [0.,0.,0], [1.,0.,1.]])
mask = np.array([[True, False, True], [False, False, False], [True, False, True]])

ag = AcquisitionGeometry.create_Cone3D(source_position=[0,0, -1000], detector_position=[0,0, 1000]).set_panel([3,3]).set_angles([0])
ad = AcquisitionData(array=a, geometry=ag)

# Create a BadPixelCorrector processor
bad_pixel_corrector = BadPixelCorrector(mask)
# Apply the processor to the data
corrected_data = bad_pixel_corrector(ad)
# Check the result
print("Result: ")
print(corrected_data.array)

#%%
a = np.array([[1.,0.,1.0], [0.,0.,0], [3.,0.,3.]])
mask = np.array([[True, False, True], [False, False, False], [True, False, True]])

ag = AcquisitionGeometry.create_Cone3D(source_position=[0,0, -1000], detector_position=[0,0, 1000]).set_panel([3,3]).set_angles([0])
ad = AcquisitionData(array=a, geometry=ag)

# Create a BadPixelCorrector processor
bad_pixel_corrector = BadPixelCorrector(mask)
# Apply the processor to the data
corrected_data = bad_pixel_corrector(ad)
# Check the result
print("Result: ")
print(corrected_data.array)

#%%
a = np.array([[1.,1.,1], [0,0,0], [3,3,3]])
expected_a = np.array([[1,1,1], [2,2,2], [3,3,3]])
print(a.shape)

ag = AcquisitionGeometry.create_Cone3D(source_position=[0, 0, -1000], detector_position=[0, 0, 1000]).set_panel([3,3]).set_angles([0])
ad = AcquisitionData(array=a, geometry=ag) 

mask = np.array([[True, True, True], [False, False, False], [True, True, True]])

print("Data: ")
print(ad.array)
print("Mask: ")
print(mask)

# Create a BadPixelCorrector processor
bad_pixel_corrector = BadPixelCorrector(mask)
# Apply the processor to the data
corrected_data = bad_pixel_corrector(ad)
# Check the result
print("Result: ")
print(corrected_data.array)

assert np.allclose(corrected_data.array, expected_a)

#%%
a = np.array([[0,0,1.,1.], [0.,0.,2.,2.], [4.,3.,0.,0.], [4.,3.,0.,0.]])
mask = np.array([[False,False,True,True], [False, False, True, True], [True, True, False, False], [True, True, False, False]])

ag = AcquisitionGeometry.create_Cone3D(source_position=[0,0, -1000], detector_position=[0,0, 1000]).set_panel([4,4]).set_angles([0])
ad = AcquisitionData(array=a, geometry=ag)

# Create a BadPixelCorrector processor
bad_pixel_corrector = BadPixelCorrector(mask)
# Apply the processor to the data
corrected_data = bad_pixel_corrector(ad)
# Check the result
print("Result: ")
print(corrected_data.array)

#%%

a_x = np.array([[1.,1.,1.], [0.,0.,0.], [3.,3.,3.]])
a_y = np.array([[2,2,2], [0,0,0], [3,3,3]])
a_z = np.array([[3,3,3], [0,0,0], [3,3,3]])
a = np.array([a_x, a_y, a_z])

e_a_x = np.array([[1,1,1], [2,2,2], [3,3,3]])
e_a_y = np.array([[2,2,2], [2.5,2.5,2.5], [3,3,3]])
e_a_z = np.array([[3,3,3], [3,3,3], [3,3,3]])
print(a.shape)
             
expected_a = np.array([e_a_x, e_a_y, e_a_z])
print(a.shape)

ag = AcquisitionGeometry.create_Cone3D(source_position=[0, 0, -1000], detector_position=[0, 0, 1000]).set_panel([3,3]).set_angles([0]).set_channels(3)
ad = AcquisitionData(array=a, geometry=ag) 

print(ag.dimension_labels)


print(ad)

mask = np.array([[True, True, True], [False, False, False], [True, True, True]])

print("Data: ")
print(ad.array)
print("Mask: ")
print(mask)

# Create a BadPixelCorrector processor
bad_pixel_corrector = BadPixelCorrector(mask)
# Apply the processor to the data
corrected_data = bad_pixel_corrector(ad)
# Check the result
print("Result: ")
print(corrected_data.array)

assert np.allclose(corrected_data.array, expected_a)


#%%
# if __name__ == '__main__':

print('BadPixelCorrector: main')
# Create a 2D image
ig = AcquisitionGeometry.create_Cone3D(source_position=[0, 0, -1000], detector_position=[0, 0, 1000]).set_panel([10,10]).set_angles([0,1])
data = ig.allocate()
print(data.shape)

# make array that's one for first row, two for second row:
# a = 
# data[0].fill(1)
# data[1].fill(2)

# print(data.array[:, 2,4])

mask_coords = [(2,4), (6,8), (0,0)]

#%%
print(data.array[:].shape)

for coords in mask_coords:
    data.array[0][coords] = np.inf
    data.array[1][coords] = np.inf

print(data.array)

# Create a mask
mask = ig.allocate()
mask.fill(True)
mask = mask.array[0]

for coords in mask_coords:
    mask[coords] = False

# convert to bool:
# mask = np.array(mask, dtype=bool)
print(mask.shape)

show2D(data.as_array(), title='Original data')
show2D(mask, title='Mask')


# Create a BadPixelCorrector processor
bad_pixel_corrector = BadPixelCorrector(mask=mask)
# Apply the processor to the data
corrected_data = bad_pixel_corrector(data)
# Check the result
#print(corrected_data.as_array())
print('BadPixelCorrector: main: done')
show2D(corrected_data.as_array(), title='Corrected data')
# %%
