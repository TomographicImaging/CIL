# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 16:00:53 2018

@author: ofn77899
"""

import os
from ccpi.io.reader import NexusReader

from sys import getsizeof

import matplotlib.pyplot as plt


directory = r'E:\Documents\Dataset\CCPi\Nexus_test'
data_path="entry1/instrument/pco1_hw_hdf_nochunking/data"

reader = NexusReader(os.path.join( os.path.abspath(directory) , '74331.nxs'))

array = reader.get_acquisition_data_slice(200).array
#%%
fig = plt.subplot(3,1,1)


fig.imshow(array)
array = reader.get_acquisition_data_slice(300).array
#%%
fig = plt.subplot(3,1,2)


fig.imshow(array)


array = reader.get_acquisition_data_slice(500).array

#%%
fig = plt.subplot(3,1,3)


fig.imshow(array)
plt.show()





