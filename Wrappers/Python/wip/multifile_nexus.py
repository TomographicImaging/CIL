# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 16:00:53 2018

@author: ofn77899
"""

import os
from ccpi.io.reader import NexusReader

import h5py
from h5py import File as NexusFile
from sys import getsizeof
import matplotlib.pyplot as plt


def print_el(name, node):
    el = nf.get(name, getlink=True)
    print (el , node)
    #if type(el) == h5py.HardLink:
    #    print (name, "ExternalLink" , el)
    #else:
    #    print (type(node))
        

import numpy as np
directory = r'E:\Documents\Dataset\CCPi\Nexus_test'
data_path="entry1/instrument/pco1_hw_hdf_nochunking/data"

#hf = NexusFile(os.path.join( os.path.abspath(directory) , 'pco1-74240.hdf'), 'r')

#edo = NexusFile(os.path.join( os.path.abspath(directory) , 'edo.nxs'), 'r')
#edo.visititems(print_el)
#edo.close()

#nf = NexusFile(os.path.join( os.path.abspath(directory) , '74240.nxs'), 'r')

#nf.visititems(print_el)
#data = nf[data_path]
#print (data.shape)
#nf.close()


#nf.create_group("ext_link")
#nf.create_dataset('ext_link')
#nf['edo'] = h5py.ExternalLink(
        #os.path.join( os.path.abspath(directory) ,
#              u"pco1-74240.hdf" 
        #             )
#        , 
#        u"/entry/instrument/detector/data")
#image_keys = np.array(nf['entry1/tomo_entry/instrument/detector/image_key'])


#print (image_keys)

reader = NexusReader(os.path.join( os.path.abspath(directory) , '74331.nxs'))
#reader.data_path = data_path
#print (reader.list_file_content())

#data_path="entry1/pco1_hw_hdf_nochunking/data"
#reader.data_path = data_path
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





#
#plt.imshow(a[:,100,:])
#
#                
#                
#arr = np.array(nf['entry1/tomo_entry/data/data'][:,100,:][image_keys==0])
#
#fig = plt.figure()
#
#plt.imshow(arr)
#
#fig = plt.figure()
#
#b = reader.get_acquisition_data_subset(80,81)#.subset(['angle','horizontal'])
#plt.imshow(b.array.squeeze())
#fig = plt.figure()
#
#b2 = reader.get_acquisition_data_subset(80,81).subset(['angle','horizontal'])
#plt.imshow(b2.array)
#
#fig = plt.figure()
#
#b3 = reader.get_acquisition_data_slice(80)
#plt.imshow(b3.array)