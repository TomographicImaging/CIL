#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
function to read IP data and provide a dictionary with data and parameters as an output
"""
from scipy import io
import numpy as np
from collections import defaultdict

def read_IPdata():
    # read data from mat file (specify the location)
    alldata = io.loadmat('IP_data70channels.mat')
    data_raw = alldata.get('Data_raw') # here is raw projection data
    Phantom_ideal = alldata.get('Phantom_ideal') # here is 70 channels ideal phantom
    Photon_flux = alldata.get('Photon_flux') # photon flux for normalization
    del alldata

    # extract geometry-related parameters
    proj_numb,detectors_numb,channels = np.shape(data_raw)
    im_size  = np.size(Phantom_ideal,1)

    theta = np.linspace(0,proj_numb-1,proj_numb)*360/proj_numb   # projection angles
    dom_width   = 1.0       # width of domain in cm
    src_to_rotc = 3.0       # dist. from source to rotation center
    src_to_det  = 5.0       # dist. from source to detector
    det_width   = 2.0       # detector width

    # negative log normalisation of the raw data (avoiding of log(0))
    data_norm = np.zeros(np.shape(data_raw))
    for i in range(0,channels):
        slice1 = data_raw[:,:,i]
        indx = np.nonzero(slice1>0)
        slice2 = np.zeros((proj_numb,detectors_numb), 'float32')
        slice2[indx] = -np.log(slice1[indx]/Photon_flux[i])
        indx2 = np.nonzero(slice1==0)
        slice3 = np.zeros((proj_numb,detectors_numb), 'float32')
        slice3[indx2] = np.log(slice2[indx2]+Photon_flux[i])
        data_norm[:,:,i] = slice2 + slice3
        del indx, indx2, slice1, slice2, slice3
    data_norm = np.float32(data_norm*(im_size/dom_width))
    
    #build a dictionary for data and related parameters
    dataDICT = defaultdict(list)
    dataDICT['data_norm'].append(data_norm)
    dataDICT['data_raw'].append(data_raw)
    dataDICT['Photon_flux'].append(Photon_flux)
    dataDICT['Phantom_ideal'].append(Phantom_ideal)
    dataDICT['theta'].append(theta)
    dataDICT['proj_numb'].append(proj_numb)
    dataDICT['detectors_numb'].append(detectors_numb)
    dataDICT['channels'].append(channels)
    dataDICT['im_size'].append(im_size)
    dataDICT['dom_width'].append(dom_width)
    dataDICT['src_to_rotc'].append(src_to_rotc)
    dataDICT['src_to_det'].append(src_to_det)
    dataDICT['det_width'].append(det_width)
    
    return (dataDICT)