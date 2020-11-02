# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 14:39:20 2019

@author: ofn77899
"""

import sys
import unittest
import numpy
from cil.framework import DataProcessor
from cil.framework import DataContainer
from cil.framework import ImageData
from cil.framework import AcquisitionData
from cil.framework import ImageGeometry
from cil.framework import AcquisitionGeometry
from timeit import default_timer as timer
from cil.io.reader import NexusReader
from cil.processors import CenterOfRotationFinder

import wget
import os
import math

try:
    from cil.plugins.ccpi_reconstruction.processors import AcquisitionDataPadder
    has_ccpi_rec = True
except ImportError as ie:
    has_ccpi_rec = False



class TestDataProcessor(unittest.TestCase):
    @unittest.skipUnless(has_ccpi_rec, "Please install ccpi-reconstruction to run test")
    def setUp(self):
        wget.download('https://github.com/DiamondLightSource/Savu/raw/master/test_data/data/24737_fd.nxs')
        self.filename = '24737_fd.nxs'
    @unittest.skipUnless(has_ccpi_rec, "Please install ccpi-reconstruction to run test")
    def tearDown(self):
        os.remove(self.filename)
    @unittest.skipUnless(has_ccpi_rec, "Please install ccpi-reconstruction to run test")
    def test_AcquisitionDataPadder(self):
        reader = NexusReader(self.filename)
        ad = reader.get_acquisition_data_whole()
        print (ad.geometry)
        cf = CenterOfRotationFinder()
        cf.set_input(ad)
        print ("Center of rotation", cf.get_output())
        self.assertAlmostEqual(86.25, cf.get_output())
        
        adp = AcquisitionDataPadder(acquisition_geometry=cf.get_input().geometry,center_of_rotation=cf.get_output(),pad_value=0)
        adp.set_input(ad)
        padded_data = adp.get_output()
        print ("Padded data shape", padded_data.shape)
        print ("      " , padded_data.dimension_labels)
        idx = None
        for k,v in padded_data.dimension_labels.items():
            if v == AcquisitionGeometry.HORIZONTAL:
                idx = k

        padded_axis = padded_data.shape[idx]
        self.assertEqual(padded_axis , math.ceil(cf.get_output() * 2))
        #numpy.save("pippo.npy" , padded_data.as_array())
    
