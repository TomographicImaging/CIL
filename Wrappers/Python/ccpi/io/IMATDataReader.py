#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 10:30:25 2019

@author: evelina
"""


from ccpi.framework import AcquisitionData, AcquisitionGeometry
import numpy
import matplotlib.pyplot as plt
import os
import csv


astropyAvailable = True
try:
    from astropy.io import fits
except:
    astropyAvailable = False


class IMATDataReader(object):
    
    def __init__(self, 
                 **kwargs):
        '''
        Constructor
        
        Input:
            Projections loader relies on following namimg convention:
            pathname_projection + projection_prefix + angle + / + IMAT + \
            projection_counter + projection_channel_prefix + channel_number.fits
            Example:
            /pathname_projection/Angle_0.0/IMAT00010699_Tomo_000_00008.fits
            where:
                projection_prefix = "Angle_"
                angle = 0.0 (taken from angles list)
                projection_counter = 10699
                projection_channel_prefix = "Tomo_000"
                channel_number = 8 (we loop through channels depending on user's input)
            
            Flats (before and after) loader relies on following namimg convention:
            flat_path + flat_prefix + flat_number + / + IMAT + \
            flat_counter + flat_channel_prefix + channel_number.fits
            Example:
            /flat_path/Flat1/IMAT00010887_ Tomo_000_00003.fits
            where:
                flat_prefix = "Flat"
                flat_number = 1 (we loop through flats depending on 
                                 num_flat_before/ num_flat_after, 
                                 relies on 1-based indexing)
                flat_counter = 10887
                projection_channel_prefix = " Tomo_000"
                channel number = 3 (we loop through channels depending on user's input)
            
            angles                  list of angles to load. 
            
            shutter_values_file     full path to a text file with shutter values
            
            pixel_num_h_0           number of pixels along X axis, default is 512
            
            pixel_size_h_0          pixel size along X axis, default is 0.055 mm
            
            pixel_num_v_0           number of pixels along Y axis, default is 512
            
            pixel_size_v_0          pixel size along Y axis, default is 0.055 mm
            
            fligt_path              flight path used to convert TOF to Angstroms,
                                    default 56.4 m
                                    
            roi                     region-of-interest to load. If roi = -1 (default), 
                                    full projections will be loaded. Otherwise roi is 
                                    given by [row0, column0, row1, column1], where 
                                    row0, column0 are coordinates of top left corner and 
                                    row1, column1 are coordinates of bottom right corner.
                            
            binning                 number of pixels to bin (combine) along 0 (column) 
                                    and 1 (row) dimension. If binning = [1, 1] (default),
                                    projections in original resolution are loaded. Note, 
                                    if binning[0] != binning[1], then loaded projections
                                    will have anisotropic pixels, which are currently not 
                                    supported by the Framework
            
            wavelength_range        wavelength range (in Angstroms) to load. 
                                    If wavelength_range = -1 (default), 
                                    all channels will be loaded. Otherwise, 
                                    wavelength_range should be given as [float0, float1],
                                    where float1 > float0, (Python-style inclusive-lower-bound, 
                                    exclusive-upper-bound).
                                    Note, if wavelength_range != -1, then intervals must be equal -1.
            
            intervals               shutter intervals to load. If intervals = -1 (default), 
                                    all channels will be loaded. Otherwise, intervals
                                    should be given as [int0, int1], where int1 > int0 
                                    (Python-style inclusive-lower-bound, exclusive-upper-bound).
                                    Note, if intervals != -1, then wavelength_range must be equal -1.
                                    
        '''

        self.projection_path = kwargs.get('pathname_projection', None)
        self.projection_prefix = kwargs.get('projection_prefix', None)
        self.projection_channel_prefix = kwargs.get('projection_channel_prefix', None)
        self.projection_counter = kwargs.get('projection_counter', None)
        self.angles = kwargs.get('angles', None)
        self.shutter_values_file = kwargs.get('shutter_values_file', None)
        self.pixel_num_h_0 = kwargs.get('pixel_num_h_0', 512) 
        self.pixel_size_h_0 = kwargs.get('pixel_size_h_0', 0.055)
        self.pixel_num_v_0 = kwargs.get('pixel_num_v_0', 512)
        self.pixel_size_v_0 = kwargs.get('pixel_size_v_0', 0.055)
        self.fligt_path = kwargs.get('fligt_path', 56.4)
        self.flat_after_path = kwargs.get('pathname_flat_after', None)
        self.flat_after_prefix = kwargs.get('flat_after_prefix', None)
        self.flat_after_channel_prefix = kwargs.get('flat_after_channel_prefix', None)
        self.flat_after_counter = kwargs.get('flat_after_counter', None)
        self.num_flat_after = kwargs.get('num_flat_after', 0)
        self.flat_before_path = kwargs.get('pathname_before_after', None)
        self.flat_before_prefix = kwargs.get('flat_before_prefix', None)
        self.flat_before_channel_prefix = kwargs.get('flat_before_channel_prefix', None)
        self.flat_before_counter = kwargs.get('flat_before_counter', None)
        self.num_flat_before = kwargs.get('num_before_after', 0)
        self.roi = kwargs.get('roi', -1)
        self.binning = kwargs.get('binning', [1, 1])
        self.wavelength_range = kwargs.get('wavelength_range', -1)
        self.intervals = kwargs.get('intervals', -1)
        
        if ((self.projection_path != None) and 
            (self.projection_prefix != None) and 
            (self.projection_channel_prefix != None) and 
            (self.projection_counter != None) and 
            (self.angles != None) and
            (self.shutter_values_file != None)):
                
            self.set_up(projection_path = self.projection_path, 
                        projection_prefix = self.projection_prefix, 
                        projection_channel_prefix = self.projection_channel_prefix,
                        projection_counter = self.projection_counter,
                        angles = self.angles,
                        shutter_values_file = self.shutter_values_file,
                        pixel_num_h_0 = self.pixel_num_h_0,
                        pixel_size_h_0 = self.pixel_size_h_0,
                        pixel_num_v_0 = self.pixel_num_v_0,
                        pixel_size_v_0 = self.pixel_size_v_0,
                        fligt_path = self.fligt_path,
                        flat_after_path = self.flat_after_path,
                        flat_after_prefix = self.flat_after_prefix,
                        flat_after_channel_prefix = self.flat_after_channel_prefix,
                        flat_after_counter = self.flat_after_counter,
                        num_flat_after = self.num_flat_after,
                        flat_before_path = self.flat_before_path,
                        flat_before_prefix = self.flat_before_prefix,
                        flat_before_channel_prefix = self.flat_before_channel_prefix,
                        flat_before_counter = self.flat_before_counter,
                        num_flat_before = self.num_flat_before,
                        roi = self.roi,
                        binning = self.binning,
                        wavelength_range = self.wavelength_range,
                        intervals = self.intervals)
            
    def set_up(self,
               projection_path, 
               projection_prefix, 
               projection_channel_prefix,
               projection_counter,
               angles,
               shutter_values_file,
               pixel_num_h_0 = 512,
               pixel_size_h_0 = 0.055,
               pixel_num_v_0 = 512,
               pixel_size_v_0 = 0.055,
               fligt_path = 56.4,
               flat_after_path = None,
               flat_after_prefix = None,
               flat_after_channel_prefix = None,
               flat_after_counter = None,
               num_flat_after = 0,
               flat_before_path = None,
               flat_before_prefix = None,
               flat_before_channel_prefix = None,
               flat_before_counter = None,
               num_flat_before = 0,
               roi = -1,
               binning = [1, 1],
               wavelength_range = -1,
               intervals = -1):
    
        self.projection_path = projection_path
        self.projection_prefix = projection_prefix
        self.projection_channel_prefix = projection_channel_prefix
        self.projection_counter = projection_counter
        self.angles = angles
        self.shutter_values_file = shutter_values_file
        self.pixel_num_h_0 = pixel_num_h_0
        self.pixel_size_h_0 = pixel_size_h_0
        self.pixel_num_v_0 = pixel_num_v_0
        self.pixel_size_v_0 = pixel_size_v_0
        self.fligt_path = fligt_path
        self.flat_after_path = flat_after_path
        self.flat_after_prefix = flat_after_prefix
        self.flat_after_channel_prefix = flat_after_channel_prefix
        self.flat_after_counter = flat_after_counter
        self.num_flat_after = num_flat_after
        self.flat_before_path = flat_before_path
        self.flat_before_prefix = flat_before_prefix
        self.flat_before_channel_prefix = flat_before_channel_prefix
        self.flat_before_counter = flat_before_counter
        self.num_flat_before = num_flat_before
        self.roi = roi
        self.binning = binning
        self.wavelength_range = wavelength_range
        self.intervals = intervals
        
        if ((self.projection_path == None) or 
            (self.projection_prefix == None) or 
            (self.projection_channel_prefix == None) or 
            (self.projection_counter == None) or
            (self.angles == None) or
            (self.shutter_values_file == None)):
            raise Exception('A minimal set of following parameters is required \
                            to set up the IMATDataReader: projection_path, \
                            projection_prefix, projection_channel_prefix, \
                            projection_counter, angles and shutter_values_file')
            
        # check ROI
        if (self.roi != -1): 
            if not ((isinstance(self.roi, list)) or 
                    (len(self.roi) == 4) or 
                    (self.roi[0] < self.roi[2]) or 
                    (self.roi[1] < self.roi[3])):
                raise Exception('Not valid ROI. \
                                ROI must be defined as [row0, column0, row1, column1] \
                                such that ((row0 < row1) and (column0 < column1))')
        
        # check binning parameters
        if not ((isinstance(self.binning, list)) or 
                (len(self.binning) == 2)):
            raise Exception('Not valid binning parameters. \
                            Binning must be defined as [int, int]')
            
        # check wavelength range
        if (self.wavelength_range != -1):
            if not ((len(self.wavelength_range) == 2) or 
                    (self.wavelength_range[1] > self.wavelength_range[0])):
                raise Exception('Not valid wavelength range. \
                                Wavelength range must be defined as [float0, float1] \
                                such that (float1 > float0)')
                
        # check wavelength range
        if (self.intervals != -1):
            if not ((isinstance(self.intervals, list)) or 
                    (len(self.intervals) == 2) or
                    (self.intervals[1] > self.intervals[0])):
                raise Exception('Not valid intervals range. \
                                Intervals must be defined as [int0, int1] \
                                such that (int1 > int0)')
        
        # check wavelength_range or intervals is given
        if ((self.wavelength_range != -1) and (self.intervals != -1)):
            raise Exception('Input conflict. Either wavelength_range or intervals must be given.')
        
        # check that astropy library is installed
        if (astropyAvailable == False):
            raise Exception("ASTROPY is not available, cannot load FITS files")
            
        # check if shutter values file exists
        if not(os.path.isfile(self.shutter_values_file)):
            raise Exception('File {} does not exist'.format(self.shutter_values_file)) 
            
        # parse file with shutter values
        with open(self.shutter_values_file, 'r') as shutter_values_csv:
            csv_reader = csv.reader(shutter_values_csv, delimiter = '\t')
            
            # number of shutter intervals
            self._n_intervals = sum([1 for row in csv_reader])
           
            shutter_values_csv.seek(0)
            
            # read limits of each shutter interval
            counter = 0
            # left limit
            tof_lim_1 = numpy.zeros(self._n_intervals, dtype = float)
            # right limit
            tof_lim_2 = numpy.zeros(self._n_intervals, dtype = float)
            # channel width
            tof_channel_width = numpy.zeros(self._n_intervals, dtype = float)
            
            for row in csv_reader:
                tof_lim_1[counter] = float(row[0])
                tof_lim_2[counter] = float(row[1])
                tof_channel_width[counter] = float(row[3])
                
                counter += 1
        
        # calculate number of channels in each shutter interval
        # TOF is in seconds, channels in microseconds
        n_channels_per_interval = numpy.int_(numpy.floor((tof_lim_2 - tof_lim_1) / (tof_channel_width * 1e-6)))
        n_channels_total = numpy.sum(n_channels_per_interval)
        
        # calculate edges of each energy channel in TOF
        # and interval id for every channel
        tof_channels = numpy.zeros((n_channels_total, 2), dtype = float)
        self._interval_id = numpy.zeros((n_channels_total), dtype = 'int_')
        counter = 0
        for i in range(self._n_intervals):
            tof_channels[counter:(counter + n_channels_per_interval[i]), 0] = \
                tof_lim_1[i] + tof_channel_width[i] * 1e-6 * numpy.arange(n_channels_per_interval[i])
            tof_channels[counter:(counter + n_channels_per_interval[i]), 1] = \
                tof_lim_1[i] + tof_channel_width[i] * 1e-6 * numpy.arange(1, (n_channels_per_interval[i] + 1))
            self._interval_id[counter:(counter + n_channels_per_interval[i])] = i
            counter += n_channels_per_interval[i]
    
        # edges of each energy channel in Angstrom
        angstrom_channels = (tof_channels * 3957) / self.fligt_path
        
        # calculate indeces of channels to be loaded based on self.wavelength_range
        # or self.intervals
        if (self.wavelength_range != -1):
            self._idx_left = next(x for x, val in enumerate(angstrom_channels[:, 0]) \
                                  if val > self.wavelength_range[0])
            self._idx_right = next(x for x, val in enumerate(angstrom_channels[:, 1]) \
                                   if val > self.wavelength_range[1])
            self._idx_right -= 1
            self._channel_edges = angstrom_channels[self._idx_left:(self._idx_right+1), :]
            
        elif (self.intervals != -1):
            self._idx_left = numpy.sum(n_channels_per_interval[:self.intervals[0]])
            self._idx_right = numpy.sum(n_channels_per_interval[:self.intervals[1]]) - 1
            self._channel_edges = angstrom_channels[self._idx_left:(self._idx_right+1), :]
            
        else:
            self._idx_left = 0
            self._idx_right = n_channels_total - 1
            self._channel_edges = angstrom_channels
        
        # calculate number of pixels and pixel size
        if ((self.binning == [1, 1]) and (self.roi == -1)):
            pixel_num_v = self.pixel_num_v_0
            pixel_num_h = self.pixel_num_h_0
            pixel_size_v = self.pixel_size_v_0
            pixel_size_h = self.pixel_size_h_0
            
        elif ((self.binning == [1, 1]) and (self.roi != -1)):
            pixel_num_v = self.roi[2] - self.roi[0]
            pixel_num_h = self.roi[3] - self.roi[1]
            pixel_size_v = self.pixel_size_v_0
            pixel_size_h = self.pixel_size_h_0
            
        elif ((self.binning > [1, 1]) and (self.roi == -1)):
            pixel_num_v = self.pixel_num_v_0 // self.binning[0]
            pixel_num_h = self.pixel_num_h_0 // self.binning[1]
            pixel_size_v = self.pixel_size_v_0 * self.binning[0]
            pixel_size_h = self.pixel_size_h_0 * self.binning[1]
            
        elif ((self.binning > [1, 1]) and (self.roi != -1)):
            pixel_num_v = (self.roi[2] - self.roi[0]) // self.binning[0]
            pixel_num_h = (self.roi[3] - self.roi[1]) // self.binning[1]
            pixel_size_v = self.pixel_size_v_0 * self.binning[0]
            pixel_size_h = self.pixel_size_h_0 * self.binning[1]
            
        # fill in metadata
        self._ag = AcquisitionGeometry(geom_type = 'parallel', 
                                       dimension = '3D', 
                                       angles = self.angles, 
                                       pixel_num_h = pixel_num_h, 
                                       pixel_size_h = pixel_size_h, 
                                       pixel_num_v = pixel_num_v, 
                                       pixel_size_v = pixel_size_v, 
                                       channels = self._idx_right - self._idx_left + 1,
                                       angle_unit = 'degree')
    
    
    def get_channel_edges(self):
        '''
        Return edges of every energy bin in Angstroms as a numpy array with 
        shape (channels x 2)
        '''
        return self._channel_edges
    
    
    def get_acquisition_geometry(self):
        '''
        Return AcquisitionGeometry object
        '''
        return self._ag
        
    
    def get_shutter_interval_id(self):
        '''
        Return numpy array with shutter interval ID for every channel 
        (required for overlap correction)
        '''
        return self._interval_id
    
    
    def get_n_shutter_intervals(self):
        '''
        Return number of shutter intervals (required for overlap correction)
        '''
        return self._n_intervals
    
    
    def get_shutter_counts(self):
        '''
        Parse text files with shutter counts and return dictionary:
            {"projection_shutter_counts": numpy array with shape (n_channels, n_angles),
             "flat_before_shutter_counts": numpy array with shape (n_channels, num_flat_before),
             "flat_after_shutter_counts": numpy array with shape (n_channels, num_flat_after)}
        (required for overlap correction)
        '''
        
        n_angles = numpy.shape(self.angles)[0]
        n_channels = self._idx_right - self._idx_left + 1
        
        res = {}
        
        projection_shutter_counts = numpy.zeros((n_channels, n_angles), dtype = 'int_')
        
        for i in range(n_angles):
            for j in range(n_channels):
                
                idx = self._idx_left + j
                
                # parse file with shutter counts
                shutter_counts = numpy.zeros(256, dtype = 'int_')
                
                filename_shutter_counts = (self.projection_path + 
                                           self.projection_prefix + 
                                           '{}' +
                                           '/' + 
                                           'IMAT{:08d}_' +
                                           self.projection_channel_prefix + 
                                           '_ShutterCount.txt').format(self.angles[i], self.projection_counter + i)
                
                # check if file with shutter counts exists
                if not(os.path.isfile(filename_shutter_counts)):
                    raise Exception('File {} does not exist'.format(filename_shutter_counts)) 
                
                with open(filename_shutter_counts) as shutter_counts_csv:
                    csv_reader = csv.reader(shutter_counts_csv, delimiter = '\t')
                
                    counter = 0
                
                    for row in csv_reader:
                        shutter_counts[counter] = float(row[1])
                        counter += 1            
        
                projection_shutter_counts[j, i] = shutter_counts[self._interval_id[idx]]
        
        res["projection_shutter_counts"] = projection_shutter_counts
        
        if (self.num_flat_before > 0):
            
            flat_before_shutter_counts = numpy.zeros((n_channels, self.num_flat_before), dtype = 'int_')
        
            for i in range(self.num_flat_before):
                for j in range(n_channels):
                    
                    idx = self._idx_left + j
                    
                    # parse file with shutter counts
                    shutter_counts = numpy.zeros(256, dtype = 'int_')
                    
                    filename_shutter_counts = (self.flat_before_path + 
                                               self.flat_before_prefix + 
                                               '{}' +
                                               '/' + 
                                               'IMAT{:08d}_' +
                                               self.flat_before_channel_prefix + 
                                               '_ShutterCount.txt').format(i + 1, self.flat_before_counter + i)
                    
                    # check if file with shutter counts exists
                    if not(os.path.isfile(filename_shutter_counts)):
                        raise Exception('File {} does not exist'.format(filename_shutter_counts)) 
                    
                    with open(filename_shutter_counts) as shutter_counts_csv:
                        csv_reader = csv.reader(shutter_counts_csv, delimiter = '\t')
                    
                        counter = 0
                    
                        for row in csv_reader:
                            shutter_counts[counter] = float(row[1])
                            counter += 1            
            
                    flat_before_shutter_counts[j, i] = shutter_counts[self._interval_id[idx]]
            
            res["flat_before_shutter_counts"] = flat_before_shutter_counts
            
        if (self.num_flat_after > 0):
            
            flat_after_shutter_counts = numpy.zeros((n_channels, self.num_flat_after), dtype = 'int_')
        
            for i in range(self.num_flat_after):
                for j in range(n_channels):
                    
                    idx = self._idx_left + j
                    
                    # parse file with shutter counts
                    shutter_counts = numpy.zeros(256, dtype = 'int_')
                    
                    filename_shutter_counts = (self.flat_after_path + 
                                               self.flat_after_prefix + 
                                               '{}' +
                                               '/' + 
                                               'IMAT{:08d}_' +
                                               self.flat_after_channel_prefix + 
                                               '_ShutterCount.txt').format(i + 1, self.flat_after_counter + i)
                    
                    # check if file with shutter counts exists
                    if not(os.path.isfile(filename_shutter_counts)):
                        raise Exception('File {} does not exist'.format(filename_shutter_counts)) 
                    
                    with open(filename_shutter_counts) as shutter_counts_csv:
                        csv_reader = csv.reader(shutter_counts_csv, delimiter = '\t')
                    
                        counter = 0
                    
                        for row in csv_reader:
                            shutter_counts[counter] = float(row[1])
                            counter += 1            
            
                    flat_after_shutter_counts[j, i] = shutter_counts[self._interval_id[idx]]
            
            res["flat_after_shutter_counts"] = flat_after_shutter_counts
            
        return res

    
    def load_projections(self):
        '''
        Load projections and returns AcquisitionData object
        '''
        
        n_angles = numpy.shape(self.angles)[0]
        n_channels = self._idx_right - self._idx_left + 1
        
        if (n_angles > 1):
            
            data = numpy.zeros((n_angles, n_channels, self._ag.pixel_num_v, self._ag.pixel_num_h), dtype = float)
            
            for i in range(n_angles):
                
                filename_mask = (self.projection_path + 
                                 self.projection_prefix + 
                                 '{}' +
                                 '/' + 
                                 'IMAT{:08d}_' +
                                  self.projection_channel_prefix
                                  ).format(self.angles[i], self.projection_counter + i) + \
                                 '_{:05d}.fits'
                
                data[i, :, :, :] = self._load(filename_mask)
            
            return AcquisitionData(array = data, 
                                   geometry = self._ag,
                                   dimension_labels = ['angle', 'channel', 'vertical', 'horizontal'])
        else:
            
            data = numpy.zeros((n_channels, self._ag.pixel_num_v, self._ag.pixel_num_h), dtype = float)
            
            filename_mask = (self.projection_path + 
                             self.projection_prefix + 
                             '{}' +
                             '/' + 
                             'IMAT{:08d}_' +
                             self.projection_channel_prefix
                             ).format(self.angles[0], self.projection_counter + i) + \
                             '_{:05d}.fits'
                
            data = self._load(filename_mask)
            
            return AcquisitionData(array = data, 
                                   geometry = self._ag,
                                   dimension_labels = ['channel', 'vertical', 'horizontal'])
    
    
    def load_flats_before(self):
        '''
        Loads flats before and returns numpy array with shape
        (num_flat_before, n_channels, pixel_num_v, pixel_num_h) if num_flat_before > 1
        or
        (n_channels, pixel_num_v, pixel_num_h) if num_flat_before = 1
        '''
        
        if ((self.flat_before_path == None) or 
            (self.flat_before_prefix == None) or 
            (self.flat_before_channel_prefix == None) or 
            (self.flat_before_counter == None) or
            (self.num_flat_before == 0)):
            raise Exception('A minimal set of following parameters is required \
                            to load flats acquired before scan: flat_before_path, \
                            flat_before_prefix, flat_before_channel_prefix, \
                            flat_before_counter and num_flat_before')
    
        n_channels = self._idx_right - self._idx_left + 1
        
        if (self.num_flat_before > 1):
            
            data = numpy.zeros((self.num_flat_before, n_channels, self._ag.pixel_num_v, self._ag.pixel_num_h), dtype = float)
            
            for i in range(self.num_flat_before):
                
                filename_mask = (self.flat_before_path + 
                                 self.flat_before_prefix + 
                                 '{}' +
                                 '/' + 
                                 'IMAT{:08d}_' +
                                  self.flat_before_channel_prefix
                                  ).format(i + 1, self.flat_before_counter + i) + \
                                 '_{:05d}.fits'
                
                data[i, :, :, :] = self._load(filename_mask)
                
        else:
            
            data = numpy.zeros((n_channels, self._ag.pixel_num_v, self._ag.pixel_num_h), dtype = float)
            
            filename_mask = (self.flat_before_path + 
                             self.flat_before_prefix + 
                             '{}' +
                             '/' + 
                             'IMAT{:08d}_' +
                             self.flat_before_channel_prefix
                             ).format(1, self.flat_before_counter + i) + \
                             '_{:05d}.fits'
                
            data = self._load(filename_mask)
            
        return data
    
    
    def load_flats_after(self):
        
        '''
        Loads flats after and returns numpy array with shape
        (num_flat_after, n_channels, pixel_num_v, pixel_num_h) if num_flat_after > 1
        or
        (n_channels, pixel_num_v, pixel_num_h) if num_flat_after = 1
        '''
        
        if ((self.flat_after_path == None) or 
            (self.flat_after_prefix == None) or 
            (self.flat_after_channel_prefix == None) or 
            (self.flat_after_counter == None) or
            (self.num_flat_after == 0)):
            raise Exception('A minimal set of following parameters is required \
                            to load flats acquired after scan: flat_after_path, \
                            flat_after_prefix, flat_after_channel_prefix, \
                            flat_after_counter and num_flat_after')
    
        n_channels = self._idx_right - self._idx_left + 1
        
        if (self.num_flat_after > 1):
            
            data = numpy.zeros((self.num_flat_after, n_channels, self._ag.pixel_num_v, self._ag.pixel_num_h), dtype = float)
            
            for i in range(self.num_flat_after):
                
                filename_mask = (self.flat_after_path + 
                                 self.flat_after_prefix + 
                                 '{}' +
                                 '/' + 
                                 'IMAT{:08d}_' +
                                  self.flat_after_channel_prefix
                                  ).format(i + 1, self.flat_after_counter + i) + \
                                 '_{:05d}.fits'
                
                data[i, :, :, :] = self._load(filename_mask)
        else:
            
            data = numpy.zeros((n_channels, self._ag.pixel_num_v, self._ag.pixel_num_h), dtype = float)
            
            filename_mask = (self.flat_after_path + 
                             self.flat_after_prefix + 
                             '{}' +
                             '/' + 
                             'IMAT{:08d}_' +
                             self.flat_after_channel_prefix
                             ).format(1, self.flat_after_counter + i) + \
                             '_{:05d}.fits'
                
            data = self._load(filename_mask)
            
        return data
    
        
    def _load(self,
        filename_mask):
        '''
        generic loader: loads projections or flats
        '''
        n_channels = self._idx_right - self._idx_left + 1
        
        # allocate array to store projections    
        data = numpy.zeros((n_channels, self._ag.pixel_num_v, self._ag.pixel_num_h), dtype = float)
        
        for i in range(n_channels):
            
            # generate filename
            filename = filename_mask.format(self._idx_left + i)
            
            if ((self.binning == [1, 1]) and (self.roi == -1)):
                with fits.open(filename) as file_handler:
                    data[i, :, :] = numpy.flipud(numpy.transpose(numpy.asarray(file_handler[0].data, dtype = float)))
                
            elif ((self.binning == [1, 1]) and (self.roi != -1)):
                with fits.open(filename) as file_handler:
                    tmp = numpy.asarray(file_handler[0].data[self.roi[1]:self.roi[3], 
                                                             (self.pixel_num_v_0 - self.roi[2]):(self.pixel_num_v_0 - self.roi[0])], dtype = float)
                    data[i, :, :] = numpy.flipud(numpy.transpose(tmp))
                
            elif ((self.binning > [1, 1]) and (self.roi == -1)):
                shape = (self._ag.pixel_num_v, self.binning[0], 
                         self._ag.pixel_num_h, self.binning[1])
                with fits.open(filename) as file_handler:
                    tmp = numpy.asarray(file_handler[0].data[:self._ag.pixel_num_h * self.binning[1],
                                                             self.pixel_num_v_0 - (self._ag.pixel_num_v * self.binning[0]):], dtype = float)
                    data[i, :, :] = (numpy.flipud(numpy.transpose(tmp))).reshape(shape).mean(-1).mean(1)
                        
            elif ((self.binning > [1, 1]) and (self.roi != -1)):
                shape = (self._ag.pixel_num_v, self.binning[0], 
                         self._ag.pixel_num_h, self.binning[1])
                with fits.open(filename) as file_handler:
                    tmp = numpy.asarray(file_handler[0].data[self.roi[1]:(self.roi[1] + (((self.roi[3] - self.roi[1]) // self.binning[1]) * self.binning[1])),
                                                             (self.pixel_num_v_0 - (self.roi[0] + (((self.roi[2] - self.roi[0]) // self.binning[0]) * self.binning[0]))):(self.pixel_num_v_0 - self.roi[0])], dtype = float)
                    data[i, :, :] = (numpy.flipud(numpy.transpose(tmp))).reshape(shape).mean(-1).mean(1)
        
        return data

'''
# usage example
# load angles
angles_file = open('/media/newhd/shared/Data/neutrondata/Feb2018_IMAT_rods/golden_ratio_angles.txt', 'r') 

angles = []
for angle in angles_file:
    angles.append(float(angle.strip('0')))
angles_file.close()

reader = IMATDataReader()  
reader.set_up(projection_path = '/media/newhd/shared/Data/neutrondata/IMAT_beamtime_feb_2019_raw_final/RB1820541/Tomo/', 
              projection_prefix = 'Angle_', 
              projection_channel_prefix = 'Tomo_000',
              projection_counter = 10699,
              angles = angles[0:10],
              binning = [7, 8],
              roi = [100, 150, 450, 480],
              shutter_values_file = '/media/newhd/shared/Data/neutrondata/IMAT_beamtime_feb_2019_raw_final/RB1820541/Tomo/ShutterValues.txt',
              intervals = -1,
              flat_after_path = '/media/newhd/shared/Data/neutrondata/IMAT_beamtime_feb_2019_raw_final/RB1820541/Tomo/',
              flat_after_prefix = 'Flat',
              flat_after_channel_prefix = ' Tomo_000',
              flat_after_counter = 10887,
              num_flat_after = 5,
              flat_before_path = '/media/newhd/shared/Data/neutrondata/IMAT_beamtime_feb_2019_raw_final/RB1820541/Tomo/',
              flat_before_prefix = 'Flat',
              flat_before_channel_prefix = ' Tomo_000',
              flat_before_counter = 10887,
              num_flat_before = 5) 


shutter_counts = reader.get_shutter_counts()
ag = reader.get_acquisition_geometry()
data = reader.load_projections()
flat_before = reader.load_flats_before()
flat_after = reader.load_flats_after()

plt.imshow(data.as_array()[1, 100, :, :])
plt.show()
'''
