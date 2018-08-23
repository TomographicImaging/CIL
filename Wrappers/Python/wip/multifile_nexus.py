# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 16:00:53 2018

@author: ofn77899
"""

import os
from ccpi.io.reader import NexusReader

from sys import getsizeof

import matplotlib.pyplot as plt

from ccpi.framework import DataProcessor, DataContainer
from ccpi.processors import Normalizer
from ccpi.processors import CenterOfRotationFinder
import numpy


class averager(object):
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.N = 0
        self.avg = 0
        self.min = 0
        self.max = 0
        self.var = 0
        self.ske = 0
        self.kur = 0
    
    def add_reading(self, val):
        
        if (self.N == 0):
           self.avg = val;
           self.min = val;
           self.max = val;
        elif (self.N == 1):
        #//set min/max
            self.max = val if val > self.max else self.max
            self.min = val if val < self.min else self.min
            
      
            thisavg = (self.avg + val)/2
            #// initial value is in avg
            self.var = (self.avg - thisavg)*(self.avg-thisavg) + (val - thisavg) * (val-thisavg)
            self.ske = self.var * (self.avg - thisavg)
            self.kur = self.var * self.var
            self.avg = thisavg
        else:
            self.max = val if val > self.max else self.max
            self.min = val if val < self.min else self.min
            
            M = self.N

            #// b-factor =(<v>_N + v_(N+1)) / (N+1)
            #float b = (val -avg) / (M+1);
            b = (val -self.avg) / (M+1)
            
            self.kur = self.kur + (M *(b*b*b*b) * (1+M*M*M))- (4* b * self.ske) + (6 * b *b * self.var * (M-1))
    
            self.ske = self.ske + (M * b*b*b *(M-1)*(M+1)) - (3 * b * self.var * (M-1))
    
            #//var = var * ((M-1)/M) + ((val - avg)*(val - avg)/(M+1)) ;
            self.var = self.var * ((M-1)/M) + (b * b * (M+1))
            
            self.avg = self.avg * (M/(M+1)) + val / (M+1)
    
        self.N += 1
    
    def stats(self, vector):
        i = 0
        while i < vector.size:
            self.add_reading(vector[i])
            i+=1

avg = averager()
a = numpy.linspace(0,39,40)
avg.stats(a)
print ("average" , avg.avg, a.mean())
print ("variance" , avg.var, a.var())
b = a - a.mean()
b *= b
b = numpy.sqrt(sum(b)/(a.size-1))
print ("std" , numpy.sqrt(avg.var), b)
#%%         

class DataStatMoments(DataProcessor):
    '''Normalization based on flat and dark
    
    This processor read in a AcquisitionData and normalises it based on 
    the instrument reading with and without incident photons or neutrons.
    
    Input: AcquisitionData
    Parameter: 2D projection with flat field (or stack)
               2D projection with dark field (or stack)
    Output: AcquisitionDataSetn
    '''
    
    def __init__(self, axis, skewness=False, kurtosis=False, offset=0):
        kwargs = {
                  'axis'     : axis,
                  'skewness' : skewness,
                  'kurtosis' : kurtosis,
                  'offset'   : offset,
                  }
        #DataProcessor.__init__(self, **kwargs)
        super(DataStatMoments, self).__init__(**kwargs)
        
    
    def check_input(self, dataset):
        #self.N = dataset.get_dimension_size(self.axis)
        return True
    
    @staticmethod
    def add_sample(dataset, N, axis, stats=None, offset=0):
        #dataset = self.get_input()
        if (N == 0):
            # get the axis number along to calculate the stats
            
            
            #axs = dataset.get_dimension_size(self.axis)
            # create a placeholder for the output
            if stats is None:
                ax = dataset.get_dimension_axis(axis)
                shape = [dataset.shape[i] for i in range(len(dataset.shape)) if i != ax]
                # output has 4 components avg, std, skewness and kurtosis + last avg+ (val-thisavg)
                shape.insert(0, 4+2)
                stats = numpy.zeros(shape)
                
            stats[0] = dataset.subset(**{axis:N-offset}).array[:]
            
            #avg = val
        elif (N == 1):
            # val
            stats[5] = dataset.subset(**{axis:N-offset}).array
            stats[4] = stats[0] + stats[5]
            stats[4] /= 2         # thisavg
            stats[5] -= stats[4]  # (val - thisavg) 
            
            #float thisavg = (avg + val)/2;
            
            #// initial value is in avg
            #var = (avg - thisavg)*(avg-thisavg) + (val - thisavg) * (val-thisavg);
            stats[1] = stats[5] * stats[5] + stats[5] * stats[5]
            #skewness = var * (avg - thisavg);
            stats[2] = stats[1] * stats[5]
            #kurtosis = var * var;
            stats[3] = stats[1] * stats[1]
            #avg = thisavg;
            stats[0] = stats[4]
        else:
        
            #float M = (float)N;
            M = N
            #val
            stats[4] = dataset.subset(**{axis:N-offset}).array
            #// b-factor =(<v>_N + v_(N+1)) / (N+1)
            stats[5] = stats[4] - stats[0]
            stats[5] /= (M+1)    # b factor
            #float b = (val -avg) / (M+1);
    
            #kurtosis = kurtosis + (M *(b*b*b*b) * (1+M*M*M))- (4* b * skewness) + (6 * b *b * var * (M-1));
            #if self.kurtosis:
            #    stats[3] += (M * stats[5] * stats[5] * stats[5] * stats[5]) - \
            #                (4 * stats[5] * stats[2]) + \
            #                (6 * stats[5] * stats[5] * stats[1] * (M-1))
                        
            #skewness = skewness + (M * b*b*b *(M-1)*(M+1)) - (3 * b * var * (M-1));
            #if self.skewness:
            #    stats[2] = stats[2] +  (M * stats[5]* stats[5] * stats[5] * (M-1)*(M-1) ) -\
            #                3 * stats[5] * stats[1] * (M-1) 
            #//var = var * ((M-1)/M) + ((val - avg)*(val - avg)/(M+1)) ;
            #var = var * ((M-1)/M) + (b * b * (M+1));
            stats[1] = ((M-1)/M) * stats[1] + (stats[5] * stats[5] * (M+1))
           
            #avg = avg * (M/(M+1)) + val / (M+1)
            stats[0] = stats[0] * (M/(M+1)) + stats[4] / (M+1)
    
        N += 1
        return stats , N
    
           
    def process(self):
        
        data = self.get_input()
        
        #stats, i = add_sample(0)
        N = data.get_dimension_size(self.axis)
        ax = data.get_dimension_axis(self.axis)
        stats = None
        i = 0
        while i < N:
            stats , i = DataStatMoments.add_sample(data, i, self.axis, stats, offset=self.offset)
        self.offset += N
        labels = ['StatisticalMoments']
        
        labels += [data.dimension_labels[i] \
                   for i in range(len(data.dimension_labels)) if i != ax]
        y = DataContainer( stats[:4] , False, 
                    dimension_labels=labels)
        return y

directory = r'E:\Documents\Dataset\CCPi\Nexus_test'
data_path="entry1/instrument/pco1_hw_hdf_nochunking/data"

reader = NexusReader(os.path.join( os.path.abspath(directory) , '74331.nxs'))

print ("read flat")
read_flat = NexusReader(os.path.join( os.path.abspath(directory) , '74240.nxs'))
read_flat.data_path = data_path
flatsslice = read_flat.get_acquisition_data_whole()
avg = DataStatMoments('angle')

avg.set_input(flatsslice)
flats = avg.get_output()

ave = averager()
ave.stats(flatsslice.array[:,0,0])

print ("avg" , ave.avg, flats.array[0][0][0])
print ("var" , ave.var, flats.array[1][0][0])

print ("read dark")
read_dark = NexusReader(os.path.join( os.path.abspath(directory) , '74243.nxs'))
total_size = read_dark.get_projection_dimensions()[0]

## darks are very many so we proceed in batches
batchsize = 40
batchlimits = [batchsize * (i+1) for i in range(int(total_size/batchsize))] + [total_size]
#avg.N = 0
avg.offset = 0
N = 0
for batch in range(len(batchlimits)):  
    print ("running batch " , batch)
    bmax = batchlimits[batch]
    bmin = bmax-batchsize
    
    darksslice = read_dark.get_acquisition_data_batch(bmin,bmax)
    if batch == 0:
        #create stats
        ax = darksslice.get_dimension_axis('angle')
        shape = [darksslice.shape[i] for i in range(len(darksslice.shape)) if i != ax]
        # output has 4 components avg, std, skewness and kurtosis + last avg+ (val-thisavg)
        shape.insert(0, 4+2)
        print ("create stats shape ", shape)
        stats = numpy.zeros(shape)
    print ("N" , N)        
    #avg.set_input(darksslice)
    i = bmin
    while i < bmax:
        stats , i = DataStatMoments.add_sample(darksslice, i, 'angle', stats, bmin)
        print ("{0}-{1}-{2}".format(bmin, i , bmax ) )
    
darks = stats
#%%

fig = plt.subplot(2,2,1)
fig.imshow(flats.subset(StatisticalMoments=0).array)
fig = plt.subplot(2,2,2)
fig.imshow(numpy.sqrt(flats.subset(StatisticalMoments=1).array))
fig = plt.subplot(2,2,3)
fig.imshow(darks.subset(StatisticalMoments=0).array)
fig = plt.subplot(2,2,4)
fig.imshow(numpy.sqrt(darks.subset(StatisticalMoments=1).array))
plt.show()


#%%
norm = Normalizer(flat_field=flats.array[0,200,:], dark_field=darks.array[0,200,:])
#norm.set_flat_field(flats.array[0,200,:])
#norm.set_dark_field(darks.array[0,200,:])
norm.set_input(reader.get_acquisition_data_slice(200))

n = Normalizer.normalize_projection(norm.get_input().as_array(), flats.array[0,200,:], darks.array[0,200,:], 1e-5)

#%%


#%%
fig = plt.subplot(2,1,1)


fig.imshow(norm.get_input().as_array())
fig = plt.subplot(2,1,2)
fig.imshow(norm.get_output().as_array())


plt.show()






