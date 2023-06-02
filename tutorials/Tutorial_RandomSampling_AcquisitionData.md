

```python
import os
from cil.processors import Binner, TransmissionAbsorptionConverter, Slicer, CentreOfRotationCorrector
from cil.optimisation.utilities import RandomSampling, SequentialSampling
from cil.utilities import dataexample

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
```

### Read SYNCHROTRON_PARALLEL_BEAM_DATA 


```python
data = dataexample.SYNCHROTRON_PARALLEL_BEAM_DATA.get()
data_raw20 = data.get_slice(vertical=20)
scale = data_raw20.sum()/data_raw20.size
data /= scale
data.log(out=data)
data *= -1
data = CentreOfRotationCorrector.xcorrelation(slice_index='centre')(data)
data.reorder('astra')
```


```python
print("Acquisition Data shape", data.shape)
print("Acquisition Data geometry labels", data.geometry.dimension_labels)
```

    Acquisition Data shape (135, 91, 160)
    Acquisition Data geometry labels ('vertical', 'angle', 'horizontal')


### Number of projection angles = 91, divisors =1, 7, 13, and 91

Split acquisition data into subsets with uniform random sampling **without** replacement.


```python
num_batches = 13
data_split, method = data.split_to_subsets(num_batches, method= "random", info=True)

print("Number of subsets = {}".format(method.num_batches))

for i,k in enumerate(method.partition_list):
    print("Subset {} : {}\n".format(i,k))

for i,k in enumerate(data_split):
    print("Subset {} : Angles used = {}\n".format(i,list(k.geometry.angles)))
    
    
slice_ind = 50

plt.rcParams['xtick.labelsize']=15
plt.rcParams['ytick.labelsize']=15

fig = plt.figure(figsize=(20, 20)) 
grid = AxesGrid(fig, 111,
                nrows_ncols=(method.num_batches, 1),
                axes_pad=0.5
                )
k = 0
for ax in grid:    
    ax.imshow(data_split[k].array[slice_ind], cmap="inferno")
    ax.set_title("Angles used = {}".format(list(data_split[k].geometry.angles)),fontsize=15)    
    k+=1    
plt.show()
    
```

    WARNING:root:Batch size is (constant) self.num_indices//self.num_batches 


    Number of subsets = 13
    Subset 0 : [19  6 74  4 76 82 79]
    
    Subset 1 : [24 86 71 73 85 28  0]
    
    Subset 2 : [33 29 13 42 17 66 10]
    
    Subset 3 : [47 41 36  1 63 30 38]
    
    Subset 4 : [35 20  5 52 80 51 49]
    
    Subset 5 : [81 25 12 14 46 58 57]
    
    Subset 6 : [78 22 69 15  7 70 11]
    
    Subset 7 : [21 62 56 34 44  8  3]
    
    Subset 8 : [45 87 43 68  2 84 23]
    
    Subset 9 : [88 27 48 72 31 16 55]
    
    Subset 10 : [89 60 83 77 61 75 65]
    
    Subset 11 : [59 26 18 67  9 40 53]
    
    Subset 12 : [64 90 54 50 37 39 32]
    
    Subset 0 : Angles used = [-50.2, -76.1999, 59.8, -80.2, 63.8, 75.8, 69.8]
    
    Subset 1 : Angles used = [-40.2, 83.8, 53.8, 57.8, 81.8, -32.2, -88.2]
    
    Subset 2 : Angles used = [-22.2, -30.2, -62.2, -4.1999, -54.2, 43.7999, -68.2]
    
    Subset 3 : Angles used = [5.8, -6.2, -16.2, -86.2, 37.8, -28.2, -12.2]
    
    Subset 4 : Angles used = [-18.2, -48.2, -78.2, 15.7999, 71.8, 13.8, 9.8]
    
    Subset 5 : Angles used = [73.8, -38.2, -64.1999, -60.2, 3.8001, 27.8, 25.8]
    
    Subset 6 : Angles used = [67.8001, -44.1999, 49.8, -58.2, -74.2, 51.8, -66.2]
    
    Subset 7 : Angles used = [-46.2, 35.8, 23.8, -20.2, -0.2, -72.1999, -82.2]
    
    Subset 8 : Angles used = [1.8, 85.8, -2.2, 47.8, -84.2001, 79.8, -42.1999]
    
    Subset 9 : Angles used = [87.8, -34.2, 7.8, 55.8, -26.2, -56.2, 21.8]
    
    Subset 10 : Angles used = [89.8, 31.8, 77.8, 65.8, 33.8, 61.8, 41.8]
    
    Subset 11 : Angles used = [29.8, -36.1997, -52.2, 45.8, -70.2, -8.2, 17.8]
    
    Subset 12 : Angles used = [39.8, 91.7999, 19.8, 11.7999, -14.2, -10.1999, -24.2]
    



![png](Tutorial_RandomSampling_AcquisitionData_files/Tutorial_RandomSampling_AcquisitionData_5_2.png)


### Number of projection angles = 91, divisors =1, 7, 13, and 91

Sampling method is created with different seed.


```python
num_batches = 13
method = RandomSampling.uniform(91, num_batches, replace=False, seed=40)
data_split = data.split_to_subsets(method=method)


print("Number of subsets = {}".format(method.num_batches))

for i,k in enumerate(method.partition_list):
    print("Subset {} : {}\n".format(i,k))

for i,k in enumerate(data_split):
    print("Subset {} : Angles used = {}\n".format(i,list(k.geometry.angles)))
    
    
slice_ind = 50

plt.rcParams['xtick.labelsize']=15
plt.rcParams['ytick.labelsize']=15

fig = plt.figure(figsize=(20, 20)) 
grid = AxesGrid(fig, 111,
                nrows_ncols=(method.num_batches, 1),
                axes_pad=0.5
                )
k = 0
for ax in grid:    
    ax.imshow(data_split[k].array[slice_ind], cmap="inferno")
    ax.set_title("Angles used = {}".format(list(data_split[k].geometry.angles)),fontsize=15)    
    k+=1    
plt.show()
    
```

    WARNING:root:Batch size is (constant) self.num_indices//self.num_batches 


    Number of subsets = 13
    Subset 0 : [90 83 25 49 28 66 12]
    
    Subset 1 : [ 5 64 32 81  0 31 76]
    
    Subset 2 : [14 84 78 20 26 69 62]
    
    Subset 3 : [73 39  6  1 24 46 45]
    
    Subset 4 : [85 75 67 47 74 79 71]
    
    Subset 5 : [63 16 41 58 23 88 56]
    
    Subset 6 : [86  3 82 65 42 33  8]
    
    Subset 7 : [77 21 61 55 30 10 15]
    
    Subset 8 : [44 59 11 60 34 57 36]
    
    Subset 9 : [89 43 40 17 54 72 68]
    
    Subset 10 : [19  7 35 87  2 29 53]
    
    Subset 11 : [37 50  4 22 13 38 51]
    
    Subset 12 : [52 48 18  9 80 27 70]
    
    Subset 0 : Angles used = [91.7999, 77.8, -38.2, 9.8, -32.2, 43.7999, -64.1999]
    
    Subset 1 : Angles used = [-78.2, 39.8, -24.2, 73.8, -88.2, -26.2, 63.8]
    
    Subset 2 : Angles used = [-60.2, 79.8, 67.8001, -48.2, -36.1997, 49.8, 35.8]
    
    Subset 3 : Angles used = [57.8, -10.1999, -76.1999, -86.2, -40.2, 3.8001, 1.8]
    
    Subset 4 : Angles used = [81.8, 61.8, 45.8, 5.8, 59.8, 69.8, 53.8]
    
    Subset 5 : Angles used = [37.8, -56.2, -6.2, 27.8, -42.1999, 87.8, 23.8]
    
    Subset 6 : Angles used = [83.8, -82.2, 75.8, 41.8, -4.1999, -22.2, -72.1999]
    
    Subset 7 : Angles used = [65.8, -46.2, 33.8, 21.8, -28.2, -68.2, -58.2]
    
    Subset 8 : Angles used = [-0.2, 29.8, -66.2, 31.8, -20.2, 25.8, -16.2]
    
    Subset 9 : Angles used = [89.8, -2.2, -8.2, -54.2, 19.8, 55.8, 47.8]
    
    Subset 10 : Angles used = [-50.2, -74.2, -18.2, 85.8, -84.2001, -30.2, 17.8]
    
    Subset 11 : Angles used = [-14.2, 11.7999, -80.2, -44.1999, -62.2, -12.2, 13.8]
    
    Subset 12 : Angles used = [15.7999, 7.8, -52.2, -70.2, 71.8, -34.2, 51.8]
    



![png](Tutorial_RandomSampling_AcquisitionData_files/Tutorial_RandomSampling_AcquisitionData_7_2.png)

