

```python
import os
from cil.processors import Binner, TransmissionAbsorptionConverter, Slicer, CentreOfRotationCorrector
from cil.optimisation.utilities import RandomSampling, SequentialSampling
from cil.utilities import dataexample
from cil.optimisation.functions import LeastSquares, SGFunction, SAGAFunction
from cil.plugins.astra import ProjectionOperator
from cil.optimisation.algorithms import FISTA, ISTA
from cil.plugins.ccpi_regularisation.functions import FGP_TV
from cil.optimisation.utilities import OptimalityDistance
from cil.utilities.display import show2D

import numpy as np
import matplotlib.pyplot as plt

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


## Deterministic Reconstruction


```python
alpha = 0.1
ag = data.geometry
ig = ag.get_ImageGeometry()
G = (alpha/ig.voxel_size_x) * FGP_TV(max_iteration = 100, device="gpu") 
A = ProjectionOperator(ig, ag, device = "gpu")

```


```python
initial = ig.allocate()
F_FISTA = LeastSquares(A, b = data, c = 0.5)
step_size_fista = 1./F_FISTA.L
fista = FISTA(initial = initial, f=F_FISTA, step_size = step_size_fista, g=G, update_objective_interval = 100, 
            max_iteration = 500)
fista.run(verbose=1)
```

         Iter   Max Iter     Time/Iter            Objective
                                   [s]                     
            0        500         0.000          2.52152e+05
          100        500         0.255          4.75054e+03
          200        500         0.255          4.75010e+03
          300        500         0.256          4.75013e+03
          400        500         0.256          4.75010e+03
          500        500         0.257          4.75007e+03
    -------------------------------------------------------
          500        500         0.257          4.75007e+03
    Stop criterion has been reached.
    



```python
num_batches = 13
data_split, method = data.split_to_subsets(num_batches, method= "ordered", info=True)

```

    WARNING:root:Batch size is (constant) self.num_indices//self.num_batches 



```python
def list_of_functions(data):
    
    list_funcs = []
    ig = data[0].geometry.get_ImageGeometry()
    
    for d in data:
        ageom_subset = d.geometry        
        Ai = ProjectionOperator(ig, ageom_subset, device = 'gpu')    
        fi = LeastSquares(Ai, b = d, c = 0.5)
        list_funcs.append(fi)   
        
    return list_funcs


list_func = list_of_functions(data_split)
```


```python
num_subsets = 13
selection = RandomSampling.uniform(len(list_func), num_subsets, seed=41) # shuffle=True
sg_func = SGFunction(list_func, selection=selection)
```

    WARNING:root:Batch size is (constant) self.num_indices//self.num_batches 



```python
selection.show_epochs(2)
```

     Epoch : 0, indices used : [8, 12, 10, 9, 9, 1, 3, 10, 4, 11, 2, 4, 12] 
     Epoch : 1, indices used : [6, 10, 7, 0, 4, 6, 8, 12, 9, 4, 11, 5, 4] 
    



```python
num_epochs = 100
step_size_ista = 1./sg_func.L
sgd = ISTA(initial = initial, f = sg_func, step_size = step_size_ista, g=G, 
            update_objective_interval = selection.num_batches, 
            max_iteration = num_epochs * selection.num_batches)  
ci = OptimalityDistance(sgd, fista.solution)
sgd.run(verbose=1, callback=ci)
```

         Iter   Max Iter     Time/Iter            Objective
                                   [s]                     
            0       1300         0.000          2.52163e+05
           13       1300         0.222          1.10165e+04
           26       1300         0.223          6.76761e+03
           39       1300         0.224          5.85847e+03
           52       1300         0.225          5.51966e+03
           65       1300         0.226          5.35203e+03
           78       1300         0.227          5.23418e+03
           91       1300         0.227          5.14465e+03
          104       1300         0.227          5.11278e+03
          117       1300         0.228          5.07592e+03
          130       1300         0.228          5.03367e+03
          143       1300         0.229          5.00120e+03
          156       1300         0.229          4.96286e+03
          169       1300         0.229          4.93151e+03
          182       1300         0.229          4.92002e+03
          195       1300         0.229          4.89513e+03
          208       1300         0.229          4.88118e+03
          221       1300         0.229          4.88496e+03
          234       1300         0.229          4.87001e+03
          247       1300         0.229          4.85471e+03
          260       1300         0.229          4.84755e+03
          273       1300         0.229          4.84864e+03
          286       1300         0.230          4.85553e+03
          299       1300         0.229          4.84468e+03
          312       1300         0.229          4.84701e+03
          325       1300         0.230          4.82785e+03
          338       1300         0.229          4.83429e+03
          351       1300         0.229          4.82521e+03
          364       1300         0.229          4.82734e+03
          377       1300         0.229          4.82151e+03
          390       1300         0.230          4.81510e+03
          403       1300         0.230          4.82564e+03
          416       1300         0.230          4.84147e+03
          429       1300         0.230          4.83294e+03
          442       1300         0.230          4.84293e+03
          455       1300         0.230          4.81976e+03
          468       1300         0.230          4.81623e+03
          481       1300         0.230          4.81928e+03
          494       1300         0.230          4.81758e+03
          507       1300         0.230          4.81662e+03
          520       1300         0.230          4.80171e+03
          533       1300         0.230          4.80449e+03
          546       1300         0.230          4.80448e+03
          559       1300         0.230          4.81435e+03
          572       1300         0.230          4.80617e+03
          585       1300         0.230          4.80518e+03
          598       1300         0.230          4.82175e+03
          611       1300         0.230          4.81344e+03
          624       1300         0.230          4.82498e+03
          637       1300         0.230          4.83271e+03
          650       1300         0.230          4.81568e+03
          663       1300         0.230          4.81736e+03
          676       1300         0.230          4.80894e+03
          689       1300         0.230          4.80896e+03
          702       1300         0.230          4.82132e+03
          715       1300         0.230          4.82557e+03
          728       1300         0.230          4.81261e+03
          741       1300         0.230          4.83628e+03
          754       1300         0.230          4.82536e+03
          767       1300         0.230          4.82247e+03
          780       1300         0.230          4.82990e+03
          793       1300         0.230          4.84714e+03
          806       1300         0.230          4.81943e+03
          819       1300         0.230          4.81302e+03
          832       1300         0.230          4.80889e+03
          845       1300         0.230          4.81401e+03
          858       1300         0.230          4.80012e+03
          871       1300         0.230          4.81280e+03
          884       1300         0.231          4.80094e+03
          897       1300         0.231          4.80446e+03
          910       1300         0.231          4.82129e+03
          923       1300         0.231          4.81089e+03
          936       1300         0.231          4.81584e+03
          949       1300         0.231          4.83076e+03
          962       1300         0.231          4.81326e+03
          975       1300         0.231          4.80866e+03
          988       1300         0.231          4.81976e+03
         1001       1300         0.231          4.82478e+03
         1014       1300         0.231          4.80956e+03
         1027       1300         0.231          4.81501e+03
         1040       1300         0.231          4.80746e+03
         1053       1300         0.231          4.80590e+03
         1066       1300         0.231          4.80776e+03
         1079       1300         0.231          4.81951e+03
         1092       1300         0.231          4.82822e+03
         1105       1300         0.231          4.79813e+03
         1118       1300         0.231          4.80804e+03
         1131       1300         0.231          4.79533e+03
         1144       1300         0.231          4.79922e+03
         1157       1300         0.231          4.81914e+03
         1170       1300         0.230          4.80559e+03
         1183       1300         0.230          4.80061e+03
         1196       1300         0.230          4.80831e+03
         1209       1300         0.230          4.82025e+03
         1222       1300         0.230          4.80262e+03
         1235       1300         0.230          4.84353e+03
         1248       1300         0.230          4.80732e+03
         1261       1300         0.230          4.81104e+03
         1274       1300         0.230          4.80435e+03
         1287       1300         0.230          4.80254e+03
         1300       1300         0.230          4.81062e+03
    -------------------------------------------------------
         1300       1300         0.230          4.81062e+03
    Stop criterion has been reached.
    



```python
initial = ig.allocate()
F_FISTA = LeastSquares(A, b = data, c = 0.5)
step_size_fista = 1./F_FISTA.L
fista1 = FISTA(initial = initial, f=F_FISTA, step_size = step_size_fista, g=G, update_objective_interval = 1, 
            max_iteration = num_epochs)
ci = OptimalityDistance(fista1, fista.solution)
fista1.run(verbose=0, callback=ci)
```


```python
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 4
mpl.rcParams['lines.markersize'] = 12

plt.figure(figsize=(30,20))
plt.semilogy([l for l in fista1.optimality_distance], label="FISTA", color="coral", marker="d")
plt.semilogy([l for l in sgd.optimality_distance], label="SGD", color="red", marker="d")
plt.ylabel("$||x_{k} - x^{*}||_{2}$",fontsize=40)
plt.xlabel("Epochs", fontsize=40)
plt.tick_params(axis='x',  labelsize=40)
plt.tick_params(axis='y',  labelsize=40)
plt.legend(loc='upper right', prop={'size': 40})
plt.grid()
plt.show()

plt.figure(figsize=(30,20))
plt.semilogy([np.abs(l-fista.objective[-1]) for l in fista1.objective], label="FISTA", color="coral", marker="d")
plt.semilogy([np.abs(l-fista.objective[-1]) for l in sgd.objective], label="SGD", color="red", marker="d")
plt.ylabel("$|f(x_{k}) - f(x^{*})|$",fontsize=40)
plt.xlabel("Epochs", fontsize=40)
plt.tick_params(axis='x',  labelsize=40)
plt.tick_params(axis='y',  labelsize=40)
plt.legend(loc='upper right', prop={'size': 40})
plt.grid()
plt.show()
```


![png](Tutorial_StochasticTomographyReconstruction_files/Tutorial_StochasticTomographyReconstruction_13_0.png)



![png](Tutorial_StochasticTomographyReconstruction_files/Tutorial_StochasticTomographyReconstruction_13_1.png)



```python
show2D([fista1.solution.array[:,70], sgd.solution.array[:,70], fista.solution.array[:,70]], 
       origin="upper", cmap="gray", 
       num_cols=3, fix_range=(0,0.1))
```


![png](Tutorial_StochasticTomographyReconstruction_files/Tutorial_StochasticTomographyReconstruction_14_0.png)





    <cil.utilities.display.show2D at 0x7ff6192e7670>




```python
show2D([np.abs(fista1.solution.array[:,70] - fista.solution.array[:,70]),
        np.abs(sgd.solution.array[:,70] - fista.solution.array[:,70]) ], 
       origin="upper", cmap="gray", fix_range=(0,0.001),
       num_cols=2)
```


![png](Tutorial_StochasticTomographyReconstruction_files/Tutorial_StochasticTomographyReconstruction_15_0.png)





    <cil.utilities.display.show2D at 0x7ff6192e1340>


