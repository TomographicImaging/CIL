
# Stochastic Gradient Descent for Tomography reconstruction

We will use the Stochastic Optimisation framework in CIL to reconstruct real X-ray data  with Total Variation regularisation.

$$\min_{u}\quad\frac{1}{2} \|A u - d\|^{2} + \alpha\,\mathrm{TV}(u)$$

We first read our tomographic data. We use the [Walnut Dataset](https://zenodo.org/record/4822516#.Y6Gu_OxBw0p) for this task. 

```python
data_name = "valnut_tomo-A.txrm"
filename = os.path.join(pathname,data_name )

reader = ZEISSDataReader()
reader.set_up(file_name=filename)
data3D = reader.read()
data3D.reorder('astra')
```

The [Walnut Dataset](https://zenodo.org/record/4822516#.Y6Gu_OxBw0p) is a 3D volume. However, for this tutorial, we use only the central slice and select half of the total number of projections.

```python
# import libs

import os
from cil.io import ZEISSDataReader
from cil.processors import Binner, TransmissionAbsorptionConverter, Slicer
from cil.plugins.astra import ProjectionOperator, FBP
from cil.optimisation.functions import LeastSquares, L2NormSquared
from cil.utilities.display import show2D, show_geometry
from cil.optimisation.algorithms import FISTA, ISTA
from cil.plugins.ccpi_regularisation.functions import FGP_TV
import numpy as np
```


```python
# Extract vertical slice
data2D = data3D.get_slice(vertical='centre')

# Select every 2 angles
sliced_data = Slicer(roi={'angle':(0,1601,2)})(data2D)

# Reduce background regions
binned_data = Binner(roi={'horizontal':(120,-120,2)})(sliced_data)
```

```python

# Create absorption data 
data = TransmissionAbsorptionConverter()(binned_data) 

# Remove circular artifacts
data -= np.mean(data.as_array()[80:100,0:30])

# Get Image and Acquisition geometries for one slice
ag2D = data.geometry
ag2D.set_angles(ag2D.angles, initial_angle=0.2, angle_unit='radian')
ig2D = ag2D.get_ImageGeometry()

A = ProjectionOperator(ig2D, ag2D, device = "gpu")

```

```python

print(" Acquisition Geometry 2D: {} with labels {}".format(ag2D.shape, ag2D.dimension_labels))
print(" Image Geometry 2D: {} with labels {}".format(ig2D.shape, ig2D.dimension_labels))

```

```bash
 Acquisition Geometry 2D: (800, 392) with labels ('angle', 'horizontal')
 Image Geometry 2D: (392, 392) with labels ('horizontal_y', 'horizontal_x')
```

```python
fbp_recon = FBP(ig2D, ag2D)(data)
show2D(fbp_recon, cmap="inferno", origin="upper", fix_range=(0,0.06))
```


```python
alpha = 0.003 # for walnut
G = (alpha/ig2D.voxel_size_x) * FGP_TV(max_iteration = 100, device="gpu") 
initial = ig2D.allocate()
F_FISTA = LeastSquares(A, b = data, c = 0.5)
step_size_ista = 1./F_FISTA.L
fista = FISTA(initial = initial, f=F_FISTA, step_size = step_size_ista, g=G, update_objective_interval = 1000, 
            max_iteration = 1000)
fista.run(verbose=1)
```