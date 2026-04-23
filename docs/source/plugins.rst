..    Copyright 2019 United Kingdom Research and Innovation
      Copyright 2019 The University of Manchester

      Licensed under the Apache License, Version 2.0 (the "License");
      you may not use this file except in compliance with the License.
      You may obtain a copy of the License at

          http://www.apache.org/licenses/LICENSE-2.0

      Unless required by applicable law or agreed to in writing, software
      distributed under the License is distributed on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
      See the License for the specific language governing permissions and
      limitations under the License.

     Authors:
     CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt

CIL Plugins
************

CCPi Regularisation
===================

This plugin allows the use of regularisation functions from the `CCPi Regularisation toolkit
<https://github.com/vais-ral/CCPi-Regularisation-Toolkit>`_
(`10.1016/j.softx.2019.04.003 <https://www.sciencedirect.com/science/article/pii/S2352711018301912>`_,
a set of CPU/GPU optimised regularisation modules for iterative image reconstruction and
other image processing tasks.

Total variation
---------------

.. autoclass:: cil.plugins.ccpi_regularisation.functions.FGP_TV


Other regularisation functions
------------------------------

.. autoclass:: cil.plugins.ccpi_regularisation.functions.TGV
   :members:
   :special-members:

.. autoclass:: cil.plugins.ccpi_regularisation.functions.FGP_dTV
   :members:
   :special-members:

.. autoclass:: cil.plugins.ccpi_regularisation.functions.TNV
   :members:
   :special-members:


TomoPhantom
===========
This plugin allows the use of part of `TomoPhantom
<https://github.com/dkazanc/TomoPhantom>`_
(`10.1016/j.softx.2018.05.003 <https://doi.org/10.1016/j.softx.2018.05.003>`_,
a toolbox written in C language to generate customisable 2D-4D phantoms (with a
temporal capability).

.. autofunction:: cil.plugins.TomoPhantom.get_ImageData

TIGRE
=====
This plugin allows the use of `TIGRE
<https://github.com/CERN/TIGRE>`_
(`10.1088/2057-1976/2/5/055010 <http://iopscience.iop.org/article/10.1088/2057-1976/2/5/055010>`_
for forward and back projections and filter back projection reconstruction.

FBP
---
This reconstructs with FBP for parallel-beam data, and with FDK weights for cone-beam data

.. autoclass:: cil.plugins.tigre.FBP
   :exclude-members: check_input, get_input
   :members:
   :inherited-members: set_input, get_output


Projection Operator
-------------------

.. autoclass:: cil.plugins.tigre.ProjectionOperator
   :members:



ASTRA
=====
This plugin allows the use of `ASTRA-toolbox
<https://github.com/astra-toolbox/astra-toolbox>`_
(`10.1364/OE.24.025129 <http://dx.doi.org/10.1364/OE.24.025129>`_)
for forward and back projections and filter back projection reconstruction.


FBP
---
This reconstructs with FBP for parallel-beam data, and with FDK weights for cone-beam data

.. autoclass:: cil.plugins.astra.FBP
   :exclude-members: check_input, get_input
   :members:
   :inherited-members: set_input, get_output


Projection Operator
-------------------

.. autoclass:: cil.plugins.astra.ProjectionOperator
   :members:

:ref:`Return Home <mastertoc>`
