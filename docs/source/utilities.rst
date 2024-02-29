..    Copyright 2021 United Kingdom Research and Innovation
      Copyright 2021 The University of Manchester
    
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
     
Utilities
*********


Test datasets
=============

A range of small test datasets to generate and use


A set of simulated volumes and CT data
--------------------------------------

.. autoclass:: cil.utilities.dataexample.SIMULATED_CONE_BEAM_DATA
   :members:

.. autoclass:: cil.utilities.dataexample.SIMULATED_PARALLEL_BEAM_DATA
   :members:

.. autoclass:: cil.utilities.dataexample.SIMULATED_CONE_BEAM_DATA
   :members:


A CT dataset from the Diamond Light Source
------------------------------------------

.. autoclass:: cil.utilities.dataexample.SYNCHROTRON_PARALLEL_BEAM_DATA
   :members:


Simulated image data
--------------------

.. autoclass:: cil.utilities.dataexample.TestData
   :members:
   :inherited-members:



Image Quality metrics
=====================

.. automodule:: cil.utilities.quality_measures
   :members:


Visualisation
============

show2D - Display 2D slices
--------------------------

.. autoclass:: cil.utilities.display.show2D
   :members:
   :inherited-members:

show1D - Display 1D slices
--------------------------

.. autoclass:: cil.utilities.display.show1D
   :members:
   :inherited-members:

show_geometry - Display system geometry
---------------------------------------

.. autoclass:: cil.utilities.display.show_geometry
   :members:
   :inherited-members:


islicer - interactive display of 2D slices
------------------------------------------

.. autoclass:: cil.utilities.jupyter.islicer
   :members:
   :inherited-members:


link_islicer - link islicer objects by index
--------------------------------------------

.. autoclass:: cil.utilities.jupyter.link_islicer
   :members:
   :inherited-members:


:ref:`Return Home <mastertoc>`