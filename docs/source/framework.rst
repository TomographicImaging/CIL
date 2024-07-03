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

Framework
*********


AcquisitionGeometry
===================

The :code:`AcquisitionGeometry` class holds the system acquisition parameters.

.. autoclass:: cil.framework.AcquisitionGeometry

We create the appropriate :code:`AcquisitionGeometry` for our data by using the static methods:

Parallel2D Geometry
-------------------
.. automethod:: cil.framework.AcquisitionGeometry.create_Parallel2D

Parallel3D Geometry
-------------------
.. automethod:: cil.framework.AcquisitionGeometry.create_Parallel3D

Cone2D Geometry (Fanbeam)
-------------------------
.. automethod:: cil.framework.AcquisitionGeometry.create_Cone2D

Cone3D Geometry
---------------
.. automethod:: cil.framework.AcquisitionGeometry.create_Cone3D


Configure the geometry
----------------------
This gives us an acquisition geometry object configured with the spatial geometry of the system.

It is then necessary to configure the panel, angular data and dimension labels:

.. automethod:: cil.framework.AcquisitionGeometry.set_panel
.. automethod:: cil.framework.AcquisitionGeometry.set_angles
.. automethod:: cil.framework.AcquisitionGeometry.set_labels
.. automethod:: cil.framework.AcquisitionGeometry.set_channels


Use the geometry
----------------
We can use this geometry to generate related objects. Including a 2D slice :code:`AcquisitionGeometry`, an :code:`AcquisitionData` container, and a default :code:`ImageGeometry`.

.. automethod:: cil.framework.AcquisitionGeometry.get_slice
.. automethod:: cil.framework.AcquisitionGeometry.allocate
.. automethod:: cil.framework.AcquisitionGeometry.get_ImageGeometry



ImageGeometry
=============


The :code:`ImageGeometry` class holds meta data describing the reconstruction volume of interest. This will be centred on the rotation axis position defined
in :code:`AcquisitionGeometry`, with the z-direction aligned with the rotation axis direction.

.. autoclass:: cil.framework.ImageGeometry
   :members:


BlockGeometry
=============

.. autoclass:: cil.framework.BlockGeometry
   :members:
   :inherited-members:



Data Containers
===============

:code:`AcquisitionData` and :code:`ImageData` inherit from the same parent :code:`DataContainer` class,
therefore they largely behave the same way.

There are algebraic operations defined for both :code:`AcquisitionData` and :code:`ImageData`.
Following operations are defined:

* binary operations (between two DataContainers or scalar and DataContainer)

  * :code:`+` addition
  * :code:`-` subtraction
  * :code:`/` division
  * :code:`*` multiplication
  * :code:`**` power
  * :code:`maximum`
  * :code:`minimum`

* in-place operations

  * :code:`+=`
  * :code:`-=`
  * :code:`*=`
  * :code:`**=`
  * :code:`/=`

* unary operations

  * :code:`abs`
  * :code:`sqrt`
  * :code:`sign`
  * :code:`conjugate`

* reductions

  * :code:`sum`
  * :code:`norm`
  * :code:`dot` product


DataContainer
-------------

.. autoclass:: cil.framework.DataContainer
   :members:
   :inherited-members:

AcquisitionData
---------------

.. autoclass:: cil.framework.AcquisitionData
   :members:
   :inherited-members:

ImageData
---------

.. autoclass:: cil.framework.ImageData
   :members:
   :inherited-members:

VectorData
----------

.. autoclass:: cil.framework.VectorData
   :members:
   :inherited-members:


BlockDataContainer
------------------

A :code:`BlockDataContainer` can be instantiated from a number of `DataContainer`_ and subclasses
represents a column vector of :code:`DataContainer` s.

.. code:: python

  bdc = BlockDataContainer(DataContainer0, DataContainer1)

This provide a base class that will behave as normal :code:`DataContainer`.

.. autoclass:: cil.framework.BlockDataContainer
   :members:
   :inherited-members:

Partitioner
===========

This class allows a user to partition an instance of :code:`AcquisitionData` into a number of subsets. For example, to use with a stochastic optimisation method. 

For example: 

.. code-block :: python

   from cil.utilities import dataexample
   from cil.plugins.astra.operators import ProjectionOperator
   
   # get the data  
   data = dataexample.SIMULATED_PARALLEL_BEAM_DATA.get()
   data.reorder('astra')
   data = data.get_slice(vertical='centre')

   # create the geometries 
   ag = data.geometry 
   ig = ag.get_ImageGeometry()

   # partition the data and build the projectors
   n_batches = 10 
   partitioned_data = data.partition(num_batches=n_batches, mode='sequential') # Choose mode from `sequential`, `staggered` or `random_permutation` 
   A_partitioned = ProjectionOperator(ig, partitioned_data.geometry, device = "cpu")


The `partition` method is defined as part of:

.. autoclass:: cil.framework.Partitioner
   :members:
   

DataOrder
=========
.. autoclass:: cil.framework.DataOrder
   :members:
   :inherited-members:

DataProcessor
=============
.. autoclass:: cil.framework.DataProcessor
   :members:
   :inherited-members:

.. autoclass:: cil.framework.Processor
   :members:
   :inherited-members:

:ref:`Return Home <mastertoc>`
