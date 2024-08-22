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

This method partitions an instance of tomography :code:`AcquisitionData` into a number of batches. For example, to use with a stochastic optimisation method. 

The partitioning is done by taking batches of angles and the corresponding data collected by taking projections along these angles. The partitioner method chooses what angles go in which batch depending on the `mode` and takes in an `AquisitionData` object and outputs a `BlockDataContainer` where each element in the block is  `AquisitionData` object with the batch of data and corresponding geometry. 
We consider a **batch** to be a subset of the :code:`AcquisitionData` and the verb, **to partition**, to be the act of splitting into batches. 
 

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

   # partition the data into batches contained in the elements of a BlockDataContainer
   data_partitioned = data.partition(num_batches=10, mode='staggered') # Choose mode from `sequential`, `staggered` or `random_permutation` 
   # From the partitioned data build a BlockOperator container the projectors for each batch 
   A_partitioned = ProjectionOperator(ig, data_partitioned.geometry, device = "cpu")

   print('The total number of angles is ', len(data.geometry.angles))
   print('The first 30 angles are ', data.geometry.angles[:30])

   print('In batch zero the number of angles is ', len(data_partitioned[0].geometry.angles))
   print('The angles in batch zero are ', data_partitioned[0].geometry.angles)
   print('The angles in batch one are ', data_partitioned[1].geometry.angles)

.. code-block :: RST

   The total number of angles is  300
   The first 30 angles are  [ 0.   1.2  2.4  3.6  4.8  6.   7.2  8.4  9.6 10.8 12.  13.2 14.4 15.6
   16.8 18.  19.2 20.4 21.6 22.8 24.  25.2 26.4 27.6 28.8 30.  31.2 32.4
   33.6 34.8]
   In batch zero the number of angles is  30
   The angles in batch zero are  [  0.  12.  24.  36.  48.  60.  72.  84.  96. 108. 120. 132. 144. 156.
   168. 180. 192. 204. 216. 228. 240. 252. 264. 276. 288. 300. 312. 324.
   336. 348.]
   The angles in batch one are  [  1.2  13.2  25.2  37.2  49.2  61.2  73.2  85.2  97.2 109.2 121.2 133.2
   145.2 157.2 169.2 181.2 193.2 205.2 217.2 229.2 241.2 253.2 265.2 277.2
   289.2 301.2 313.2 325.2 337.2 349.2]


The :code:`partition` method is defined as part of:

.. autoclass:: cil.framework.Partitioner
   :members:
   

Labels
=========
Classes which define the accepted labels

.. autoclass:: cil.framework.labels.ImageDimension
   :members:
   :undoc-members:

.. autoclass:: cil.framework.labels.AcquisitionDimension
   :members:
   :undoc-members:

.. autoclass:: cil.framework.labels.FillType
   :members:

.. autoclass:: cil.framework.labels.AngleUnit
   :members:
   :undoc-members:

.. autoclass:: cil.framework.labels.AcquisitionType
   :members:
   

DataProcessor
=============
.. autoclass:: cil.framework.DataProcessor
   :members:
   :inherited-members:

.. autoclass:: cil.framework.Processor
   :members:
   :inherited-members:

:ref:`Return Home <mastertoc>`
