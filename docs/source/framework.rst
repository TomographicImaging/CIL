Framework 
*********

``DataContainer`` and subclasses ``AcquisitionData`` and ``ImageData`` are 
meant to contain data and meta-data in ``AcquisitionGeometry`` and 
``ImageGeometry`` respectively.

DataContainer and subclasses
============================



.. autoclass:: ccpi.framework.DataContainer
   :members:
   :private-members:
   :special-members:
.. autoclass:: ccpi.framework.ImageData
   :members:
.. autoclass:: ccpi.framework.AcquisitionData
   :members:
.. autoclass:: ccpi.framework.VectorData
   :members:

.. autoclass:: ccpi.framework.ImageGeometry
   :members:
.. autoclass:: ccpi.framework.AcquisitionGeometry
   :members:
.. autoclass:: ccpi.framework.VectorGeometry
   :members:

Block Framework 
===============

The block framework allows writing complex `optimisation problems`_. These 
classes are required for it to work. They provide a base class that will 
behave as normal ``DataContainer``.

.. autoclass:: ccpi.framework.BlockDataContainer
   :members:
   :private-members:
   :special-members:
.. autoclass:: ccpi.framework.BlockGeometry
   :members:
   :private-members:
   :special-members:

DataProcessor
=============
.. autoclass:: ccpi.framework.DataProcessor
   :members:
   :no-undoc-members:
   :special-members:

.. autoclass:: ccpi.processors.CenterOfRotationFinder
   :members:
.. autoclass:: ccpi.processors.Normalizer
   :members:
.. autoclass:: ccpi.processors.Resizer
   :members:

:ref:`Return Home <mastertoc>`

.. _optimisation problems: optimisation.html
