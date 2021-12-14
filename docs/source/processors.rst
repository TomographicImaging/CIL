Processors
**********

This module allows the user to manipulate or pre-process their data.

Data Manipulation
=================

These processors can be used on `ImageData` or `AcquisitionData` objects.


Data Slicer
-----------

.. autoclass:: cil.processors.Slicer
   :exclude-members: check_input, get_input
   :members:
   :inherited-members: set_input, get_output


Data Binner
-----------

.. autoclass:: cil.processors.Binner
   :exclude-members: check_input, get_input
   :members:
   :inherited-members: set_input, get_output


Data Padder
-----------

.. autoclass:: cil.processors.Padder
   :exclude-members: check_input, get_input
   :members:
   :inherited-members: set_input, get_output


Mask Generator from Data
------------------------

.. autoclass:: cil.processors.MaskGenerator
   :exclude-members: check_input, get_input
   :members:
   :inherited-members: set_input, get_output


Data Masking
------------

.. autoclass:: cil.processors.Masker
   :exclude-members: check_input, get_input
   :members:
   :inherited-members: set_input, get_output


Pre-processors
==============

These processors can be used with `AcquisitionData` objects


Centre Of Rotation Correction
-----------------------------

.. autoclass:: cil.processors.CentreOfRotationCorrector
   :exclude-members: check_input, get_input
   :members:
   :inherited-members: set_input, get_output


Data Normaliser
---------------

.. autoclass:: cil.processors.Normaliser
   :exclude-members: check_input, get_input
   :members:
   :inherited-members: set_input, get_output


Transmission to Absorption Converter
-------------------------------------

.. autoclass:: cil.processors.TransmissionAbsorptionConverter
   :exclude-members: check_input, get_input
   :members:
   :inherited-members: set_input, get_output


Absorption to Transmission Converter
------------------------------------

.. autoclass:: cil.processors.AbsorptionTransmissionConverter
   :exclude-members: check_input, get_input
   :members:
   :inherited-members: set_input, get_output


Ring Remover
------------

.. autoclass:: cil.processors.RingRemover
   :exclude-members: check_input, get_input
   :members:
   :inherited-members: set_input, get_output
   

:ref:`Return Home <mastertoc>`