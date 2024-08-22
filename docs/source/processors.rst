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


Centre Of Rotation Corrector
----------------------------

In the ideal alignment of a CT instrument, the projection of the axis of rotation onto the
detector coincides with the vertical midline of the detector. In practice this is hard to achieve
due to misalignments and/or kinematic errors in positioning of CT instrument components.
A slight offset of the center of rotation with respect to the theoretical position will contribute
to the loss of resolution; in more severe cases, it will cause severe artifacts in the reconstructed
volume (double-borders).

:code:`CentreOfRotationCorrector` can be used to estimate the offset of center of rotation from the data.

:code:`CentreOfRotationCorrector` supports both parallel and cone-beam geometry with 2 different algorithms:

* Cross-correlation, is suitable for single slice parallel-beam geometry. It requires two projections 180 degree apart.

* Image sharpness method, which maximising the sharpness of a reconstructed slice. It can be used on single
slice parallel-beam, and centre-slice of cone-beam geometry. For use only with datasets that can be reconstructed with FBP/FDK.


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


Paganin Processor
-----------------

.. autoclass:: cil.processors.PaganinProcessor
   :exclude-members: check_input, get_input
   :members:
   :inherited-members:

:ref:`Return Home <mastertoc>`
