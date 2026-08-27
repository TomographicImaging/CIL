..    Copyright 2026 United Kingdom Research and Innovation
      Copyright 2026 The University of Manchester

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

Dependencies & Compatibility
*****************************

While building the CIL package we test with specific versions of dependencies. These are listed in the `build.yml <https://github.com/TomographicImaging/CIL/blob/master/.github/workflows/build.yml>`_ GitHub workflow and `cil_development.yml <https://github.com/TomographicImaging/CIL/blob/master/scripts/cil_development.yml>`_.
The following table tries to resume the tested versions of CIL and its required and optional dependencies. If you use these packages as a backend please remember to cite them in addition to CIL.

Core Dependencies
=================
.. list-table::
   :header-rows: 1
   :widths: 18 10 22 30 15

   * - Package
     - Tested Version
     - Conda install command
     - Description
     - License

   * - `Python <https://www.python.org/>`_
     - 3.10–3.13
     - ``python>=3.10,<=3.13``
     - 
     - `PSF-2.0 <https://docs.python.org/3/license.html>`_

   * - `Numpy <https://github.com/numpy/numpy>`_
     - 1.26–2.4
     - ``numpy>=1.26,<=2.4``
     - 
     - `BSD-3-Clause <https://numpy.org/doc/stable/license.html>`_

   * - `IPP <https://www.intel.com/content/www/us/en/developer/tools/oneapi/ipp.html#gs.gxwq5p>`_
     - 2021.12
     - ``-c ccpi ipp=2021.12``
     - Intel Integrated Performance Primitives Library (required for CIL recon class)
     - `ISSL <http://www.intel.com/content/www/us/en/developer/articles/license/end-user-license-agreement.html>`_


Optional Dependencies
=====================
.. list-table::
   :header-rows: 1
   :widths: 18 10 22 30 15

   * - Package
     - Tested Version
     - Conda install command
     - Description
     - License

   * - `ASTRA toolbox <http://www.astra-toolbox.com>`_
     - 2.0–2.4
     - ``astra-toolbox::astra-toolbox=2.4``
     - CT projectors, FBP and FDK
     - `GPL-3.0 <https://github.com/astra-toolbox/astra-toolbox/blob/master/COPYING>`_

   * - `TIGRE <https://github.com/CERN/TIGRE>`_
     - 3.1.3
     - ``ccpi::tigre=3.1.3``
     - CT projectors, FBP and FDK
     - `BSD-3-Clause <https://github.com/CERN/TIGRE/blob/master/LICENSE.txt>`_

   * - `CCPi Regularisation Toolkit <https://github.com/TomographicImaging/CCPi-Regularisation-Toolkit>`_
     - 26.0.0
     - CPU: ``ccpi::ccpi-regulariser=26.0.0=cpu*``  
       GPU: ``ccpi::ccpi-regulariser=26.0.0=cuda*``
     - Toolbox of regularisation methods
     - `Apache-2.0 <https://github.com/TomographicImaging/CCPi-Regularisation-Toolkit/blob/master/LICENSE>`_

   * - `TomoPhantom <https://github.com/dkazanc/TomoPhantom>`_
     - 3.1.4
     - ``httomo::tomophantom=3.1.4``
     - Generates phantoms for test data
     - `Apache-2.0 <https://github.com/dkazanc/TomoPhantom/blob/master/LICENSE>`_

   * - `ipykernel <https://github.com/ipython/ipykernel>`_
     - 
     - ``ipykernel``
     - Provides the IPython kernel for Jupyter notebooks
     - `BSD-3-Clause <https://github.com/ipython/ipykernel/blob/main/LICENSE>`_

   * - `ipywidgets <https://github.com/jupyter-widgets/ipywidgets>`_
     - 
     - ``ipywidgets``
     - Enables visualisation tools in Jupyter notebooks
     - `BSD-3-Clause <https://github.com/jupyter-widgets/ipywidgets/blob/main/LICENSE>`_

   * - `zenodo_get <https://github.com/dvolgyes/zenodo_get>`_
     - >= 1.6
     - ``zenodo_get>=1.6``
     - Downloads datasets from Zenodo; used by ``dataexample`` in CIL-Demos
     - `AGPL-3.0 <https://github.com/dvolgyes/zenodo_get?tab=AGPL-3.0-1-ov-file>`_

CT Data Readers
================

.. list-table::
   :header-rows: 1
   :widths: 18 10 22 30 15

   * - Package
     - Tested Version
     - Conda install command
     - Description
     - License

   * - `olefile <https://github.com/decalage2/olefile>`_
     - >= 0.46
     - ``olefile>=0.46``
     - Processes Microsoft OLE2 files; used to read ZEISS data
     - `BSD-style <https://github.com/decalage2/olefile?tab=License-1-ov-file>`_

   * - `dxchange <https://github.com/data-exchange/dxchange>`_
     - >= 0.2.1
     - ``dxchange>=0.2.1``
     - Interface with TomoPy for loading tomography data
     - `BSD-style <https://github.com/data-exchange/dxchange?tab=License-1-ov-file>`_

