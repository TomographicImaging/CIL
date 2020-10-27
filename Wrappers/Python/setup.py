# -*- coding: utf-8 -*-
#  CCP in Tomographic Imaging (CCPi) Core Imaging Library (CIL).

#   Copyright 2017 UKRI-STFC
#   Copyright 2017 University of Manchester

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from distutils.core import setup
import os
import sys


cil_version=os.environ['CIL_VERSION']
if  cil_version == '':
    print("Please set the environmental variable CIL_VERSION")
    sys.exit(1)

setup(
    name="cil",
    version=cil_version,
    packages=['cil' , 'cil.io',
              'cil.framework', 'cil.optimisation',
              'cil.optimisation.functions',
              'cil.optimisation.algorithms',
              'cil.optimisation.operators',
              'cil.processors',
              'cil.utilities', 'cil.utilities.jupyter'],
    data_files = [('share/cil', ['data/boat.tiff',
                                  'data/peppers.tiff',
                                  'data/camera.png',
                                  'data/resolution_chart.tiff',
                                  'data/shapes.png',
                                  'data/24737_fd_normalised.nxs'])],


    # metadata for upload to PyPI
    author="CCPi developers",
    maintainer="Edoardo Pasca",
    maintainer_email="edoardo.pasca@stfc.ac.uk",
    description='CCPi Core Imaging Library - Python Framework Module',
    license="Apache v2.0",
    keywords="Python Framework",
    url="http://www.ccpi.ac.uk/cil",
)
