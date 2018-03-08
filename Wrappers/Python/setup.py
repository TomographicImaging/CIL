#!/usr/bin/env python
# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library developed by
#   Visual Analytics and Imaging System Group of the Science Technology
#   Facilities Council, STFC

#   Copyright 2018 Edoardo Pasca

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

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

cil_version=os.environ['CIL_VERSION']
if  cil_version == '':
    print("Please set the environmental variable CIL_VERSION")
    sys.exit(1)

setup(
    name="ccpi-common",
    version=cil_version,
    packages=['ccpi' , 'ccpi.reconstruction' , 'ccpi.astra'],

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    #install_requires=['docutils>=0.3'],

#    package_data={
#        # If any package contains *.txt or *.rst files, include them:
#        '': ['*.txt', '*.rst'],
#        # And include any *.msg files found in the 'hello' package, too:
#        'hello': ['*.msg'],
#    },
    # zip_safe = False,

    # metadata for upload to PyPI
    author="Edoardo Pasca",
    author_email="edoardo.pasca@stfc.ac.uk",
    description='CCPi Core Imaging Library - Python Framework Module',
    license="Apache v2.0",
    keywords="Python Framework",
    url="http://www.ccpi.ac.uk",   # project home page, if any

    # could also include long_description, download_url, classifiers, etc.
)
