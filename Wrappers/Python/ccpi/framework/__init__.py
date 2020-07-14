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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import sys
from datetime import timedelta, datetime
import warnings
from functools import reduce

from .framework import DataContainer
from .framework import ImageData, AcquisitionData
from .framework import ImageGeometry, AcquisitionGeometry, AcquisitionGeometrySubsetGenerator
from .framework import VectorData, VectorGeometry
from .framework import find_key, message
from .framework import DataProcessor
from .framework import AX, PixelByPixelDataProcessor, CastDataContainer
from .BlockDataContainer import BlockDataContainer
from .BlockGeometry import BlockGeometry
from .TestData import TestData
