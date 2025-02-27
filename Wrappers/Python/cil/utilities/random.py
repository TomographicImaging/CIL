#  Copyright 2025 United Kingdom Research and Innovation
#  Copyright 2025 The University of Manchester
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Authors:
# CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt

import numpy

class RandomGenerator:
    def __init__(self, seed=None):
        self._seed = seed
        self._rng = numpy.random.Generator(numpy.random.PCG64DXSM(seed))

    def set_seed(self, seed):
        self._seed = seed
        self._rng = numpy.random.Generator(numpy.random.PCG64DXSM(seed))

    def __getattr__(self, name):
        return getattr(self._rng, name)

global_rng = RandomGenerator(seed=None)