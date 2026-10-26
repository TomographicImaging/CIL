#  Copyright 2018 United Kingdom Research and Innovation
#  Copyright 2018 The University of Manchester
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
# Joshua DM Hellier (University of Manchester) [refactorer]
import ctypes
from importlib.metadata import distribution

try:
    dist = distribution("cil").locate_file("cil/lib")
    cilacc_path = next(dist.glob("*cilacc.*"))
except StopIteration:
    raise FileNotFoundError("cilacc library not found")
cilacc = ctypes.cdll.LoadLibrary(str(cilacc_path))
