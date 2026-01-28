#  Copyright 2026 United Kingdom Research and Innovation
#  Copyright 2026 The University of Manchester
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
# Jeppe Klitgaard (DTU)

import importlib.metadata
from packaging.version import Version
version = importlib.metadata.version("cil")

__v = Version(version)
major, minor, patch = __v.major, __v.minor, __v.micro
commit_hash = __v.local.split(".", 1)[0]
num_commit = __v.dev
