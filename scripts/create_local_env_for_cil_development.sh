#  -*- coding: utf-8 -*-
# Copyright 2020 United Kingdom Research and Innovation
# Copyright 2020 The University of Manchester
    
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
    
#     http://www.apache.org/licenses/LICENSE-2.0
    
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
    
# Authors:
# CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt
#! /bin/bash

while getopts hn:p:e: option
 do
 case "${option}"
 in
  n) numpy=${OPTARG};;
  p) python=${OPTARG};;
  e) name=${OPTARG};;
  h)
   echo "Usage: $0 [-n numpy_version] [-p python version] [-e environment name]"
   exit 
   ;;
  *)
   echo "Wrong option passed. Use the -h option to get some help." >&2
   exit 1
  ;;
 esac
done

echo Numpy $numpy
echo Python $python
echo Environment name $name

set -x

conda create --name $name cmake python=$python numpy=$numpy scipy matplotlib \
  h5py pillow libgcc-ng dxchange olefile pywavelets python-wget scikit-image \
  packaging  numba ipp ipp-devel ipp-include tqdm -c conda-forge -c intel  -c defaults --override-channels
