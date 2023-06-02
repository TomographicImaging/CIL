#  -*- coding: utf-8 -*-
# Copyright 2019 United Kingdom Research and Innovation
# Copyright 2019 The University of Manchester
    
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

if [ `python -c "from __future__ import print_function; import platform; print (platform.system())"`  == "Darwin" ] ;
then 
  echo "Darwin"; 
  cmake ${RECIPE_DIR}/../ -DCONDA_BUILD=ON \
                        -DCMAKE_BUILD_TYPE="Release"\
                        -DLIBRARY_LIB=$CONDA_PREFIX/lib \
                        -DLIBRARY_INC=$CONDA_PREFIX \
                        -DCMAKE_INSTALL_PREFIX=$PREFIX\
                        -DOPENMP_INCLUDES=${CONDA_PREFIX}/include \
                        -DOPENMP_LIBRARIES=${CONDA_PREFIX}/lib
else 
  echo "something else"; 

  cmake ${RECIPE_DIR}/../ -DCONDA_BUILD=ON \
                        -DCMAKE_BUILD_TYPE="Release"\
                        -DLIBRARY_LIB=$CONDA_PREFIX/lib \
                        -DLIBRARY_INC=$CONDA_PREFIX \
                        -DCMAKE_INSTALL_PREFIX=$PREFIX

fi

cmake --build . --target install
