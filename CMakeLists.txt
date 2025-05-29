#  Copyright 2020 United Kingdom Research and Innovation
#  Copyright 2020 The University of Manchester
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
if(APPLE)
  cmake_minimum_required(VERSION 3.16)
else()
  cmake_minimum_required(VERSION 3.5)
endif()

if(NOT SKBUILD_PROJECT_NAME)
  set(SKBUILD_PROJECT_NAME cil)
endif()
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

cmake_policy(PUSH)

cmake_policy(SET CMP0074 NEW)
if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.27)
  cmake_policy(SET CMP0148 OLD)
endif()

# If APPLE don't use @rpath
if (APPLE)
  option(SHARED_LIBS_ABS_PATH "Force shared libraries to be installed with absolute paths (as opposed to rpaths)" ON)
  mark_as_advanced( SHARED_LIBS_ABS_PATH )
  if (SHARED_LIBS_ABS_PATH)
    # Set install_name_dir as the absolute path to install_prefix/lib
    GET_FILENAME_COMPONENT(CMAKE_INSTALL_NAME_DIR ${CMAKE_INSTALL_PREFIX}/lib REALPATH)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  endif(SHARED_LIBS_ABS_PATH)
endif(APPLE)

if(NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

add_subdirectory(src/Core)

cmake_policy(POP)
