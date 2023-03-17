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
# Richard Brown (University College London)

if(APPLE)
  cmake_minimum_required(VERSION 3.16)
else()
  cmake_minimum_required(VERSION 3.4)
endif()

project (cil LANGUAGES CXX)

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

set(CMAKE_BUILD_TYPE "Release")


# CIL version TAG is determined by git describe as for the conda package
# variable GIT_DESCRIBE_TAG 
# https://docs.conda.io/projects/conda-build/en/stable/resources/define-metadata.html#templating-with-jinja 
# https://stackoverflow.com/questions/38919840/get-package-version-for-conda-meta-yaml-from-source-file

find_package(Git REQUIRED)

find_package(Python COMPONENTS Interpreter REQUIRED)

if (Python_FOUND)
  execute_process(COMMAND "${Python_EXECUTABLE}" ${CMAKE_CURRENT_SOURCE_DIR}/Wrappers/Python/CMake/parse_git_describe.py "${GIT_EXECUTABLE}"
                RESULT_VARIABLE worked 
                OUTPUT_VARIABLE CIL_VERSION
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                OUTPUT_STRIP_TRAILING_WHITESPACE)
  if (NOT worked EQUAL 0)
    message(FATAL_ERROR "git describe returned ${worked}")
  else()
    string(REPLACE " " ";" CIL_VERSION ${CIL_VERSION})
    list(LENGTH CIL_VERSION CIL_version_len)
    
    list(GET CIL_VERSION 0 CIL_VERSION_MAJOR)
    list(GET CIL_VERSION 1 CIL_VERSION_MINOR)
    list(GET CIL_VERSION 2 CIL_VERSION_PATCH)
    list(GET CIL_VERSION 3 CIL_NUM_COMMIT)
    list(GET CIL_VERSION 4 CIL_COMMIT_HASH)

  endif()
endif()


add_subdirectory(src/Core)
add_subdirectory(Wrappers/Python)
