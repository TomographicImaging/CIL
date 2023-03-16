# -*- coding: utf-8 -*-
#  Copyright 2019 United Kingdom Research and Innovation
#  Copyright 2019 The University of Manchester
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

option (BUILD_PYTHON_WRAPPER "Build Python Wrapper" ON)

if (BUILD_PYTHON_WRAPPER)
  find_package(PythonInterp REQUIRED)
  
  #set(PYTHON_DEST_DIR "" CACHE PATH "Directory of the Python wrappers")
  if (PYTHON_DEST_DIR)
    set(PYTHON_DEST "${PYTHON_DEST_DIR}")
  else() 
    message(STATUS "python version ${PYTHON_DEST}")
    execute_process(COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/CMake/sp_dir.py 
      OUTPUT_VARIABLE SP_DIR)
    # remove trailing newline  
    string(REGEX REPLACE "\n$" "" PYTHON_DEST ${SP_DIR})
    # append default site-packages
    set(PYTHON_DEST "${PYTHON_DEST}/site-packages")
  endif()

  file(TO_CMAKE_PATH ${PYTHON_DEST} TMP)
  set (PYTHON_DEST ${TMP})
  unset (TMP)

  message(STATUS "Python wrappers will be installed in " ${PYTHON_DEST})

  message("CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")

  set(CMAKE_BUILD_TYPE "Release")
  
  find_package(PythonLibs)
  if (PYTHONINTERP_FOUND)
    message(STATUS "Found PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}")
    message(STATUS "Python version ${PYTHON_VERSION_STRING}")
  endif()
  if (PYTHONLIBS_FOUND)
    message(STATUS "Found PYTHON_INCLUDE_DIRS=${PYTHON_INCLUDE_DIRS}")
    message(STATUS "Found PYTHON_LIBRARIES=${PYTHON_LIBRARIES}")
  endif()

  if (PYTHONINTERP_FOUND)
      message("Python found " ${PYTHON_EXECUTABLE})
      set(SETUP_PY    "${CMAKE_CURRENT_SOURCE_DIR}/setup.py")
      set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/timestamp")
      file(GLOB_RECURSE DEPS ${CMAKE_CURRENT_SOURCE_DIR}/cil/*.py )
      
      # adds the CIL_VERSION to the setup.py
      if (EXISTS ${SETUP_PY})
        file(REMOVE ${SETUP_PY})
      endif()
      configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CMake/setup.py.in ${SETUP_PY})
      # adds the version.py file to the source code with correct version string.
      if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cil/version.py)
        file(REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/cil/version.py)
      endif()
      configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CMake/version.py.in ${CMAKE_CURRENT_SOURCE_DIR}/cil/version.py)



      # add to add_custom_command DEPENDS the list of python files of the project.
      # as a hack I remove ${OUTPUT}. This should trigger the new build.
      file( REMOVE ${OUTPUT} )
      
      add_custom_target(pythonsetup ALL
          COMMAND ${CMAKE_COMMAND} -E env ${PYTHON_EXECUTABLE} -m pip install . -t ${PYTHON_DEST} --upgrade
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
          COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
          DEPENDS cilacc)
      
      # install the data if not a conda build. With conda the data is in the dependency package cil-data
      if (NOT CONDA_BUILD)
        install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data/ DESTINATION ${CMAKE_INSTALL_PREFIX}/share/cil)
      endif()
          
      
      add_custom_target(PythonWrapper ALL DEPENDS pythonsetup)
       
  endif()

endif()
