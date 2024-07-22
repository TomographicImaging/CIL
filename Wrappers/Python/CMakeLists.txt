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
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import os; print(os.path.dirname(os.__file__))"
      OUTPUT_VARIABLE PYTHON_DEST
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(PYTHON_DEST "${PYTHON_DEST}/site-packages")
  endif()

  file(TO_CMAKE_PATH ${PYTHON_DEST} TMP)
  set (PYTHON_DEST ${TMP})
  unset (TMP)

  message(STATUS "Python wrappers will be installed in " ${PYTHON_DEST})

  message("CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")

  if(NOT DEFINED)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo")
  endif()

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
      set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/timestamp")
      # TODO: add to add_custom_command DEPENDS the list of python files of the project.
      # file(GLOB_RECURSE DEPS ${CMAKE_CURRENT_SOURCE_DIR}/cil/*.py )

      # as a hack I remove ${OUTPUT}. This should trigger the new build.
      file( REMOVE ${OUTPUT} )

      add_custom_target(pythonsetup ALL
          COMMAND ${CMAKE_COMMAND} -E env ${PYTHON_EXECUTABLE} -m pip install . -t ${PYTHON_DEST} --upgrade
          WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
          COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
          DEPENDS cilacc)

      # install the data if not a conda build. With conda the data is in the dependency package cil-data
      if (NOT CONDA_BUILD)
        install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data/ DESTINATION ${CMAKE_INSTALL_PREFIX}/share/cil)
      endif()

      add_custom_target(PythonWrapper ALL DEPENDS pythonsetup)

  endif()

endif()
