find_package(PythonInterp REQUIRED)

if(PYTHON_DEST_DIR)
  file(TO_CMAKE_PATH "${PYTHON_DEST_DIR}" PYTHON_DEST)
endif()

message("CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")
if(NOT DEFINED)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

if (PYTHONINTERP_FOUND)
  # TODO: add to add_custom_command DEPENDS the list of python files of the project.
  # file(GLOB_RECURSE DEPS ${CMAKE_CURRENT_SOURCE_DIR}/cil/*.py )
  message("SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CIL: $ENV{SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CIL}")
  message("CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
  if(PYTHON_DEST_DIR)
    message(STATUS "CIL install using ${PYTHON_EXECUTABLE} -m pip install -t ${PYTHON_DEST}")
    add_custom_target(pythonsetup ALL
      COMMAND ${PYTHON_EXECUTABLE} -m pip install ${CMAKE_SOURCE_DIR} -t ${PYTHON_DEST} --upgrade
      DEPENDS cilacc)
  else()
    message(STATUS "CIL install using ${PYTHON_EXECUTABLE} -m pip install")
    add_custom_target(pythonsetup ALL
      COMMAND ${PYTHON_EXECUTABLE} -m pip install ${CMAKE_SOURCE_DIR} --upgrade
      DEPENDS cilacc)
  endif()
  # install the data if not a conda build. With conda the data is in the dependency package cil-data
  if (NOT CONDA_BUILD)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data/ DESTINATION ${CMAKE_INSTALL_PREFIX}/share/cil)
  endif()
endif()
