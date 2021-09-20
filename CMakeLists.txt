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


execute_process(COMMAND git describe 
                RESULT_VARIABLE worked 
                OUTPUT_VARIABLE CIL_VERSION
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                OUTPUT_STRIP_TRAILING_WHITESPACE)
if (NOT worked EQUAL 0)
  message(FATAL_ERROR "git describe returned ${worked}")
endif()

add_subdirectory(src/Core)
add_subdirectory(Wrappers/Python)
