option (BUILD_PYTHON_WRAPPER "Build Python Wrapper" ON)

if (BUILD_PYTHON_WRAPPER)
  find_package(PythonInterp REQUIRED)
  
  set(PYTHON_DEST_DIR "" CACHE PATH "Directory of the Python wrappers")
  if (PYTHON_DEST_DIR)
    set(PYTHON_DEST "${PYTHON_DEST_DIR}")
  else() 
    message(STATUS "python version ${PYTHON_DEST}")
    execute_process(COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/sp_dir.py 
      OUTPUT_VARIABLE SP_DIR)
    # remove trailing newline  
    string(REGEX REPLACE "\n$" "" PYTHON_DEST ${SP_DIR})
    # append default site-packages
    set(PYTHON_DEST "${PYTHON_DEST}/site-packages")
  endif()
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
      set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup-cil.py.in")
      set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
      #set(DEPS        "${CMAKE_CURRENT_SOURCE_DIR}/module/__init__.py")
      set (DEPS       "${CMAKE_BINARY_DIR}/")
      set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/build/timestamp")

      #configure_file(${SETUP_PY_IN} ${SETUP_PY})

      
      if (CONDA_BUILD)
        add_custom_command(OUTPUT ${OUTPUT}
                        #COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}
                        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}
                        COMMAND ${CMAKE_COMMAND} -E env CIL_VERSION=${CIL_VERSION}
                                                        ${PYTHON_EXECUTABLE} ${SETUP_PY} -vv install
                                                        #echo "EDO"
                        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                        COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
                        DEPENDS cilacc)

      else()
        if (WIN32)
          add_custom_command(OUTPUT ${OUTPUT}
                        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}
                        COMMAND ${CMAKE_COMMAND} -E env CIL_VERSION=${CIL_VERSION}
                                                        PREFIX=${CMAKE_SOURCE_DIR}/src/
                                                        LIBRARY_INC=${CMAKE_SOURCE_DIR}/src/include
                                                        LIBRARY_LIB=${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}
                                                        ${PYTHON_EXECUTABLE} ${SETUP_PY} build_py
                        COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
                        DEPENDS cilacc)
        else()
          add_custom_command(OUTPUT ${OUTPUT}
                        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}
                        COMMAND ${CMAKE_COMMAND} -E env PREFIX=${CMAKE_SOURCE_DIR}/src/
                                                        LIBRARY_INC=${CMAKE_SOURCE_DIR}/src/include
                                                        LIBRARY_LIB=${CMAKE_BINARY_DIR}/
                                                        ${PYTHON_EXECUTABLE} ${SETUP_PY} build_py
                        COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
                        DEPENDS cilacc
                        )
        endif()
        #set (PYTHON_DEST ${CMAKE_INSTALL_PREFIX}/python/)

        install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/build/lib/ccpi

                DESTINATION ${PYTHON_DEST})
        install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data/ DESTINATION ${CMAKE_INSTALL_PREFIX}/share/ccpi)
        #file(TOUCH ${PYTHON_DEST}/edo/__init__.py)
        
      endif()
      
      
      add_custom_target(PythonWrapper ALL DEPENDS ${OUTPUT})
      
  endif()

endif()
