
set (CMAKE_C_STANDARD 11)
set (CMAKE_BUILD_TYPE Release)

if(APPLE)
    if(NOT DEFINED OPENMP_INCLUDES OR NOT DEFINED OPENMP_LIBRARIES)
        set(OPENMP_INCLUDES "OPENMP_INCLUDES-NOTFOUND" CACHE PATH "Need to set OpenMP include dir on OSX")
        set(OPENMP_LIBRARIES "OPENMP_LIBRARIES-NOTFOUND" CACHE PATH "Need to set OpenMP lib dir on OSX")
    endif()
    if(NOT EXISTS ${OPENMP_INCLUDES} OR NOT EXISTS ${OPENMP_LIBRARIES})
        message(FATAL_ERROR "Need to set OPENMP_INCLUDES and OPENMP_LIBRARIES on OSX.")
    endif()
    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        #set(OpenMP_C "${CMAKE_C_COMPILER}")
        set(OpenMP_C_FLAGS "-Xclang -fopenmp -Wno-unused-command-line-argument -L${OPENMP_LIBRARIES}" CACHE STRING "C FLAGS")
        #set(OpenMP_C_LIB_NAMES "omp" "libomp" "libgomp" "libiomp5")
        set (OpenMP_C_LIBRARY "omp")
        set (OpenMP_C_LIB_NAMES ${OpenMP_C_LIBRARY} CACHE STRING "Name of the omp lib")
        set(OpenMP_omp_LIBRARY "omp")
        set(OpenMP_libomp_LIBRARY "libomp")
        #set(OpenMP_libgomp_LIBRARY "libgomp")
        #set(OpenMP_libiomp5_LIBRARY "libiomp5")
        set (OpenMP_C_LIBRARY_PATH ${OPENMP_LIBRARIES})
        set (OpenMP_LINKER_FLAGS "-L${OpenMP_C_LIBRARY_PATH}")
        
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
        set(OpenMP_CXX_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
        set(OpenMP_CXX_LIB_NAMES "libomp" "libgomp" "libiomp5")
        set(OpenMP_libomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
        set(OpenMP_libgomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
        set(OpenMP_libiomp5_LIBRARY ${OpenMP_CXX_LIB_NAMES})
    endif()
endif()

message ("OpenMP_C_FLAGS ${OpenMP_C_FLAGS}")
find_package(OpenMP REQUIRED)
add_definitions(${OpenMP_C_FLAGS})
#add_definitions(${OpenMP_CXX_FLAGS})

  
  if (APPLE)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    set (CMAKE_SHARED_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_SHARED_LINKER_FLAGS}")
    set (CMAKE_STATIC_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_STATIC_LINKER_FLAGS}")   
    include_directories(${OPENMP_INCLUDES})
    link_directories(${OPENMP_LIBRARIES})
  else()
    if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.9.0") 
      set (OpenMP_EXE_LINKER_FLAGS OpenMP::OpenMP_CXX)
      set (OpenMP_EXE_LINKER_FLAGS OpenMP::OpenMP_C)
    else()
      message(WARNING "Your CMake version is old. OpenMP linking flags  might be incorrect.")
      # need to explicitly set this. Definitely for gcc, hopefully also for other systems.
      # See https://gitlab.kitware.com/cmake/cmake/issues/15392
      set (OpenMP_EXE_LINKER_FLAGS ${OpenMP_CXX_FLAGS})
      set (OpenMP_EXE_LINKER_FLAGS ${OpenMP_C_FLAGS} ${OpenMP_LINKER_FLAGS})
    endif()
  endif()


#if (OPENMP_FOUND)
#    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
#   set (CMAKE_SHARED_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_SHARED_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
#   set (CMAKE_STATIC_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_STATIC_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
#   if (UNIX)
#    set (FLAGS "-O3 -funsigned-char -Wall  -Wl,--no-undefined -march=native")  
#    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAGS}")
#    set (CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} ${FLAGS}")
#  
#    set (EXTRA_LIBRARIES 
#      "gomp"
#      "m"
#      )
#   endif()
# endif()

if (WIN32)
	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /Ddll_EXPORTS")
endif()


message("CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")
message("CMAKE_C_FLAGS ${CMAKE_C_FLAGS}")
message("CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS}")
message("CMAKE_SHARED_LINKER_FLAGS ${CMAKE_SHARED_LINKER_FLAGS}")
message("CMAKE_STATIC_LINKER_FLAGS ${CMAKE_STATIC_LINKER_FLAGS}")



add_library(cilacc SHARED ${CMAKE_CURRENT_SOURCE_DIR}/utilities.c
                          ${CMAKE_CURRENT_SOURCE_DIR}/FiniteDifferenceLibrary.c
                          ${CMAKE_CURRENT_SOURCE_DIR}/axpby.c )

target_link_libraries(cilacc ${OpenMP_C_LIB_NAMES} )
include_directories(cilacc PUBLIC 
				  ${CMAKE_CURRENT_SOURCE_DIR}/include
		          )

## Install
#include(GNUInstallDirs)
#install(TARGETS cilacc
#	RUNTIME DESTINATION bin
#    LIBRARY DESTINATION lib
#    ARCHIVE DESTINATION lib
#	CONFIGURATIONS ${CMAKE_BUILD_TYPE} 
#	)

if (UNIX)
message ("I'd install into ${CMAKE_INSTALL_PREFIX}/lib")
install(TARGETS cilacc
	LIBRARY DESTINATION lib
	CONFIGURATIONS ${CMAKE_BUILD_TYPE} 
	)
elseif(WIN32)
message ("I'd install into ${CMAKE_INSTALL_PREFIX} lib bin")
  install(TARGETS cilacc 
	RUNTIME DESTINATION bin
	ARCHIVE DESTINATION lib
	CONFIGURATIONS ${CMAKE_BUILD_TYPE} 
	)
endif()

install(DIRECTORY ${PROJECT_SOURCE_DIR}/src/Core/include/
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include/cil) 


