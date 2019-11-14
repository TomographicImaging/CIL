
set (CMAKE_C_STANDARD 11)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
   set (CMAKE_SHARED_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_SHARED_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
   set (CMAKE_STATIC_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_STATIC_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
   if (UNIX)
    set (FLAGS "-O3 -funsigned-char -Wall  -Wl,--no-undefined -march=native")  
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAGS}")
    set (CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} ${FLAGS}")
  

    set (EXTRA_LIBRARIES 
      "gomp"
      "m"
      )
   endif()
endif()

if (WIN32)
	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /Ddll_EXPORTS")
endif()


message("CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")
message("CMAKE_C_FLAGS ${CMAKE_C_FLAGS}")
message("CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS}")
message("CMAKE_SHARED_LINKER_FLAGS ${CMAKE_SHARED_LINKER_FLAGS}")
message("CMAKE_STATIC_LINKER_FLAGS ${CMAKE_STATIC_LINKER_FLAGS}")



add_library(cilacc SHARED ${CMAKE_CURRENT_SOURCE_DIR}/axpby.c)

target_link_libraries(cilacc ${EXTRA_LIBRARIES} )
include_directories(cilacc PUBLIC 
				  ${CMAKE_CURRENT_SOURCE_DIR}/include
		          )

## Install
include(GNUInstallDirs)
install(TARGETS cilacc
	RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
	CONFIGURATIONS ${CMAKE_BUILD_TYPE} 
	)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/src/Core/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}) 


