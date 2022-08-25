add_library(libpressio_hdf5_filter
  ./src/libpressio_hdf5_filter.cc
  ./src/libpressio_hdf5_props.cc
  ./include/libpressio_hdf5_filter.h
  )
target_link_libraries(libpressio_hdf5_filter PUBLIC ${HDF5_C_LIBRARIES})
target_include_directories(libpressio_hdf5_filter PUBLIC ${HDF5_C_INCLUDE_DIRS})
target_compile_definitions(libpressio_hdf5_filter PUBLIC ${HDF5_C_DEFINITIONS})
if(${HDF5_IS_PARALLEL})
  find_package(MPI REQUIRED)
  target_link_libraries(libpressio_hdf5_filter PUBLIC MPI::MPI_CXX)
endif()
target_link_libraries(libpressio_hdf5_filter PRIVATE
  libpressio
  nlohmann_json::nlohmann_json
  )
target_include_directories(libpressio_hdf5_filter 
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/libpressio_hdf5>
  )
target_compile_options(libpressio_hdf5_filter PRIVATE 
  $<$<CONFIG:Debug>: -Wall -Wextra -Wpedantic>
  $<$<CONFIG:RelWithDebInfo>: ${NO_OMIT_FRAME_POINTER_FLAG}>
  )

install(FILES ./include/libpressio_hdf5_filter.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libpressio)
install(TARGETS libpressio_hdf5_filter EXPORT LibPressioTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
