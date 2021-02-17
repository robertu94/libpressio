add_library(libpressio_hdf5_filter
  ./src/libpressio_hdf5_filter.cc
  ./src/libpressio_hdf5_props.cc
  ./include/libpressio_hdf5_filter.h
  )
target_link_libraries(libpressio_hdf5_filter PRIVATE
  libpressio
  hdf5::hdf5
  nlohmann_json::nlohmann_json
  MPI::MPI_CXX
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

export(TARGETS libpressio_hdf5_filter NAMESPACE libpressio_hdf5_filter:: FILE libpressio_hdf5_filter.cmake)
install(TARGETS libpressio_hdf5_filter EXPORT libpressio_hdf5_filter
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
install(EXPORT libpressio_hdf5_filter NAMESPACE libpressio_hdf5_filter:: DESTINATION share/libpressio_hdf5_filter/cmake)


