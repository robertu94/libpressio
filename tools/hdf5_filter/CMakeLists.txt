add_library(libpressio_hdf5_filter
  ./src/libpressio_hdf5_filter.cc
  ./src/libpressio_hdf5_props.cc
  ./include/libpressio_hdf5_filter.h
  ./src/libpressio_hdf5_filter_impl.h
  )
add_executable(pressio_hdf5_filter_tool
  ./src/libpressio_hdf5_filter_tool.cc
  ./src/libpressio_hdf5_filter.cc
  ./src/libpressio_hdf5_props.cc
  ./include/libpressio_hdf5_filter.h
  ./src/libpressio_hdf5_filter_impl.h
  )
function(lp_add_hdf_deps name)
  target_link_libraries(${name} PUBLIC ${HDF5_C_LIBRARIES})
  target_include_directories(${name} PUBLIC ${HDF5_C_INCLUDE_DIRS})
  target_include_directories(${name} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)
  target_compile_definitions(${name} PUBLIC ${HDF5_C_DEFINITIONS})
  if(${HDF5_IS_PARALLEL})
    find_package(MPI REQUIRED)
    target_link_libraries(${name} PUBLIC MPI::MPI_CXX)
  endif()
  target_link_libraries(${name} PRIVATE
    libpressio
    nlohmann_json::nlohmann_json
    )
  target_include_directories(${name} 
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/libpressio_hdf5>
    )
  target_compile_options(${name} PRIVATE 
    $<$<CONFIG:Debug>: -Wall -Wextra -Wpedantic>
    $<$<CONFIG:RelWithDebInfo>: ${NO_OMIT_FRAME_POINTER_FLAG}>
    )
endfunction()
lp_add_hdf_deps(libpressio_hdf5_filter)
lp_add_hdf_deps(pressio_hdf5_filter_tool)

install(FILES ./include/libpressio_hdf5_filter.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libpressio)
install(TARGETS libpressio_hdf5_filter EXPORT LibPressioTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/hdf5/lib/plugin
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/hdf5/lib/plugin
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
install(TARGETS pressio_hdf5_filter_tool EXPORT LibPressioTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
