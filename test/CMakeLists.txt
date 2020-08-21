configure_file(GTestCMakeLists.txt.in googletest-download/CMakeLists.txt)

execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download
)
if(result)
  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()

execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download
)
if(result)
  message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
  ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
  EXCLUDE_FROM_ALL)
include(GoogleTest)

function(add_gtest)
  get_filename_component(test_name ${ARGV0} NAME_WE)
  add_executable(${test_name} ${ARGV})
  target_link_libraries(${test_name} PRIVATE libpressio gtest_main gtest gmock)
  gtest_discover_tests(${test_name})
endfunction()

add_gtest(test_pressio_data.cc)
target_include_directories(test_pressio_data PRIVATE ${CMAKE_SOURCE_DIR}/src)
add_gtest(test_pressio_options.cc)
add_gtest(test_io.cc)

if(LIBPRESSIO_HAS_HDF)
  add_gtest(test_hdf5.cc)
endif()

if(LIBPRESSIO_HAS_SZ)
  add_gtest(test_sz_plugin.cc)
  target_include_directories(test_sz_plugin PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/../include)
  target_link_libraries(test_sz_plugin PRIVATE SZ)

  add_executable(sz_basic sz_basic.c make_input_data.cc)
  target_link_libraries(sz_basic PRIVATE libpressio SZ)
  add_test(sz_basic_test sz_basic)

  add_executable(sz_cpp sz_cpp.cc make_input_data.cc)
  target_link_libraries(sz_cpp PRIVATE libpressio  SZ)
  add_test(sz_cpp_test sz_cpp)

  add_executable(metrics metrics.c make_input_data.cc)
  target_link_libraries(metrics PRIVATE libpressio  SZ)
  add_test(metrics_test metrics)

  add_executable(test_external_helper test_external_helper.cc)
  add_executable(test_external test_external.cc make_input_data.cc)
  target_link_libraries(test_external PRIVATE libpressio SZ)
  target_link_libraries(test_external_helper PRIVATE libpressio SZ)
  add_test(test_external test_external ${CMAKE_CURRENT_BINARY_DIR}/test_external_helper)

  add_gtest(test_register_metrics.cc make_input_data.cc)
  target_link_libraries(test_register_metrics PRIVATE SZ)
  add_gtest(test_metrics_options.cc make_input_data.cc)
  target_link_libraries(test_metrics_options PRIVATE SZ)
  add_gtest(test_register_compressor.cc make_input_data.cc)
  target_link_libraries(test_register_compressor PRIVATE SZ)
endif()


if(LIBPRESSIO_HAS_ZFP)
  add_executable(zfp_basic zfp_basic.c make_input_data.cc)
  target_link_libraries(zfp_basic PRIVATE libpressio zfp::zfp)
  add_test(zfp_basic_test zfp_basic)
endif()

if(LIBPRESSIO_HAS_MAGICK)
  add_executable(magick_basic magick_basic.cc)
  target_link_libraries(magick_basic PRIVATE libpressio)
  add_test(magick_basic_test magick_basic)
endif()

if(LIBPRESSIO_HAS_SZ AND LIBPRESSIO_HAS_ZFP AND LIBPRESSIO_HAS_MGARD AND
    LIBPRESSIO_HAS_MAGICK AND LIBPRESSIO_HAS_FPZIP)
  add_executable(full full.c make_input_data.cc)
  target_link_libraries(full PRIVATE libpressio SZ)
  add_test(full_test full)
endif()

if(LIBPRESSIO_HAS_LIBDISTRIBUTED)
  find_package(MPI)
  add_gtest(test_distributed.cc)
  target_link_libraries(test_distributed PRIVATE MPI::MPI_CXX)
endif()

find_package(Python3)
if(BUILD_PYTHON_WRAPPER AND LIBPRESSIO_HAS_SZ)
  add_test(test_python_wrapper 
    ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_python.py
    ${CMAKE_BINARY_DIR}/swig
  )
endif()
