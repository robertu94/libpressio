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
  target_link_libraries(${test_name} libpressio gtest_main gtest gmock)
  gtest_discover_tests(${test_name})
endfunction()

add_gtest(test_pressio_data.cc)
add_gtest(test_pressio_options.cc)
add_gtest(test_sz_plugin.cc)
target_include_directories(test_sz_plugin PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/../include)

add_executable(sz_basic sz_basic.c)
target_link_libraries(sz_basic libpressio)
add_test(sz_basic_test sz_basic)

add_executable(zfp_basic zfp_basic.c)
target_link_libraries(zfp_basic libpressio)
add_test(zfp_basic_test zfp_basic)

add_executable(full full.c)
target_link_libraries(full libpressio)
add_test(full_test full)
