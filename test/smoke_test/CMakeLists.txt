cmake_minimum_required(VERSION 3.18)
project(LibPressioSmokeTests)

find_package(LibPressio REQUIRED)
find_package(std_compat REQUIRED)
add_executable(pressio_smoke_tests smoke_test.cc)
target_link_libraries(pressio_smoke_tests LibPressio::libpressio)
install(TARGETS pressio_smoke_tests)
