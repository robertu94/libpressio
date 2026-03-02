cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(libpressio-build-tools VERSION "1.0.7" LANGUAGES CXX)

if(DEFINED ENV{CXXFLAGSHOST})
    set(CMAKE_CXX_FLAGS "$ENV{CXXFLAGSHOST}")
endif()

message("STATUS" "pressio_register_generator CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")

add_executable(pressio_register_generator ../../src/pressio_register_generator.cc)
target_compile_features(pressio_register_generator PRIVATE cxx_std_11)
