cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(libpressio-build-tools VERSION "1.0.7" LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "-isysroot /nix/store/5gfsv5n8zhpnl9yhggjpxrxg0jyflwja-apple-sdk-11.3/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk -isystem /nix/store/rcaj8r2chyqccxgzf62d8j3iyyjxgs2x-libcxx-19.1.7-dev/include/c++/v1/ -isystem /nix/store/v5a0gqs9r5v406xma832691y8k6srn1i-clang-20.1.8-lib/lib/clang/20/include/")

add_executable(pressio_register_generator ../../src/pressio_register_generator.cc)
target_compile_features(pressio_register_generator PRIVATE cxx_std_11)
