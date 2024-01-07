find_package(Clang REQUIRED)
find_package(fmt REQUIRED)
list(APPEND CMAKE_MODULE_PATH "${CLANG_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)
include(AddClang)
set( LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Core
  Option
  Support
  )


add_clang_library(libpressio_modernize SHARED
  ./src/libpressio_modernize.cc
  ./src/libpressio_highlevel.cc
  ./src/libpressio_invalidations.cc
  ./include/libpressio_modernize.h
  )
clang_target_link_libraries(libpressio_modernize
  PRIVATE
  clangAnalysis
  clangAST
  clangASTMatchers
  clangBasic
  clangFormat
  clangFrontend
  clangLex
  clangRewrite
  clangSema
  clangSerialization
  clangTooling
  clangToolingCore
  clangTidy
  clangTidyModule
  )
target_link_libraries(libpressio_modernize PRIVATE
    fmt::fmt
    )
target_include_directories(libpressio_modernize PUBLIC 
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
