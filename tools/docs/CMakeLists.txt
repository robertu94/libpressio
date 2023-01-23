add_executable(generate_docs
  generate_docs.cc
  )
target_link_libraries(generate_docs
  PRIVATE
  libpressio
  ${CMAKE_DL_LIBS}
  )

