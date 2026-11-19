execute_process(
  COMMAND "${CMAKE_BINARY_DIR}/tools/docs/generate_docs" -m c -o "$ENV{LIBPRESSIO_GENERATE_DOCS_OUTPUT}"
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
  RESULT_VARIABLE gen_result
)

if(NOT gen_result EQUAL 0)
  file(WRITE "$ENV{LIBPRESSIO_GENERATE_DOCS_OUTPUT}" "# Compressors Modules {#compressors}\n\nCompressor documentation generation failed on this platform during build.\n")
endif()
