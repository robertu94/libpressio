# Apple's leaks can return zero even when its own stack logger asserts.
# Preserve its report and fail explicitly on known tool failures.
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env MallocStackLogging=1
          /usr/bin/leaks --atExit -- "${PYTHON_EXECUTABLE}"
          "${TEST_SCRIPT}" "${BINDINGS_DIR}"
  RESULT_VARIABLE status OUTPUT_VARIABLE report ERROR_VARIABLE errors)
message("${report}${errors}")
if(NOT "${status}" STREQUAL "0")
  message(FATAL_ERROR "Python leaks check failed (status ${status})")
endif()
if("${report}${errors}" MATCHES "Assertion failed:|Couldn't get task port|\\[fatal\\]")
  message(FATAL_ERROR "Apple leaks failed internally; the leak check is inconclusive")
endif()
