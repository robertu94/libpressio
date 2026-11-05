%module pressio_memory_test
%{
#include "python_memory_helpers.h"
#include <exception>
%}
%include <stdint.i>
%exception {
  try {
    $action
  } catch (std::exception const& error) {
    SWIG_exception_fail(SWIG_RuntimeError, error.what());
  }
}
%include "python_memory_helpers.h"
