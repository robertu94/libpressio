#include "libpressio_ext/cpp/errorable.h"
int pressio_errorable::set_error(int code, std::string const& msg) {
  error.msg = msg;
  return error.code = code;
}
const char* pressio_errorable::error_msg() const {
  return error.msg.c_str();
}

int pressio_errorable::error_code() const {
  return error.code;
}
