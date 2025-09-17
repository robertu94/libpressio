#include "libpressio_ext/cpp/errorable.h"
namespace libpressio {
int pressio_errorable::set_error(int code, std::string const& msg) {
  error.msg = msg;
  return error.code = code;
}
void pressio_errorable::clear_error() {
  error.msg.clear();
  error.code = 0;
}
const char* pressio_errorable::error_msg() const {
  return error.msg.c_str();
}

int pressio_errorable::error_code() const {
  return error.code;
}
}
