#include "liblossy_plugin.h"
liblossy_plugin::~liblossy_plugin()=default;
int liblossy_plugin::major_version() const { return 0; }
int liblossy_plugin::minor_version() const { return 0; }
int liblossy_plugin::patch_version() const { return 0; }
void liblossy_plugin::set_error(int code, std::string const& msg) {
  error.code = code;
  error.msg = msg;
}
const char* liblossy_plugin::error_msg() const {
  return error.msg.c_str();
}

int liblossy_plugin::error_code() const {
  return error.code;
}
