#include "libpressio_ext/cpp/configurable.h"

namespace libpressio {
struct pressio_options pressio_configurable::get_documentation() const {
  return {};
}

struct pressio_options pressio_configurable::get_configuration() const {
  return {};
}
struct pressio_options pressio_configurable::get_options() const {
  return {};
}

int pressio_configurable::set_options(struct pressio_options const&) {
  return 0;
}

std::vector<std::string> pressio_configurable::children() const { return {}; }

int pressio_configurable::check_options(struct pressio_options const&) {
  return 0;
}
}
