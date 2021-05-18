#include "libpressio_ext/cpp/configurable.h"

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

int pressio_configurable::check_options(struct pressio_options const&) {
  return 0;
}
