#include <set>
#include <string>
#include <algorithm>
#include <iterator>
#include <sstream>
#include "libpressio_plugin.h"
#include "pressio_options_iter.h"
#include "pressio_options.h"

libpressio_plugin::~libpressio_plugin()=default;
int libpressio_plugin::major_version() const { return 0; }
int libpressio_plugin::minor_version() const { return 0; }
int libpressio_plugin::patch_version() const { return 0; }
int libpressio_plugin::set_error(int code, std::string const& msg) {
  error.msg = msg;
  return error.code = code;
}
const char* libpressio_plugin::error_msg() const {
  return error.msg.c_str();
}

int libpressio_plugin::error_code() const {
  return error.code;
}

std::set<std::string> get_keys(struct pressio_options const* options) {
  std::set<std::string> keys;
  struct pressio_options_iter* iter = pressio_options_get_iter(options);
  while(pressio_options_iter_has_value(iter))
  {
    const char* key = pressio_options_iter_get_key(iter);
    keys.emplace(key);
    pressio_options_iter_next(iter);
  }
  pressio_options_iter_free(iter);
  return keys;
}

int libpressio_plugin::check_options(struct pressio_options const* options) {
  struct pressio_options* my_options = get_options();
  auto my_keys = get_keys(my_options);
  auto keys = get_keys(options);
  std::set<std::string> extra_keys;
  std::set_difference(
      std::begin(keys), std::end(keys),
      std::begin(my_keys), std::end(my_keys),
      std::inserter(extra_keys, std::begin(extra_keys))
  );
  if(!extra_keys.empty()) {
    std::stringstream ss;
    ss << "extra keys: ";

    std::copy(std::begin(extra_keys), std::end(extra_keys), std::ostream_iterator<std::string>(ss, " "));
    set_error(1, ss.str());
  }
  pressio_options_free(my_options);
  return !extra_keys.empty();
}
