#include <sstream>
#include <algorithm>
#include <libpressio_ext/cpp/pressio.h>
#include <libpressio_ext/cpp/options.h>
#include <libpressio_ext/cpp/io.h>

pressio_registry<std::unique_ptr<libpressio_io_plugin>>& io_plugins() {
  static pressio_registry<std::unique_ptr<libpressio_io_plugin>> registry;
  return registry;
}

extern "C" {
struct pressio_io* pressio_get_io(struct pressio* library, const char* io_module) {
  auto plugin = library->get_io(io_module);
  if(plugin) return new pressio_io(std::move(plugin));
  else return nullptr;
}
void pressio_io_free(struct pressio_io* io) {
  delete io;
}
const char* pressio_supported_io_modules() {
  return pressio::supported_io();
}

struct pressio_options* pressio_io_get_configuration(struct pressio_io const* io) {
  return new pressio_options((*io)->get_configuration());
}
struct pressio_options* pressio_io_get_documentation(struct pressio_io const* io) {
  return new pressio_options((*io)->get_documentation());
}
struct pressio_options* pressio_io_get_options(struct pressio_io const* io) {
  return new pressio_options((*io)->get_options());
}
int pressio_io_set_options(struct pressio_io* io, struct pressio_options const * options) {
  return (*io)->set_options(*options);
}
int pressio_io_check_options(struct pressio_io* io, struct pressio_options const * options) {
  return (*io)->check_options(*options);
}
struct pressio_data* pressio_io_read(struct pressio_io* io, struct pressio_data* data) {
  return (*io)->read(data);
}
int pressio_io_write(struct pressio_io* io, struct pressio_data const* data) {
  return (*io)->write(data);
}
int pressio_io_error_code(struct pressio_io const* io) {
  return (*io)->error_code();
}
const char* pressio_io_error_msg(struct pressio_io const* io) {
  return (*io)->error_msg();
}
const char* pressio_io_version(struct pressio_io const* io) {
  return (*io)->version();
}
int pressio_io_major_version(struct pressio_io const* io) {
  return (*io)->major_version();
}
int pressio_io_minor_version(struct pressio_io const* io) {
  return (*io)->minor_version();
}
int pressio_io_patch_version(struct pressio_io const* io) {
  return (*io)->patch_version();
}
struct pressio_io* pressio_io_clone(struct pressio_io* io) {
  return new pressio_io((*io)->clone());
}

void pressio_io_set_name(struct pressio_io* io, const char* new_name) {
  (*io)->set_name(new_name);
}


const char* pressio_io_get_name(struct pressio_io const* io) {
  return (*io)->get_name().c_str();
}

int pressio_io_write_many(struct pressio_io* io, const struct pressio_data** data_begin, size_t num_data) {
  return (*io)->write_many(data_begin, data_begin+num_data);
}

int pressio_io_read_many(struct pressio_io* io, struct pressio_data** data_begin, size_t num_data) {
  return (*io)->read_many(data_begin, data_begin+num_data);
}

}

struct pressio_data* libpressio_io_plugin::read(struct pressio_data* data) {
  clear_error();
  return read_impl(data);
}
int libpressio_io_plugin::write(struct pressio_data const* data) {
  clear_error();
  return write_impl(data);
}
int libpressio_io_plugin::check_options(struct pressio_options const& options) {
  clear_error();
  return check_options_impl(options);
}
int libpressio_io_plugin::set_options(struct pressio_options const& options) {
  clear_error();
  return set_options_impl(options);
}
struct pressio_options libpressio_io_plugin::get_documentation() const {
  pressio_options opts;
  set(opts, "pressio:thread_safe", "level of thread safety provided by the compressor");
  set(opts, "pressio:stability", "level of stablity provided by the compressor; see the README for libpressio");
  opts.copy_from(get_documentation_impl());
  return opts;
}
struct pressio_options libpressio_io_plugin::get_configuration() const {
  return get_configuration_impl();
}
struct pressio_options libpressio_io_plugin::get_options() const {
  return get_options_impl();
}
int libpressio_io_plugin::check_options_impl(struct pressio_options const&) {
  return 0;
}
void libpressio_io_plugin::set_name(std::string const& new_name) {
  pressio_configurable::set_name(new_name);
}
