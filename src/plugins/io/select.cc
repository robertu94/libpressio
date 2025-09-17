#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/io.h"
#include "libpressio_ext/cpp/domain_manager.h"
#include "pressio_compressor.h"
#include "std_compat/memory.h"

namespace libpressio { namespace io { namespace select_ns {
struct select_io: public libpressio_io_plugin {
  struct pressio_data* read_impl(struct pressio_data* dims) override {
    auto read_data = impl->read(dims);
    auto selected_data = new pressio_data(read_data->select(start, stride, size, block));
    return selected_data;
  }

  int write_impl(struct pressio_data const* indata) override{
    pressio_data data  = domain_manager().make_readable(domain_plugins().build("malloc"), *indata);
    auto selected_data = data.select();
    return impl->write(&selected_data);
  }

  struct pressio_options get_configuration_impl() const override{
    pressio_options opts;
    set_meta_configuration(opts, "select:io", io_plugins(), impl);
    set(opts, "pressio:stability",  "unstable");
    set(opts, "pressio:thread_safe",  pressio_thread_safety_single);
    return opts;
  }

  int set_options_impl(struct pressio_options const& options) override{
    pressio_data data;
    get_meta(options, "select:io", io_plugins(), impl_id, impl);
    if(options.get("select:start", &data) == pressio_options_key_set) {
      start = data.to_vector<size_t>();
    }
    if(options.get("select:stride", &data) == pressio_options_key_set) {
      stride = data.to_vector<size_t>();
    }
    if(options.get("select:size", &data) == pressio_options_key_set) {
      size = data.to_vector<size_t>();
    }
    if(options.get("select:block", &data) == pressio_options_key_set) {
      block = data.to_vector<size_t>();
    }
    return 0;
  }

  struct pressio_options get_options_impl() const override{
    pressio_options opts;
    set_meta(opts, "select:io", impl_id, impl);
    set(opts, "select:start", pressio_data(std::begin(start), std::end(start)));
    set(opts, "select:stride", pressio_data(std::begin(stride), std::end(stride)));
    set(opts, "select:size", pressio_data(std::begin(size), std::end(size)));
    set(opts, "select:block", pressio_data(std::begin(block), std::end(block)));
    return opts;
  }

  struct pressio_options get_documentation_impl() const override{
    pressio_options opts;
    set_meta_docs(opts, "select:io", "IO to use before selection", impl);
    set(opts, "pressio:description", "selects a subset from a buffer read in or written out");
    set(opts, "select:start", "starting block for selection");
    set(opts, "select:stride", "stride of selection");
    set(opts, "select:size", "size of selection in blocks");
    set(opts, "select:block", "size of each block");
    return opts;
  }

  int patch_version() const override{ 
    return 1;
  }
  const char* version() const override{
    return "0.0.1";
  }

  const char* prefix() const override {
    return "select";
  }

  void set_name_impl(std::string const& new_name) override {
    if(new_name != "") {
      impl->set_name(new_name + "/" + impl->prefix());
    } else {
      impl->set_name(new_name);
    }
  }
  std::vector<std::string> children() const final {
      return { impl->get_name() };
  }

  std::shared_ptr<libpressio_io_plugin> clone() override {
    return compat::make_unique<select_io>(*this);
  }

  private:
  std::string impl_id = "posix";
  pressio_io impl = io_plugins().build("posix");
  std::vector<size_t> start{}, stride{}, size{}, block{};
};

pressio_register registration(io_plugins(), "select", [](){ return compat::make_unique<select_io>(); });
} } }
