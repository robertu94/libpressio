#include <iostream>
#include <fstream>
#include "pressio_compressor.h"
#include "libpressio_ext/io/posix.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/io.h"
#include "std_compat/memory.h"

namespace libpressio { namespace io { namespace copy_template_ns {

struct copy_template_io : public libpressio_io_plugin {
  virtual struct pressio_data* read_impl(struct pressio_data* data) override {
    return impl->read(data);
  }
  virtual int write_impl(struct pressio_data const* data) override{
    if(not template_path.empty()) {
      if(copy_template()) return error_code();
    }
    return impl->write(data);
  }
  virtual int set_options_impl(struct pressio_options const& options) override{
    get_meta(options, "copy_template:io", io_plugins(), impl_id, impl);
    get(options, "copy_template:template_path", &template_path);
    get(options, "io:path", &path);
    return 0;
  }
  virtual struct pressio_options get_options_impl() const override{
    pressio_options opts;
    set_meta(opts, "copy_template:io", impl_id, impl);
    set(opts, "copy_template:template_path", template_path);
    set(opts, "io:path", path);
    return opts;
  }
  virtual struct pressio_options get_configuration_impl() const override{
    pressio_options opts;
    set_meta_configuration(opts, "copy_template:io", io_plugins(), impl);
    set(opts, "pressio:stability", "stable");
    set(opts, "pressio:thread_safe",  pressio_thread_safety_single);
    return opts;
  }
  virtual struct pressio_options get_documentation_impl() const override{
    pressio_options opts;
    set_meta_docs(opts, "copy_template:io", "io object to use after applying template", impl);
    set(opts, "pressio:description", "copy a template before preforming writes");
    set(opts, "copy_template:template_path", "path to the template file to copy");
    set(opts, "io:path", "path to write the file to");
    return opts;
  }



  int patch_version() const override{ 
    return 1;
  }
  virtual const char* version() const override{
    return "0.0.1";
  }
  const char* prefix() const override {
    return "copy_template";
  }

  std::shared_ptr<libpressio_io_plugin> clone() override {
    return compat::make_unique<copy_template_io>(*this);
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

  private:
  int copy_template() {
    std::ifstream template_file(template_path, std::ios::binary);
    std::ofstream output_file(path, std::ios::binary);
    if(!template_file) return set_error(1, "template_file does not exist");
    if(!output_file) return set_error(2, "output_file does not exist");
    output_file << template_file.rdbuf();
    return 0;
  }

  std::string path;
  std::string template_path;
  std::string impl_id = "posix";
  pressio_io impl = io_plugins().build("posix");
};

pressio_register registration(io_plugins(), "copy_template", [](){ return compat::make_unique<copy_template_io>(); });
} }}
