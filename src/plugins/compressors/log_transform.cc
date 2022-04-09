#include <libpressio_ext/cpp/compressor.h>
#include <libpressio_ext/cpp/pressio.h>
#include <std_compat/memory.h>
#include <sstream>
#include <cmath>

namespace libpressio { namespace log_transform {

struct log_encoder {
  template <class T>
  pressio_data operator()(T const* begin, T const* end) {
    pressio_data d = pressio_data::owning(pressio_dtype_from_type<T>(), {static_cast<size_t>(end - begin)});
    T* ptr = static_cast<T*>(d.data());

    const size_t len = end-begin;
    for (size_t i = 0; i < len; ++i) {
      ptr[i] = log(begin[i]);
    }

    return d;
  }
};

struct log_decoder {
  template <class T>
  pressio_data operator()(T const* begin, T const* end) {
    pressio_data d = pressio_data::owning(pressio_dtype_from_type<T>(), {static_cast<size_t>(end - begin)});
    T* ptr = static_cast<T*>(d.data());

    const size_t len = end-begin;
    for (size_t i = 0; i < len; ++i) {
      ptr[i] = exp(begin[i]);
    }

    return d;
  }
};



class log_transform: public libpressio_compressor_plugin {
  pressio_options get_options_impl() const override {
    pressio_options opts;
    set_meta(opts, "log_transform:compressor", meta_id, meta);
    return opts;
  }
  pressio_options get_documentation_impl() const override {
    pressio_options opts;
    set_meta_docs(opts, "log_transform:compressor", "compressor to apply after encoding", meta);
    set(opts, "pressio:description", R"(log_transform

applies a log transform to prior to compression and experimental transform post decompression.

y[0] = log(x[0]);

)");

    return opts;
  }
  pressio_options get_configuration_impl() const override {
    pressio_options opts;
    opts.copy_from(meta->get_configuration());
    set(opts, "pressio:thread_safe", static_cast<int32_t>(get_threadsafe(*meta)));
    set(opts, "pressio:stability", "experimental");
    return opts;
  }
  int set_options_impl(const pressio_options &options) override {
    get_meta(options, "log_transform:compressor", compressor_plugins(), meta_id, meta);
    return 0;
  }
  int compress_impl(const pressio_data *input, struct pressio_data *output) override {
    auto log_encoded = pressio_data_for_each<pressio_data>(*input, log_encoder{});
    return meta->compress(&log_encoded, output);
  }
  int decompress_impl(const pressio_data *input, struct pressio_data *output) override {
    int ret = meta->decompress(input, output);
    *output = pressio_data_for_each<pressio_data>(*output, log_decoder{});
    return ret;
  }
  void set_name_impl(std::string const& new_name) override {
    if(new_name != "") {
    meta->set_name(new_name + "/" + meta->prefix());
    } else {
    meta->set_name(new_name);
    }
  }
  const char* prefix() const override { return "log_transform"; }
  const char* version() const override { 
    const static std::string version_str = [this]{
      std::stringstream ss;
      ss << major_version() << '.' << minor_version() << '.' << patch_version();
      return ss.str();
    }();
    return version_str.c_str();
  }
  virtual std::shared_ptr<libpressio_compressor_plugin> clone() override {
    return compat::make_unique<log_transform>(*this);
  }

  std::string meta_id = "noop";
  pressio_compressor meta = compressor_plugins().build("noop");
};

static pressio_register log_transform_register(
    compressor_plugins(),
    "log_transform",
    []{
      return compat::make_unique<log_transform>();
    }
    );
} }
