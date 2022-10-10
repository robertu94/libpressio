#include <libpressio_ext/cpp/compressor.h>
#include <libpressio_ext/cpp/pressio.h>
#include <std_compat/memory.h>
#include <sstream>

namespace libpressio { namespace delta_encoder {

struct delta_encoder {
  template <class T>
  pressio_data operator()(T const* begin, T const* end) {
    pressio_data d = pressio_data::owning(pressio_dtype_from_type<T>(), {static_cast<size_t>(end - begin)});
    T* ptr = static_cast<T*>(d.data());

    const size_t len = end-begin;
    ptr[0] = begin[0];
    for (size_t i = 1; i < len; ++i) {
      ptr[i] = begin[i] - begin[i-1];
    }

    return d;
  }
};

struct delta_decoder {
  template <class T>
  pressio_data operator()(T const* begin, T const* end) {
    pressio_data d = pressio_data::owning(pressio_dtype_from_type<T>(), {static_cast<size_t>(end - begin)});
    T* ptr = static_cast<T*>(d.data());

    const size_t len = end-begin;
    ptr[0] = begin[0];
    ptr[1] = begin[0] + begin[1];
    for (size_t i = 2; i < len; ++i) {
      ptr[i] = begin[i] + ptr[i-1];
    }

    return d;
  }
};



class delta_encoding: public libpressio_compressor_plugin {
  pressio_options get_options_impl() const override {
    pressio_options opts;
    set_meta(opts, "delta_encoding:compressor", meta_id, meta);
    return opts;
  }
  pressio_options get_documentation_impl() const override {
    pressio_options opts;
    set_meta_docs(opts, "delta_encoding:compressor", "compressor to apply after encoding", meta);
    set(opts, "pressio:description", R"(delta_encoding

applies delta encoding to prior to compression and reverses it post decompression.

y[0] = x[0];
y[i] = x[i] - x[i-1];

)");

    return opts;
  }
  pressio_options get_configuration_impl() const override {
    pressio_options opts;
    opts.copy_from(meta->get_configuration());
    set(opts, "pressio:thread_safe", get_threadsafe(*meta));
    set(opts, "pressio:stability", "experimental");
    return opts;
  }
  int set_options_impl(const pressio_options &options) override {
    get_meta(options, "delta_encoding:compressor", compressor_plugins(), meta_id, meta);
    return 0;
  }
  int compress_impl(const pressio_data *input, struct pressio_data *output) override {
    auto delta_encoded = pressio_data_for_each<pressio_data>(*input, delta_encoder{});
    return meta->compress(&delta_encoded, output);
  }
  int decompress_impl(const pressio_data *input, struct pressio_data *output) override {
    int ret = meta->decompress(input, output);
    *output = pressio_data_for_each<pressio_data>(*output, delta_decoder{});
    return ret;
  }
  void set_name_impl(std::string const& new_name) override {
    if(new_name != "") {
    meta->set_name(new_name + "/" + meta->prefix());
    } else {
    meta->set_name(new_name);
    }
  }
  pressio_options get_metrics_results_impl() const override {
    return meta->get_metrics_results();
  }
  const char* prefix() const override { return "delta_encoding"; }
  const char* version() const override { 
    const static std::string version_str = [this]{
      std::stringstream ss;
      ss << major_version() << '.' << minor_version() << '.' << patch_version();
      return ss.str();
    }();
    return version_str.c_str();
  }
  virtual std::shared_ptr<libpressio_compressor_plugin> clone() override {
    return compat::make_unique<delta_encoding>(*this);
  }

  std::string meta_id = "noop";
  pressio_compressor meta = compressor_plugins().build("noop");
};

static pressio_register delta_encoding_register(
    compressor_plugins(),
    "delta_encoding",
    []{
      return compat::make_unique<delta_encoding>();
    }
    );
} }
