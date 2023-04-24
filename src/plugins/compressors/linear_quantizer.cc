#include <libpressio_ext/cpp/compressor.h>
#include <libpressio_ext/cpp/pressio.h>
#include <std_compat/memory.h>
#include <sstream>

namespace  libpressio { namespace linear_quantizer {

struct linear_quantizer_step_finder {
  template <class T>
  double operator()(T const* begin, T const* end) {
    auto ret = std::minmax(begin, end);
    return std::abs(ret.second-ret.first)/auto_step;
  }
  int64_t auto_step;
};

struct linear_quantizer_encoder {
  template <class T>
  pressio_data operator()(T const* begin, T const* end) {
    pressio_data d = pressio_data::owning(pressio_dtype_from_type<int64_t>(), {static_cast<size_t>(end - begin)});
    int64_t* ptr = static_cast<int64_t*>(d.data());
    const size_t len = end-begin;
    for (size_t i = 0; i < len; ++i) {
      ptr[i] = begin[i]/step;
    }

    return d;
  }
  double step;
};

struct linear_quantizer_decoder {
  template <class T, class V>
  int operator()(T const* begin, T const* end, V* ptr) {
    const size_t len = end-begin;
    for (size_t i = 0; i < len; ++i) {
      ptr[i] = begin[i]*step;
    }
    return 0;
  }
  double step;
};



class linear_quantizer: public libpressio_compressor_plugin {
  pressio_options get_options_impl() const override {
    pressio_options opts;
    set_meta(opts, "linear_quantizer:compressor", meta_id, meta);
    set(opts, "pressio:abs", step);
    set(opts, "linear_quantizer:step", step);
    set(opts, "linear_quantizer:auto_step", auto_step);
    return opts;
  }
  pressio_options get_documentation_impl() const override {
    pressio_options opts;
    set_meta_docs(opts, "linear_quantizer:compressor", "compressor to apply after encoding", meta);
    set(opts, "pressio:description", R"(linear_quantizer

applies linear_quantizer encoding to prior to compression and reverses it post decompression.

  y[i] = (x[i])/step

)");
    set(opts, "linear_quantizer:step", "the size of step to use while quantizing");
    set(opts, "linear_quantizer:auto_step", "the number of steps to assume while automatically determine the step size. 0 means use manual step size");

    return opts;
  }
  pressio_options get_configuration_impl() const override {
    pressio_options opts;
    set_meta_configuration(opts, "linear_quantizer:compressor", compressor_plugins(), meta);
    set(opts, "pressio:thread_safe", get_threadsafe(*meta));
    set(opts, "pressio:stability", "experimental");
    return opts;
  }
  int set_options_impl(const pressio_options &options) override {
    get_meta(options, "linear_quantizer:compressor", compressor_plugins(), meta_id, meta);
    get(options, "pressio:abs", &step);
    get(options, "linear_quantizer:step", &step);
    get(options, "linear_quantizer:auto_step", &auto_step);
    return 0;
  }
  int compress_impl(const pressio_data *input, struct pressio_data *output) override {
    if(auto_step) {
      step = pressio_data_for_each<double>(*input, linear_quantizer_step_finder{auto_step});
    }
    auto linear_quantizer_encoded = pressio_data_for_each<pressio_data>(*input, linear_quantizer_encoder{step});
    linear_quantizer_encoded.set_dimensions(std::vector<size_t>(input->dimensions()));
    if(meta->compress(&linear_quantizer_encoded, output) != 0) {
      set_error(meta->error_code(), meta->error_msg());
    }
    return meta->error_code();
  }
  int decompress_impl(const pressio_data *input, struct pressio_data *output) override {
    pressio_data quantized_output = pressio_data::owning(
        pressio_int64_dtype,
        output->dimensions()
        );
    int ret = meta->decompress(input, &quantized_output);
    if(ret > 0) {
      return set_error(meta->error_code(), meta->error_msg());
    } else if(ret < 0) {
      set_error(meta->error_code(), meta->error_msg());
    }
    pressio_data_for_each<int>(quantized_output, *output, linear_quantizer_decoder{step});
    return ret;
  }
  pressio_options get_metrics_results_impl() const override {
    return meta->get_metrics_results();
  }
  void set_name_impl(std::string const& new_name) override {
    if(new_name != "") {
      meta->set_name(new_name + "/" + meta->prefix());
    } else {
      meta->set_name(new_name);
    }
  }
  std::vector<std::string> children_impl() const final {
      return { meta->get_name() };
  }
  const char* prefix() const override { return "linear_quantizer"; }
  const char* version() const override { 
    const static std::string version_str = [this]{
      std::stringstream ss;
      ss << major_version() << '.' << minor_version() << '.' << patch_version();
      return ss.str();
    }();
    return version_str.c_str();
  }
  virtual std::shared_ptr<libpressio_compressor_plugin> clone() override {
    return compat::make_unique<linear_quantizer>(*this);
  }

  int64_t auto_step = 0;
  double step = 0;
  std::string meta_id = "noop";
  pressio_compressor meta = compressor_plugins().build("noop");
};

static pressio_register linear_quantizer_register(
    compressor_plugins(),
    "linear_quantizer",
    []{
      return compat::make_unique<linear_quantizer>();
    }
    );
} }
