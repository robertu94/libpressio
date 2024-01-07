#include <libpressio_ext/cpp/libpressio.h>
#include <std_compat/memory.h>

namespace libpressio { namespace mask1d {
struct apply_mask{

  template <class T, class V>
  pressio_data operator()(T* begin, T* end, V* mask_begin, V* mask_end) {
    pressio_data d;
    if(mask_begin == nullptr || mask_begin == mask_end) {
         d = pressio_data::copy(
            pressio_dtype_from_type<T>(),
            begin,
            {static_cast<size_t>(end-begin)}
            );
    } else {
        size_t nnz = std::count_if(mask_begin, mask_end, [](V m){return m != 0;});
        d = pressio_data::owning(
            pressio_dtype_from_type<T>(),
            {nnz}
            );
        T* ptr_d = static_cast<T*>(d.data());

        while(begin != end) {
          if(*mask_begin) {
            *ptr_d = *begin;
            ptr_d++;
          }
          mask_begin++;
          begin++;
        }
    }

    return d;
  }
};

class mask_metrics: public libpressio_metrics_plugin {
  public:
    pressio_options get_metrics_results(pressio_options const &options) override {
    return plugin->get_metrics_results(options);
  }

  int begin_compress_impl(pressio_data const* input, pressio_data const* output) override {
    auto masked(pressio_data_for_each<pressio_data>(*input, mask, apply_mask{}));
    plugin->begin_compress(&masked, output);
    return 0;
  }

  int end_decompress_impl(pressio_data const* input, pressio_data const* output, int rc) override {
    auto masked(pressio_data_for_each<pressio_data>(*output, mask, apply_mask{}));
    plugin->end_decompress(input, &masked, rc);
    return 0;
  }

  struct pressio_options get_configuration_impl() const override {
    pressio_options opts;
    set_meta_configuration(opts, "mask:metrics", metrics_plugins(), plugin);
    set(opts, "pressio:stability", "stable");
    set(opts, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(opts, "predictors:requires_decompress", true);
    set(opts, "predictors:invalidate", std::vector<std::string>{"predictors:error_dependent"});
    return opts;
  }

  pressio_options get_documentation_impl() const override {
    pressio_options opts;
    set(opts, "pressio:description", "applies to mask to keep only interesting values");
    set_meta(opts, "mask:metrics", "the metrics to compute after applying the mask", plugin);
    set(opts, "mask:mask", "the mask to apply; the mask has 1 on values to include");
    return opts;
  }


  pressio_options get_options() const override {
    pressio_options opts;
    set_meta(opts, "mask:metrics", plugin_id, plugin);
    set(opts, "mask:mask", mask);
    return opts;
  }

  int set_options(pressio_options const& opts) override {
    get_meta(opts, "mask:metrics", metrics_plugins(), plugin_id, plugin);
    get(opts, "mask:mask", &mask);
    return 0;
  }

  const char* prefix() const override {
    return "mask";
  }

  void set_name_impl(std::string const& new_name) override {
    if(new_name != "") {
      plugin->set_name(new_name + "/" + plugin->prefix());
    } else {
      plugin->set_name(new_name);
    }
  }
  std::vector<std::string> children() const final {
      return { plugin->get_name() };
  }

  std::unique_ptr<libpressio_metrics_plugin> clone() override {
    return compat::make_unique<mask_metrics>(*this);
  }

  private:
  pressio_metrics plugin =  metrics_plugins().build("noop");
  std::string plugin_id = "noop";
  pressio_data mask;
};

pressio_register mask_plugin(metrics_plugins(), "mask", []{ return compat::make_unique<mask_metrics>(); });
} }
