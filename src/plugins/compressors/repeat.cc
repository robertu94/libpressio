#include "std_compat/memory.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"

namespace libpressio { namespace repeat_ns {

class repeat_compressor_plugin : public libpressio_compressor_plugin {
public:
  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    set_meta(options, "repeat:compressor", comp_id, comp);
    set(options, "repeat:count", count);
    set(options, "repeat:clone_output", clone_output);

    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    set_meta_configuration(options, "repeat:compressor", compressor_plugins(), comp);
    set(options, "pressio:thread_safe", get_threadsafe(*comp));
    set(options, "pressio:stability", "experimental");
    return options;
  }

  struct pressio_options get_documentation_impl() const override
  {
    struct pressio_options options;
    set_meta_docs(options, "repeat:compressor", "compressor to call repeatedly", comp);
    set(options, "pressio:description", R"(call a compressor multiple times i.e. to get an average timing)");
    set(options, "repeat:count", "how many repeats to do");
    set(options, "repeat:clone_output", "clone output or re-use existing output");
    return options;
  }


  int set_options_impl(struct pressio_options const& options) override
  {
    get_meta(options, "repeat:compressor", compressor_plugins(), comp_id, comp);
    get(options, "repeat:count", &count);
    get(options, "repeat:clone_output", &clone_output);
    return 0;
  }

  int compress_impl(const pressio_data* input,
                    struct pressio_data* output) override
  {
      int rc = 0;
      pressio_data out;
      for (uint32_t i = 0; i < count; ++i) {
          if(clone_output) {
              rc = comp->compress(input, output);
          } else {
              pressio_data tmp_out(pressio_data::clone(*output));
              rc = comp->compress(input, &tmp_out);
              if (rc == 0 && i + 1 == count) {
                  *output = std::move(tmp_out);
              }
          }
          if(rc > 0) {
              set_error(rc, comp->error_msg());
              return rc;
          }
      }
      return rc;
  }

  int decompress_impl(const pressio_data* input,
                      struct pressio_data* output) override
  {
      int rc = 0;
      pressio_data out;
      for (uint32_t i = 0; i < count; ++i) {
          if(clone_output) {
              rc = comp->compress(input, output);
          } else {
              pressio_data tmp_out(pressio_data::clone(*output));
              rc = comp->decompress(input, &tmp_out);
              if (rc == 0 && i + 1 == count) {
                  *output = std::move(tmp_out);
              }
          }
          if(rc > 0) {
              set_error(rc, comp->error_msg());
              return rc;
          }
      }
      return rc;
  }

  int major_version() const override { return 0; }
  int minor_version() const override { return 0; }
  int patch_version() const override { return 1; }
  const char* version() const override { return "0.0.1"; }
  const char* prefix() const override { return "repeat"; }

  void set_name_impl(std::string const& new_name) override {
      comp->set_name(new_name + "/" + comp->prefix());
  }
  std::vector<std::string> children_impl() const final {
      return {comp->get_name()};
  }

  pressio_options get_metrics_results_impl() const override {
    return comp->get_metrics_results();
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<repeat_compressor_plugin>(*this);
  }

  std::string comp_id = "noop";
  pressio_compressor comp = compressor_plugins().build(comp_id);
  uint32_t count = 1;
  bool clone_output = false;
};

static pressio_register compressor_many_fields_plugin(compressor_plugins(), "repeat", []() {
  return compat::make_unique<repeat_compressor_plugin>();
});

} }
