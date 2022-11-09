#include <chrono>
#include "pressio_options.h"
#include "pressio_compressor.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "std_compat/memory.h"

namespace libpressio {
using std::chrono::high_resolution_clock;
using std::chrono::time_point;
using std::chrono::duration_cast;
using std::chrono::milliseconds;

namespace time_metrics{
  struct time_range{
    time_point<high_resolution_clock> begin;
    time_point<high_resolution_clock> end;
    unsigned int elapsed() const { return duration_cast<milliseconds>(end-begin).count(); }
  };
  using timer = compat::optional<time_range>;
  struct compression_actions {
    time_metrics::timer check_options;
    time_metrics::timer set_options;
    time_metrics::timer get_options;
    time_metrics::timer get_configuration;
    time_metrics::timer compress;
    time_metrics::timer compress_many;
    time_metrics::timer decompress;
    time_metrics::timer decompress_many;
  };
  struct metrics_actions {
    time_metrics::timer begin_check_options;
    time_metrics::timer begin_set_options;
    time_metrics::timer begin_get_options;
    time_metrics::timer begin_get_configuration;
    time_metrics::timer begin_compress;
    time_metrics::timer begin_compress_many;
    time_metrics::timer begin_decompress;
    time_metrics::timer begin_decompress_many;
    time_metrics::timer end_check_options;
    time_metrics::timer end_set_options;
    time_metrics::timer end_get_options;
    time_metrics::timer end_get_configuration;
    time_metrics::timer end_compress;
    time_metrics::timer end_compress_many;
    time_metrics::timer end_decompress;
    time_metrics::timer end_decompress_many;
  };

class time_plugin : public libpressio_metrics_plugin {
  public:
  int set_options(pressio_options const& options) override {
    get_meta(options, "time:metric", metrics_plugins(), child_id, child);
    return 0;
  }
  pressio_options get_options() const override {
    pressio_options opts;
    set_meta(opts, "time:metric", child_id, child);
    return opts;
  }

  int begin_check_options_impl(struct pressio_options const* opts) override {
    self_time.check_options = time_metrics::time_range();
    self_time.check_options->begin = high_resolution_clock::now();

    child_time.begin_check_options = time_metrics::time_range();
    child_time.begin_check_options->begin = high_resolution_clock::now();
    child->begin_check_options(opts);
    child_time.begin_check_options->end = high_resolution_clock::now();
    return 0;
  }

  int end_check_options_impl(struct pressio_options const* opts, int rc) override {
    self_time.check_options->end = high_resolution_clock::now();

    child_time.end_check_options = time_metrics::time_range();
    child_time.end_check_options->begin = high_resolution_clock::now();
    child->end_check_options(opts, rc);
    child_time.end_check_options->end = high_resolution_clock::now();
    return 0;
  }

  int begin_get_options_impl() override {
    self_time.get_options = time_metrics::time_range();
    self_time.get_options->begin = high_resolution_clock::now();

    child_time.begin_get_options = time_metrics::time_range();
    child_time.begin_get_options->begin = high_resolution_clock::now();
    child->begin_get_options();
    child_time.begin_get_options->end = high_resolution_clock::now();
    return 0;
  }

  int end_get_options_impl(struct pressio_options const* opts) override {
    self_time.get_options->end = high_resolution_clock::now();

    child_time.end_get_options = time_metrics::time_range();
    child_time.end_get_options->begin = high_resolution_clock::now();
    child->end_get_options(opts);
    child_time.end_get_options->end = high_resolution_clock::now();
    return 0;
  }

  int begin_get_configuration_impl() override {
    self_time.get_configuration = time_metrics::time_range();
    self_time.get_configuration->begin = high_resolution_clock::now();

    child_time.begin_get_configuration = time_metrics::time_range();
    child_time.begin_get_configuration->begin = high_resolution_clock::now();
    child->begin_get_configuration();
    child_time.begin_get_configuration->end = high_resolution_clock::now();
    return 0;
  }

  int end_get_configuration_impl(struct pressio_options const& opts) override {
    self_time.get_configuration->end = high_resolution_clock::now();

    child_time.end_get_configuration = time_metrics::time_range();
    child_time.end_get_configuration->begin = high_resolution_clock::now();
    child->end_get_configuration(opts);
    child_time.end_get_configuration->end = high_resolution_clock::now();
    return 0;
  }

  int begin_set_options_impl(struct pressio_options const& opts) override {
    self_time.set_options = time_metrics::time_range();
    self_time.set_options->begin = high_resolution_clock::now();

    child_time.begin_set_options = time_metrics::time_range();
    child_time.begin_set_options->begin = high_resolution_clock::now();
    child->begin_set_options(opts);
    child_time.begin_set_options->end = high_resolution_clock::now();
    return 0;
  }

  int end_set_options_impl(struct pressio_options const& opts, int rc) override {
    self_time.set_options->end = high_resolution_clock::now();

    child_time.end_set_options = time_metrics::time_range();
    child_time.end_set_options->begin = high_resolution_clock::now();
    child->end_set_options(opts, rc);
    child_time.end_set_options->end = high_resolution_clock::now();
    return 0;
  }

  int begin_compress_impl(const struct pressio_data * input, struct pressio_data const * output) override {
    self_time.compress = time_metrics::time_range();
    self_time.compress->begin = high_resolution_clock::now();

    child_time.begin_compress = time_metrics::time_range();
    child_time.begin_compress->begin = high_resolution_clock::now();
    child->begin_compress(input, output);
    child_time.begin_compress->end = high_resolution_clock::now();
    return 0;
  }

  int end_compress_impl(struct pressio_data const* input, pressio_data const * output, int rc) override {
    self_time.compress->end = high_resolution_clock::now();

    child_time.end_compress = time_metrics::time_range();
    child_time.end_compress->begin = high_resolution_clock::now();
    child->end_compress(input, output, rc);
    child_time.end_compress->end = high_resolution_clock::now();
    return 0;
  }

  int begin_decompress_impl(struct pressio_data const* input, pressio_data const* output) override {
    self_time.decompress = time_metrics::time_range();
    self_time.decompress->begin = high_resolution_clock::now();

    child_time.begin_decompress = time_metrics::time_range();
    child_time.begin_decompress->begin = high_resolution_clock::now();
    child->begin_decompress(input, output);
    child_time.begin_decompress->end = high_resolution_clock::now();
    return 0;
  }

  int end_decompress_impl(struct pressio_data const* input, pressio_data const* output, int rc) override {
    self_time.decompress->end = high_resolution_clock::now();

    child_time.end_decompress = time_metrics::time_range();
    child_time.end_decompress->begin = high_resolution_clock::now();
    child->end_decompress(input, output, rc);
    child_time.end_decompress->end = high_resolution_clock::now();
    return 0;
  }

  int begin_compress_many_impl(compat::span<const pressio_data* const> const& inputs,
                                   compat::span<const pressio_data* const> const& outputs) override {
    self_time.compress_many = time_metrics::time_range();
    self_time.compress_many->begin = high_resolution_clock::now();

    child_time.begin_compress_many = time_metrics::time_range();
    child_time.begin_compress_many->begin = high_resolution_clock::now();
    child->begin_compress_many(inputs, outputs);
    child_time.begin_compress_many->end = high_resolution_clock::now();
    return 0;
  }

  int end_compress_many_impl(compat::span<const pressio_data* const> const& inputs,
                                   compat::span<const pressio_data* const> const& outputs, int rc) override {
    self_time.compress_many->end = high_resolution_clock::now();

    child_time.end_compress_many = time_metrics::time_range();
    child_time.end_compress_many->begin = high_resolution_clock::now();
    child->end_compress_many(inputs, outputs, rc);
    child_time.end_compress_many->end = high_resolution_clock::now();
    return 0;
   
  }

  int begin_decompress_many_impl(compat::span<const pressio_data* const> const& inputs,
                                   compat::span<const pressio_data* const> const& outputs) override {
    self_time.decompress_many = time_metrics::time_range();
    self_time.decompress_many->begin = high_resolution_clock::now();

    child_time.begin_decompress_many = time_metrics::time_range();
    child_time.begin_decompress_many->begin = high_resolution_clock::now();
    child->begin_decompress_many(inputs, outputs);
    child_time.begin_decompress_many->end = high_resolution_clock::now();
    return 0;
  }

  int end_decompress_many_impl(compat::span<const pressio_data* const> const& inputs,
                                   compat::span<const pressio_data* const> const& outputs, int rc) override {
    self_time.decompress_many->end = high_resolution_clock::now();

    child_time.end_decompress_many = time_metrics::time_range();
    child_time.end_decompress_many->begin = high_resolution_clock::now();
    child->end_decompress_many(inputs, outputs, rc);
    child_time.end_decompress_many->end = high_resolution_clock::now();
    return 0;
  }

  struct pressio_options get_configuration() const override {
    pressio_options opts;
    opts.copy_from(child->get_configuration());
    set(opts, "pressio:stability", "stable");
    set(opts, "pressio:thread_safe", pressio_thread_safety_multiple);
    return opts;
  }

  struct pressio_options get_documentation_impl() const override {
    pressio_options opts;
    set(opts, "pressio:description", "records time used in each operation in ms");
    set(opts, "time:check_options", "time in check_options");
    set(opts, "time:set_options", "time in set options");
    set(opts, "time:get_options", "time in get options");
    set(opts, "time:get_configuration", "time in get configuration");
    set(opts, "time:compress", "time in compress");
    set(opts, "time:decompress", "time in decompress");
    set(opts, "time:compress_many", "time in compress_many");
    set(opts, "time:decompress_many", "time in decompress_many");
    set(opts, "time:metric", "time a child metrics plugin");

    set(opts, "time:begin_check_options", "time for child's begin_check_options");
    set(opts, "time:begin_set_options", "time for child's begin_set_options");
    set(opts, "time:begin_get_options", "time for child's begin_get_options");
    set(opts, "time:begin_compress", "time for child's begin_compress");
    set(opts, "time:begin_get_configuration", "time for child's begin_get_configuration");
    set(opts, "time:begin_decompress", "time for child's begin_decompress");
    set(opts, "time:begin_compress_many", "time for child's begin_compress_many");
    set(opts, "time:begin_decompress_many", "time for child's begin_decompress_many");
    set(opts, "time:end_check_options", "time for child's end_check_options");
    set(opts, "time:end_set_options", "time for child's end_set_options");
    set(opts, "time:end_get_options", "time for child's end_get_options");
    set(opts, "time:end_compress", "time for child's end_compress");
    set(opts, "time:end_get_configuration", "time for child's end_get_configuration");
    set(opts, "time:end_decompress", "time for child's end_decompress");
    set(opts, "time:end_compress_many", "time for child's end_compress_many");
    set(opts, "time:end_decompress_many", "time for child's end_decompress_many");
    return opts;
  }

  pressio_options get_metrics_results(pressio_options const & parent)  override {
    struct pressio_options opt = child->get_metrics_results(parent);

    auto set_or = [&opt, this](const char* key, time_metrics::timer time) {
      if(time) {
        set(opt, key, time->elapsed());
      } else {
        set_type(opt, key, pressio_option_uint32_type);
      }
    };

    set_or("time:check_options", self_time.check_options);
    set_or("time:set_options", self_time.set_options);
    set_or("time:get_options", self_time.get_options);
    set_or("time:compress", self_time.compress);
    set_or("time:get_configuration", self_time.get_configuration);
    set_or("time:decompress", self_time.decompress);
    set_or("time:compress_many", self_time.compress_many);
    set_or("time:decompress_many", self_time.decompress_many);

    set_or("time:begin_check_options", child_time.begin_check_options);
    set_or("time:begin_set_options", child_time.begin_set_options);
    set_or("time:begin_get_options", child_time.begin_get_options);
    set_or("time:begin_compress", child_time.begin_compress);
    set_or("time:begin_get_configuration", child_time.begin_get_configuration);
    set_or("time:begin_decompress", child_time.begin_decompress);
    set_or("time:begin_compress_many", child_time.begin_compress_many);
    set_or("time:begin_decompress_many", child_time.begin_decompress_many);
    set_or("time:end_check_options", child_time.end_check_options);
    set_or("time:end_set_options", child_time.end_set_options);
    set_or("time:end_get_options", child_time.end_get_options);
    set_or("time:end_compress", child_time.end_compress);
    set_or("time:end_get_configuration", child_time.end_get_configuration);
    set_or("time:end_decompress", child_time.end_decompress);
    set_or("time:end_compress_many", child_time.end_compress_many);
    set_or("time:end_decompress_many", child_time.end_decompress_many);

    return opt;
  }

  std::unique_ptr<libpressio_metrics_plugin> clone() override {
    return compat::make_unique<time_plugin>(*this);
  }

  const char* prefix() const override {
    return "time";
  }

  void set_name_impl(std::string const& new_name) override {
    if(new_name != "") {
    child->set_name(new_name + "/" + child->prefix());
    } else {
    child->set_name(new_name);
    }
  }

  private:
  compression_actions self_time;
  metrics_actions child_time;
  std::string child_id = "noop";
  pressio_metrics child = metrics_plugins().build("noop");
};

static pressio_register metrics_time_plugin(metrics_plugins(), "time", [](){ return compat::make_unique<time_plugin>(); });
}
}
