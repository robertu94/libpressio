#include "pressio_options.h"
#include "pressio_compressor.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "std_compat/memory.h"
#include <sys/resource.h>

namespace libpressio {
namespace memory_metrics{
  struct memory_range{
    uint64_t begin = 0;
    uint64_t end = 0;
    uint64_t used() const { return (end-begin) * 1024; }
  };
  using tracker = compat::optional<memory_range>;
  uint64_t now() {
    rusage r;
    getrusage(RUSAGE_SELF, &r);
    return r.ru_maxrss;
  }

class memory_plugin : public libpressio_metrics_plugin {
  public:

  int begin_check_options_impl(struct pressio_options const* ) override {
    check_options = memory_metrics::memory_range();
    check_options->begin = memory_metrics::now();
    return 0;
  }

  int end_check_options_impl(struct pressio_options const*, int ) override {
    check_options->end = memory_metrics::now();
    return 0;
  }

  int begin_get_options_impl() override {
    get_options = memory_metrics::memory_range();
    get_options->begin = memory_metrics::now();
    return 0;
  }

  int end_get_options_impl(struct pressio_options const* ) override {
    get_options->end = memory_metrics::now();
    return 0;
  }

  int begin_get_configuration_impl() override {
    get_configuration_tracker = memory_metrics::memory_range();
    get_configuration_tracker->begin = memory_metrics::now();
    return 0;
  }

  int end_get_configuration_impl(struct pressio_options const& ) override {
    get_configuration_tracker->end = memory_metrics::now();
    return 0;
  }


  int begin_set_options_impl(struct pressio_options const& ) override {
    set_options = memory_metrics::memory_range();
    set_options->begin = memory_metrics::now();
    return 0;
  }

  int end_set_options_impl(struct pressio_options const& , int ) override {
    set_options->end = memory_metrics::now();
    return 0;
  }

  int begin_compress_impl(const struct pressio_data * , struct pressio_data const * ) override {
    compress = memory_metrics::memory_range();
    compress->begin = memory_metrics::now();
    return 0;
  }

  int end_compress_impl(struct pressio_data const* , pressio_data const * , int ) override {
    compress->end = memory_metrics::now();
    return 0;
  }

  int begin_decompress_impl(struct pressio_data const* , pressio_data const* ) override {
    decompress = memory_metrics::memory_range();
    decompress->begin = memory_metrics::now();
    return 0;
  }

  int end_decompress_impl(struct pressio_data const* , pressio_data const* , int ) override {
    decompress->end = memory_metrics::now();
    return 0;
  }

  int begin_compress_many_impl(compat::span<const pressio_data* const> const&,
                                   compat::span<const pressio_data* const> const&) override {
    compress_many = memory_metrics::memory_range();
    compress_many->begin = memory_metrics::now();
    return 0;
  }

  int end_compress_many_impl(compat::span<const pressio_data* const> const& ,
                                   compat::span<const pressio_data* const> const& , int ) override {
    compress_many->end = memory_metrics::now();
    return 0;
   
  }

  int begin_decompress_many_impl(compat::span<const pressio_data* const> const& ,
                                   compat::span<const pressio_data* const> const& ) override {
    decompress_many = memory_metrics::memory_range();
    decompress_many->begin = memory_metrics::now();
    return 0;
  }

  int end_decompress_many_impl(compat::span<const pressio_data* const> const& ,
                                   compat::span<const pressio_data* const> const& , int ) override {
    decompress_many->end = memory_metrics::now();
    return 0;
  }

  struct pressio_options get_metrics_results(pressio_options const&) override {
    struct pressio_options opt;

    auto set_or = [&opt, this](const char* key, memory_metrics::tracker memory) {
      if(memory) {
        this->set(opt, key, memory->used());
      } else {
        this->set_type(opt, key, pressio_option_uint64_type);
      }
    };

    set_or("memory:check_options", check_options);
    set_or("memory:set_options", set_options);
    set_or("memory:get_options", get_options);
    set_or("memory:compress", compress);
    set_or("memory:decompress", decompress);
    set_or("memory:compress_many", compress_many);
    set_or("memory:decompress_many", decompress_many);

    return opt;
  }

  struct pressio_options get_configuration_impl() const override {
    pressio_options opts;
    set(opts, "pressio:stability", "stable");
    set(opts, "pressio:thread_safe", pressio_thread_safety_multiple);
    return opts;
  }


  struct pressio_options get_documentation_impl() const override {
    pressio_options opts;

    set(opts, "pressio:description", "uses getrusage to record memory usage");
    set(opts, "memory:check_options", "memory used in bytes by check options");
    set(opts, "memory:set_options", "memory used in bytes by set options");
    set(opts, "memory:get_options", "memory used in bytes by get options");
    set(opts, "memory:compress", "memory used in bytes by compress");
    set(opts, "memory:decompress", "memory used in bytes by decompress");
    set(opts, "memory:compress_many", "memory used in bytes by compress_many");
    set(opts, "memory:decompress_many", "memory used in bytes by decompress_many");

    return opts;
  }

  std::unique_ptr<libpressio_metrics_plugin> clone() override {
    return compat::make_unique<memory_plugin>(*this);
  }

  const char* prefix() const override {
    return "memory";
  }

  private:
  memory_metrics::tracker check_options;
  memory_metrics::tracker set_options;
  memory_metrics::tracker get_options;
  memory_metrics::tracker get_configuration_tracker;
  memory_metrics::tracker compress;
  memory_metrics::tracker compress_many;
  memory_metrics::tracker decompress;
  memory_metrics::tracker decompress_many;
};

static pressio_register metrics_memory_plugin(metrics_plugins(), "memory", [](){ return compat::make_unique<memory_plugin>(); });

}
}
