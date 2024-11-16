#include "pressio_data.h"
#include "pressio_compressor.h"
#include "pressio_options.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/domain_manager.h"
#include "std_compat/memory.h"
#include "std_compat/optional.h"
#include <cmath>

namespace libpressio { namespace max_error_metrics_ns {

class max_error_plugin : public libpressio_metrics_plugin {
    struct max_error_info {
        double max_err = 0;
        uint64_t max_err_index = 0;
    };
    struct compute_error_info {
        template <class T, class U>
        max_error_info operator()(T* in1, T* end1, U* in2, U* end2) {
            max_error_info info;
            size_t count = std::max(std::distance(in1, end1), std::distance(in2, end2));
            if(count >= 1) {
                info.max_err = std::fabs(in1[0]- in2[0]);
            }
            for (size_t i = 1; i < count; ++i) {
                double err = std::fabs(in1[i]- in2[i]);
                if(err > info.max_err) {
                    info.max_err = err;
                    info.max_err_index = i;
                }
            }

            return info;
        }
    };
  public:
    int begin_compress_impl(struct pressio_data const* input, pressio_data const*) override {
      if(!input || !input->has_data()) return 0;
      in = pressio_data::clone(domain_manager().make_readable(domain_plugins().build("malloc"), *input));
      return 0; }

    int end_decompress_impl(struct pressio_data const* , pressio_data const* output, int) override {
      if(!output || !output->has_data() || !in.has_data()) return 0;
      errors = pressio_data_for_each<max_error_info>(in, domain_manager().make_readable(domain_plugins().build("malloc"), *output), compute_error_info{});
      return 0;
    }

  
  struct pressio_options get_configuration_impl() const override {
    pressio_options opts;
    set(opts, "pressio:stability", "stable");
    set(opts, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(opts, "predictors:requires_decompress", true);
    set(opts, "predictors:invalidate", std::vector<std::string>{"predictors:error_dependent"});
    return opts;
  }

  struct pressio_options get_documentation_impl() const override {
    pressio_options opt;
    set(opt, "pressio:description", "computes the index of the largest error");
    set(opt, "max_error:max_error", "magnitude of the largest error");
    set(opt, "max_error:max_error_index", "index of the largest error");
    return opt;
  }

  pressio_options get_metrics_results(pressio_options const &) override {
    pressio_options opt;
    if(errors) {
        set(opt, "max_error:max_error", errors->max_err);
        set(opt, "max_error:max_error_index", errors->max_err_index);
    } else {
        set_type(opt, "max_error:max_error", pressio_option_double_type);
        set_type(opt, "max_error:max_error_index", pressio_option_uint64_type);
    }
    return opt;
  }

  std::unique_ptr<libpressio_metrics_plugin> clone() override {
    return compat::make_unique<max_error_plugin>(*this);
  }
  const char* prefix() const override {
    return "max_error";
  }

  private:
  pressio_data in;
  compat::optional<max_error_info> errors = compat::nullopt;
};

static pressio_register metrics_max_error_plugin(metrics_plugins(), "max_error", [](){ return compat::make_unique<max_error_plugin>(); });
}}
