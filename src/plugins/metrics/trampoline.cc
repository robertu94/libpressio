#include <cstdlib>

#include "pressio_options.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "std_compat/memory.h"

namespace libpressio { namespace metrics { namespace trampoline_ns {
class trampoline_metrics_plugin : public libpressio_metrics_plugin {
public:

    trampoline_metrics_plugin(
        void* auxiliary,
        struct pressio_options* (*get_configuration_impl_trampoline)(void const *),
        struct pressio_options* (*get_documentation_impl_trampoline)(void const *),
        struct pressio_options* (*get_options_trampoline)(void const *),
        int (*set_options_trampoline)(void*, struct pressio_options const *),
        int (*begin_check_options_impl_trampoline)(void*, struct pressio_options const *),
        int (*end_check_options_impl_trampoline)(void*, struct pressio_options const *, int),
        int (*begin_get_documentation_impl_trampoline)(void*),
        int (*end_get_documentation_impl_trampoline)(void*, struct pressio_options const *),
        int (*begin_get_configuration_impl_trampoline)(void*),
        int (*end_get_configuration_impl_trampoline)(void*, struct pressio_options const *),
        int (*begin_get_options_impl_trampoline)(void*),
        int (*end_get_options_impl_trampoline)(void*, struct pressio_options const *),
        int (*begin_set_options_impl_trampoline)(void*, struct pressio_options const *),
        int (*end_set_options_impl_trampoline)(void*, struct pressio_options const *, int),
        int (*begin_compress_impl_trampoline)(void*, struct pressio_data const *, struct pressio_data const *),
        int (*end_compress_impl_trampoline)(void*, struct pressio_data const *, struct pressio_data const *, int),
        int (*begin_decompress_impl_trampoline)(void*, struct pressio_data const *, struct pressio_data const *),
        int (*end_decompress_impl_trampoline)(void*, struct pressio_data const *, struct pressio_data const *, int),
        int (*begin_compress_many_impl_trampoline)(void*, struct pressio_data const * const *, size_t, struct pressio_data const * const *, size_t),
        int (*end_compress_many_impl_trampoline)(void*, struct pressio_data const * const *, size_t, struct pressio_data const * const *, size_t, int),
        int (*begin_decompress_many_impl_trampoline)(void*, struct pressio_data const * const *, size_t, struct pressio_data const * const *, size_t),
        int (*end_decompress_many_impl_trampoline)(void*, struct pressio_data const * const *, size_t, struct pressio_data const * const *, size_t, int),
        int (*view_segment_impl_trampoline)(void*, struct pressio_data const *, const char *),
        const char* prefix_trampoline,
        struct pressio_options* (*get_metrics_results_trampoline)(void const *),
        int (*error_code_trampoline)(void const *),
        const char* (*error_msg_trampoline)(void const *),
        void* (*clone_trampoline)(void const *),
        void (*release_trampoline)(void*)
    ):
      auxiliary(auxiliary),
      get_configuration_impl_trampoline(get_configuration_impl_trampoline),
      get_documentation_impl_trampoline(get_documentation_impl_trampoline),
      get_options_trampoline(get_options_trampoline),
      set_options_trampoline(set_options_trampoline),
      begin_check_options_impl_trampoline(begin_check_options_impl_trampoline),
      end_check_options_impl_trampoline(end_check_options_impl_trampoline),
      begin_get_documentation_impl_trampoline(begin_get_documentation_impl_trampoline),
      end_get_documentation_impl_trampoline(end_get_documentation_impl_trampoline),
      begin_get_configuration_impl_trampoline(begin_get_configuration_impl_trampoline),
      end_get_configuration_impl_trampoline(end_get_configuration_impl_trampoline),
      begin_get_options_impl_trampoline(begin_get_options_impl_trampoline),
      end_get_options_impl_trampoline(end_get_options_impl_trampoline),
      begin_set_options_impl_trampoline(begin_set_options_impl_trampoline),
      end_set_options_impl_trampoline(end_set_options_impl_trampoline),
      begin_compress_impl_trampoline(begin_compress_impl_trampoline),
      end_compress_impl_trampoline(end_compress_impl_trampoline),
      begin_decompress_impl_trampoline(begin_decompress_impl_trampoline),
      end_decompress_impl_trampoline(end_decompress_impl_trampoline),
      begin_compress_many_impl_trampoline(begin_compress_many_impl_trampoline),
      end_compress_many_impl_trampoline(end_compress_many_impl_trampoline),
      begin_decompress_many_impl_trampoline(begin_decompress_many_impl_trampoline),
      end_decompress_many_impl_trampoline(end_decompress_many_impl_trampoline),
      view_segment_impl_trampoline(view_segment_impl_trampoline),
      prefix_trampoline(prefix_trampoline),
      get_metrics_results_trampoline(get_metrics_results_trampoline),
      error_code_trampoline(error_code_trampoline),
      error_msg_trampoline(error_msg_trampoline),
      clone_trampoline(clone_trampoline),
      release_trampoline(release_trampoline)
    {}

    struct pressio_options get_configuration_impl() const override {
        struct pressio_options * options_ptr = get_configuration_impl_trampoline(auxiliary);
        struct pressio_options options = pressio_options(*options_ptr);
        delete options_ptr;
        return options;
    }

    struct pressio_options get_documentation_impl() const override {
        struct pressio_options * options_ptr = get_documentation_impl_trampoline(auxiliary);
        struct pressio_options options = pressio_options(*options_ptr);
        delete options_ptr;
        return options;
    }

    struct pressio_options get_options() const override {
        struct pressio_options * options_ptr = get_options_trampoline(auxiliary);
        struct pressio_options options = pressio_options(*options_ptr);
        delete options_ptr;
        return options;
    }

    int set_options(struct pressio_options const& options) override {
        int result = set_options_trampoline(auxiliary, &options);
        if (result != 0) {
            set_error(error_code_trampoline(auxiliary), error_msg_trampoline(auxiliary));
        }
        return result;
    }

    int begin_check_options_impl(struct pressio_options const * opts) override {
        int result = begin_check_options_impl_trampoline(auxiliary, opts);
        if (result != 0) {
            set_error(error_code_trampoline(auxiliary), error_msg_trampoline(auxiliary));
        }
        return result;
    }

    int end_check_options_impl(struct pressio_options const * opts, int rc) override {
        int result = end_check_options_impl_trampoline(auxiliary, opts, rc);
        if (result != 0) {
            set_error(error_code_trampoline(auxiliary), error_msg_trampoline(auxiliary));
        }
        return result;
    }

    int begin_get_documentation_impl() override {
        int result = begin_get_documentation_impl_trampoline(auxiliary);
        if (result != 0) {
            set_error(error_code_trampoline(auxiliary), error_msg_trampoline(auxiliary));
        }
        return result;
    }

    int end_get_documentation_impl(struct pressio_options const & opts) override {
        int result = end_get_documentation_impl_trampoline(auxiliary, &opts);
        if (result != 0) {
            set_error(error_code_trampoline(auxiliary), error_msg_trampoline(auxiliary));
        }
        return result;
    }

    int begin_get_configuration_impl() override {
        int result = begin_get_configuration_impl_trampoline(auxiliary);
        if (result != 0) {
            set_error(error_code_trampoline(auxiliary), error_msg_trampoline(auxiliary));
        }
        return result;
    }

    int end_get_configuration_impl(struct pressio_options const & opts) override {
        int result = end_get_configuration_impl_trampoline(auxiliary, &opts);
        if (result != 0) {
            set_error(error_code_trampoline(auxiliary), error_msg_trampoline(auxiliary));
        }
        return result;
    }

    int begin_get_options_impl() override {
        int result = begin_get_options_impl_trampoline(auxiliary);
        if (result != 0) {
            set_error(error_code_trampoline(auxiliary), error_msg_trampoline(auxiliary));
        }
        return result;
    }

    int end_get_options_impl(struct pressio_options const * opts) override {
        int result = end_get_options_impl_trampoline(auxiliary, opts);
        if (result != 0) {
            set_error(error_code_trampoline(auxiliary), error_msg_trampoline(auxiliary));
        }
        return result;
    }

    int begin_set_options_impl(struct pressio_options const & opts) override {
        int result = begin_set_options_impl_trampoline(auxiliary, &opts);
        if (result != 0) {
            set_error(error_code_trampoline(auxiliary), error_msg_trampoline(auxiliary));
        }
        return result;
    }

    int end_set_options_impl(struct pressio_options const & opts, int rc) override {
        int result = end_set_options_impl_trampoline(auxiliary, &opts, rc);
        if (result != 0) {
            set_error(error_code_trampoline(auxiliary), error_msg_trampoline(auxiliary));
        }
        return result;
    }

    int begin_compress_impl(const struct pressio_data * input, struct pressio_data const * output) override {
        int result = begin_compress_impl_trampoline(auxiliary, input, output);
        if (result != 0) {
            set_error(error_code_trampoline(auxiliary), error_msg_trampoline(auxiliary));
        }
        return result;
    }

    int end_compress_impl(struct pressio_data const * input, pressio_data const * output, int rc) override {
        int result = end_compress_impl_trampoline(auxiliary, input, output, rc);
        if (result != 0) {
            set_error(error_code_trampoline(auxiliary), error_msg_trampoline(auxiliary));
        }
        return result;
    }

    int begin_decompress_impl(struct pressio_data const * input, pressio_data const * output) override {
        int result = begin_decompress_impl_trampoline(auxiliary, input, output);
        if (result != 0) {
            set_error(error_code_trampoline(auxiliary), error_msg_trampoline(auxiliary));
        }
        return result;
    }

    int end_decompress_impl(struct pressio_data const * input, pressio_data const * output, int rc) override {
        int result = end_decompress_impl_trampoline(auxiliary, input, output, rc);
        if (result != 0) {
            set_error(error_code_trampoline(auxiliary), error_msg_trampoline(auxiliary));
        }
        return result;
    }

    int begin_compress_many_impl(compat::span<const pressio_data* const> const& inputs,
                                     compat::span<const pressio_data* const> const& outputs) override {
        int result = begin_compress_many_impl_trampoline(auxiliary, inputs.data(), inputs.size(), outputs.data(), outputs.size());
        if (result != 0) {
            set_error(error_code_trampoline(auxiliary), error_msg_trampoline(auxiliary));
        }
        return result;
    }

    int end_compress_many_impl(compat::span<const pressio_data* const> const& inputs,
                                     compat::span<const pressio_data* const> const& outputs, int rc) override {
        int result = end_compress_many_impl_trampoline(auxiliary, inputs.data(), inputs.size(), outputs.data(), outputs.size(), rc);
        if (result != 0) {
            set_error(error_code_trampoline(auxiliary), error_msg_trampoline(auxiliary));
        }
        return result;
    }

    int begin_decompress_many_impl(compat::span<const pressio_data* const> const& inputs,
                                     compat::span<const pressio_data* const> const& outputs) override {
        int result = begin_decompress_many_impl_trampoline(auxiliary, inputs.data(), inputs.size(), outputs.data(), outputs.size());
        if (result != 0) {
            set_error(error_code_trampoline(auxiliary), error_msg_trampoline(auxiliary));
        }
        return result;
    }

    int end_decompress_many_impl(compat::span<const pressio_data* const> const& inputs,
                                     compat::span<const pressio_data* const> const& outputs, int rc) override {
        int result = end_decompress_many_impl_trampoline(auxiliary, inputs.data(), inputs.size(), outputs.data(), outputs.size(), rc);
        if (result != 0) {
            set_error(error_code_trampoline(auxiliary), error_msg_trampoline(auxiliary));
        }
        return result;
    }

    int view_segment_impl(pressio_data const* data, const char* segment_id) override {
        int result = view_segment_impl_trampoline(auxiliary, data, segment_id);
        if (result != 0) {
            set_error(error_code_trampoline(auxiliary), error_msg_trampoline(auxiliary));
        }
        return result;
    }

  const char* prefix() const override {
      return prefix_trampoline.c_str();
  }

  pressio_options get_metrics_results(pressio_options const &) override {
      struct pressio_options * options_ptr = get_metrics_results_trampoline(auxiliary);
      struct pressio_options options = pressio_options(*options_ptr);
      delete options_ptr;
      return options;
  }

  std::unique_ptr<libpressio_metrics_plugin> clone() override {
    return compat::make_unique<trampoline_metrics_plugin>(
        clone_trampoline(auxiliary),
        get_configuration_impl_trampoline,
        get_documentation_impl_trampoline,
        get_options_trampoline,
        set_options_trampoline,
        begin_check_options_impl_trampoline,
        end_check_options_impl_trampoline,
        begin_get_documentation_impl_trampoline,
        end_get_documentation_impl_trampoline,
        begin_get_configuration_impl_trampoline,
        end_get_configuration_impl_trampoline,
        begin_get_options_impl_trampoline,
        end_get_options_impl_trampoline,
        begin_set_options_impl_trampoline,
        end_set_options_impl_trampoline,
        begin_compress_impl_trampoline,
        end_compress_impl_trampoline,
        begin_decompress_impl_trampoline,
        end_decompress_impl_trampoline,
        begin_compress_many_impl_trampoline,
        end_compress_many_impl_trampoline,
        begin_decompress_many_impl_trampoline,
        end_decompress_many_impl_trampoline,
        view_segment_impl_trampoline,
        prefix_trampoline.c_str(),
        get_metrics_results_trampoline,
        error_code_trampoline,
        error_msg_trampoline,
        clone_trampoline,
        release_trampoline
    );
  }

  ~trampoline_metrics_plugin(){
      release_trampoline(auxiliary);
  }

  void* auxiliary;
  struct pressio_options* (*get_configuration_impl_trampoline)(void const *);
  struct pressio_options* (*get_documentation_impl_trampoline)(void const *);
  struct pressio_options* (*get_options_trampoline)(void const *);
  int (*set_options_trampoline)(void*, struct pressio_options const *);
  int (*begin_check_options_impl_trampoline)(void*, struct pressio_options const *);
  int (*end_check_options_impl_trampoline)(void*, struct pressio_options const *, int);
  int (*begin_get_documentation_impl_trampoline)(void*);
  int (*end_get_documentation_impl_trampoline)(void*, struct pressio_options const *);
  int (*begin_get_configuration_impl_trampoline)(void*);
  int (*end_get_configuration_impl_trampoline)(void*, struct pressio_options const *);
  int (*begin_get_options_impl_trampoline)(void*);
  int (*end_get_options_impl_trampoline)(void*, struct pressio_options const *);
  int (*begin_set_options_impl_trampoline)(void*, struct pressio_options const *);
  int (*end_set_options_impl_trampoline)(void*, struct pressio_options const *, int);
  int (*begin_compress_impl_trampoline)(void*, struct pressio_data const *, struct pressio_data const *);
  int (*end_compress_impl_trampoline)(void*, struct pressio_data const *, struct pressio_data const *, int);
  int (*begin_decompress_impl_trampoline)(void*, struct pressio_data const *, struct pressio_data const *);
  int (*end_decompress_impl_trampoline)(void*, struct pressio_data const *, struct pressio_data const *, int);
  int (*begin_compress_many_impl_trampoline)(void*, struct pressio_data const * const *, size_t, struct pressio_data const * const *, size_t);
  int (*end_compress_many_impl_trampoline)(void*, struct pressio_data const * const *, size_t, struct pressio_data const * const *, size_t, int);
  int (*begin_decompress_many_impl_trampoline)(void*, struct pressio_data const * const *, size_t, struct pressio_data const * const *, size_t);
  int (*end_decompress_many_impl_trampoline)(void*, struct pressio_data const * const *, size_t, struct pressio_data const * const *, size_t, int);
  int (*view_segment_impl_trampoline)(void*, struct pressio_data const *, const char *);
  std::string prefix_trampoline;
  struct pressio_options* (*get_metrics_results_trampoline)(void const *);
  int (*error_code_trampoline)(void const *);
  const char* (*error_msg_trampoline)(void const *);
  void* (*clone_trampoline)(void const *);
  void (*release_trampoline)(void*);
};

} } }

extern "C" {
    bool pressio_register_metric(
        struct pressio* library,
        void* auxiliary,
        struct pressio_options* (*get_configuration_impl_trampoline)(void const *),
        struct pressio_options* (*get_documentation_impl_trampoline)(void const *),
        struct pressio_options* (*get_options_trampoline)(void const *),
        int (*set_options_trampoline)(void*, struct pressio_options const *),
        int (*begin_check_options_impl_trampoline)(void*, struct pressio_options const *),
        int (*end_check_options_impl_trampoline)(void*, struct pressio_options const *, int),
        int (*begin_get_documentation_impl_trampoline)(void*),
        int (*end_get_documentation_impl_trampoline)(void*, struct pressio_options const *),
        int (*begin_get_configuration_impl_trampoline)(void*),
        int (*end_get_configuration_impl_trampoline)(void*, struct pressio_options const *),
        int (*begin_get_options_impl_trampoline)(void*),
        int (*end_get_options_impl_trampoline)(void*, struct pressio_options const *),
        int (*begin_set_options_impl_trampoline)(void*, struct pressio_options const *),
        int (*end_set_options_impl_trampoline)(void*, struct pressio_options const *, int),
        int (*begin_compress_impl_trampoline)(void*, struct pressio_data const *, struct pressio_data const *),
        int (*end_compress_impl_trampoline)(void*, struct pressio_data const *, struct pressio_data const *, int),
        int (*begin_decompress_impl_trampoline)(void*, struct pressio_data const *, struct pressio_data const *),
        int (*end_decompress_impl_trampoline)(void*, struct pressio_data const *, struct pressio_data const *, int),
        int (*begin_compress_many_impl_trampoline)(void*, struct pressio_data const * const *, size_t, struct pressio_data const * const *, size_t),
        int (*end_compress_many_impl_trampoline)(void*, struct pressio_data const * const *, size_t, struct pressio_data const * const *, size_t, int),
        int (*begin_decompress_many_impl_trampoline)(void*, struct pressio_data const * const *, size_t, struct pressio_data const * const *, size_t),
        int (*end_decompress_many_impl_trampoline)(void*, struct pressio_data const * const *, size_t, struct pressio_data const * const *, size_t, int),
        int (*view_segment_impl_trampoline)(void*, struct pressio_data const *, const char *),
        const char* prefix_trampoline,
        struct pressio_options* (*get_metrics_results_trampoline)(void const *),
        int (*error_code_trampoline)(void const *),
        const char* (*error_msg_trampoline)(void const *),
        void* (*clone_trampoline)(void const *),
        void (*release_trampoline)(void*)
    ) {
        return libpressio::metrics_plugins().regsiter_factory(prefix_trampoline, [=](){
            return compat::make_unique<libpressio::metrics::trampoline_ns::trampoline_metrics_plugin>(
                clone_trampoline(auxiliary),
                get_configuration_impl_trampoline,
                get_documentation_impl_trampoline,
                get_options_trampoline,
                set_options_trampoline,
                begin_check_options_impl_trampoline,
                end_check_options_impl_trampoline,
                begin_get_documentation_impl_trampoline,
                end_get_documentation_impl_trampoline,
                begin_get_configuration_impl_trampoline,
                end_get_configuration_impl_trampoline,
                begin_get_options_impl_trampoline,
                end_get_options_impl_trampoline,
                begin_set_options_impl_trampoline,
                end_set_options_impl_trampoline,
                begin_compress_impl_trampoline,
                end_compress_impl_trampoline,
                begin_decompress_impl_trampoline,
                end_decompress_impl_trampoline,
                begin_compress_many_impl_trampoline,
                end_compress_many_impl_trampoline,
                begin_decompress_many_impl_trampoline,
                end_decompress_many_impl_trampoline,
                view_segment_impl_trampoline,
                prefix_trampoline,
                get_metrics_results_trampoline,
                error_code_trampoline,
                error_msg_trampoline,
                clone_trampoline,
                release_trampoline
            );
        });
    }
}
