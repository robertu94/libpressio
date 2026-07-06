#include <cstdlib>

#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/options.h"
#include "pressio_compressor.h"
#include "std_compat/memory.h"

namespace libpressio { namespace compressors { namespace trampoline_ns {

class trampoline_plugin: public libpressio_compressor_plugin {
  public:

      trampoline_plugin(
          void* auxiliary,
          struct pressio_options* (*get_configuration_impl_trampoline)(void const *),
          struct pressio_options* (*get_documentation_impl_trampoline)(void const *),
          struct pressio_options* (*get_options_impl_trampoline)(void const *),
          int (*check_options_impl_trampoline)(void*, struct pressio_options const *),
          int (*set_options_impl_trampoline)(void*, struct pressio_options const *),
          int (*compress_impl_trampoline)(void*, struct pressio_data const *, struct pressio_data*),
          int (*decompress_impl_trampoline)(void*, struct pressio_data const *, struct pressio_data*),
          int (*compress_many_impl_trampoline)(void*, struct pressio_data const * const *, size_t, struct pressio_data * *, size_t),
          int (*decompress_many_impl_trampoline)(void*, struct pressio_data const * const *, size_t, struct pressio_data * *, size_t),
          int major_version_trampoline,
          int minor_version_trampoline,
          int patch_version_trampoline,
          int revision_version_trampoline,
          const char* version_trampoline,
          const char* prefix_trampoline,
          struct pressio_options* (*get_metrics_results_impl_trampoline)(void const *),
          int (*error_code_trampoline)(void const *),
          const char* (*error_msg_trampoline)(void const *),
          void* (*clone_trampoline)(void const *),
          void (*release_trampoline)(void*)
      ):
        auxiliary(auxiliary),
        get_configuration_impl_trampoline(get_documentation_impl_trampoline),
        get_documentation_impl_trampoline(get_documentation_impl_trampoline),
        get_options_impl_trampoline(get_options_impl_trampoline),
        check_options_impl_trampoline(check_options_impl_trampoline),
        set_options_impl_trampoline(set_options_impl_trampoline),
        compress_impl_trampoline(compress_impl_trampoline),
        decompress_impl_trampoline(decompress_impl_trampoline),
        compress_many_impl_trampoline(compress_many_impl_trampoline),
        decompress_many_impl_trampoline(decompress_many_impl_trampoline),
        major_version_trampoline(major_version_trampoline),
        minor_version_trampoline(minor_version_trampoline),
        patch_version_trampoline(patch_version_trampoline),
        revision_version_trampoline(revision_version_trampoline),
        version_trampoline(version_trampoline),
        prefix_trampoline(prefix_trampoline),
        get_metrics_results_impl_trampoline(get_metrics_results_impl_trampoline),
        error_code_trampoline(error_code_trampoline),
        error_msg_trampoline(error_msg_trampoline),
        clone_trampoline(clone_trampoline),
        release_trampoline(release_trampoline)
      {}

  struct pressio_options get_configuration_impl() const override {
      return pressio_options(*get_configuration_impl_trampoline(auxiliary));
  }

  struct pressio_options get_documentation_impl() const override {
      return pressio_options(*get_documentation_impl_trampoline(auxiliary));
  }

  struct pressio_options get_options_impl() const override {
      return pressio_options(*get_options_impl_trampoline(auxiliary));
  }

  int check_options_impl(struct pressio_options const & options) {
      int result = check_options_impl_trampoline(auxiliary, &options);
      if (result != 0) {
          set_error(error_code_trampoline(auxiliary), error_msg_trampoline(auxiliary));
      }
      return result;
  }

  int set_options_impl(struct pressio_options const& options) override {
      int result = set_options_impl_trampoline(auxiliary, &options);
      if (result != 0) {
          set_error(error_code_trampoline(auxiliary), error_msg_trampoline(auxiliary));
      }
      return result;
  }

  int compress_impl(const pressio_data *input, struct pressio_data* output) override {
      int result = compress_impl_trampoline(auxiliary, input, output);
      if (result != 0) {
          set_error(error_code_trampoline(auxiliary), error_msg_trampoline(auxiliary));
      }
      return result;
  }

  int decompress_impl(const pressio_data *input, struct pressio_data* output) override {
      int result = decompress_impl_trampoline(auxiliary, input, output);
      if (result != 0) {
          set_error(error_code_trampoline(auxiliary), error_msg_trampoline(auxiliary));
      }
      return result;
  }

  int compress_many_impl(compat::span<const pressio_data* const> const& inputs, compat::span<pressio_data*> & outputs) {
      int result = compress_many_impl_trampoline(auxiliary, inputs.data(), inputs.size(), outputs.data(), outputs.size());
      if (result != 0) {
          set_error(error_code_trampoline(auxiliary), error_msg_trampoline(auxiliary));
      }
      return result;
    }

  int decompress_many_impl(compat::span<const pressio_data* const> const& inputs, compat::span<pressio_data* >& outputs) {
      int result = decompress_many_impl_trampoline(auxiliary, inputs.data(), inputs.size(), outputs.data(), outputs.size());
      if (result != 0) {
          set_error(error_code_trampoline(auxiliary), error_msg_trampoline(auxiliary));
      }
      return result;
    }

  int major_version() const override {
      return major_version_trampoline;
  }
  int minor_version() const override {
      return minor_version_trampoline;
  }
  int patch_version() const override {
      return patch_version_trampoline;
  }
  int revision_version () const {
      return revision_version_trampoline;
  }

  const char* version() const override {
      return version_trampoline.c_str();
  }

  const char* prefix() const override {
      return prefix_trampoline.c_str();
  }

  pressio_options get_metrics_results_impl() const override {
      return pressio_options(*get_metrics_results_impl_trampoline(auxiliary));
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override {
    return compat::make_unique<trampoline_plugin>(
        clone_trampoline(auxiliary),
        get_configuration_impl_trampoline,
        get_documentation_impl_trampoline,
        get_options_impl_trampoline,
        check_options_impl_trampoline,
        set_options_impl_trampoline,
        compress_impl_trampoline,
        decompress_impl_trampoline,
        compress_many_impl_trampoline,
        decompress_many_impl_trampoline,
        major_version_trampoline,
        minor_version_trampoline,
        patch_version_trampoline,
        revision_version_trampoline,
        version_trampoline.c_str(),
        prefix_trampoline.c_str(),
        get_metrics_results_impl_trampoline,
        error_code_trampoline,
        error_msg_trampoline,
        clone_trampoline,
        release_trampoline
    );
  }

  ~trampoline_plugin(){
      release_trampoline(auxiliary);
  }

  void* auxiliary;
  struct pressio_options* (*get_configuration_impl_trampoline)(void const *);
  struct pressio_options* (*get_documentation_impl_trampoline)(void const *);
  struct pressio_options* (*get_options_impl_trampoline)(void const *);
  int (*check_options_impl_trampoline)(void*, struct pressio_options const *);
  int (*set_options_impl_trampoline)(void*, struct pressio_options const *);
  int (*compress_impl_trampoline)(void*, struct pressio_data const *, struct pressio_data*);
  int (*decompress_impl_trampoline)(void*, struct pressio_data const *, struct pressio_data*);
  int (*compress_many_impl_trampoline)(void*, struct pressio_data const * const *, size_t, struct pressio_data * *, size_t);
  int (*decompress_many_impl_trampoline)(void*, struct pressio_data const * const *, size_t, struct pressio_data * *, size_t);
  int major_version_trampoline;
  int minor_version_trampoline;
  int patch_version_trampoline;
  int revision_version_trampoline;
  std::string version_trampoline;
  std::string prefix_trampoline;
  struct pressio_options* (*get_metrics_results_impl_trampoline)(void const *);
  int (*error_code_trampoline)(void const *);
  const char* (*error_msg_trampoline)(void const *);
  void* (*clone_trampoline)(void const *);
  void (*release_trampoline)(void*);
};

} } }

extern "C" {
    bool pressio_register_compressor(
        struct pressio* library,
        void* auxiliary,
        struct pressio_options* (*get_configuration_impl_trampoline)(void const *),
        struct pressio_options* (*get_documentation_impl_trampoline)(void const *),
        struct pressio_options* (*get_options_impl_trampoline)(void const *),
        int (*check_options_impl_trampoline)(void*, struct pressio_options const *),
        int (*set_options_impl_trampoline)(void*, struct pressio_options const *),
        int (*compress_impl_trampoline)(void*, struct pressio_data const *, struct pressio_data*),
        int (*decompress_impl_trampoline)(void*, struct pressio_data const *, struct pressio_data*),
        int (*compress_many_impl_trampoline)(void*, struct pressio_data const * const *, size_t, struct pressio_data * *, size_t),
        int (*decompress_many_impl_trampoline)(void*, struct pressio_data const * const *, size_t, struct pressio_data * *, size_t),
        int major_version_trampoline,
        int minor_version_trampoline,
        int patch_version_trampoline,
        int revision_version_trampoline,
        const char* version_trampoline,
        const char* prefix_trampoline,
        struct pressio_options* (*get_metrics_results_impl_trampoline)(void const *),
        int (*error_code_trampoline)(void const *),
        const char* (*error_msg_trampoline)(void const *),
        void* (*clone_trampoline)(void const *),
        void (*release_trampoline)(void*)
    ) {
        return libpressio::compressor_plugins().regsiter_factory(prefix_trampoline, [=](){
            return compat::make_unique<libpressio::compressors::trampoline_ns::trampoline_plugin>(
                clone_trampoline(auxiliary),
                get_configuration_impl_trampoline,
                get_documentation_impl_trampoline,
                get_options_impl_trampoline,
                check_options_impl_trampoline,
                set_options_impl_trampoline,
                compress_impl_trampoline,
                decompress_impl_trampoline,
                compress_many_impl_trampoline,
                decompress_many_impl_trampoline,
                major_version_trampoline,
                minor_version_trampoline,
                patch_version_trampoline,
                revision_version_trampoline,
                version_trampoline,
                prefix_trampoline,
                get_metrics_results_impl_trampoline,
                error_code_trampoline,
                error_msg_trampoline,
                clone_trampoline,
                release_trampoline
            );
        });
    }
}
