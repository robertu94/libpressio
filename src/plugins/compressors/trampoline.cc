// #include <cstdlib>

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
          struct pressio_options* (*get_configuration_impl_trampoline)(void*),
          struct pressio_options* (*get_documentation_impl_trampoline)(void*),
          struct pressio_options* (*get_options_impl_trampoline)(void*),
          int (*set_options_impl_trampoline)(void*, struct pressio_options const *),
          int (*compress_impl_trampoline)(void*, const struct pressio_data*, struct pressio_data*),
          int (*decompress_impl_trampoline)(void*, const struct pressio_data*, struct pressio_data*),
          int major_version_trampoline,
          int minor_version_trampoline,
          int patch_version_trampoline,
          int revision_version_trampoline,
          const char* version_trampoline,
          const char* prefix_trampoline,
          struct pressio_options* (*get_metrics_results_impl_trampoline)(void*),
          void* (*clone_trampoline)(void*),
          void (*release_trampoline)(void*)
      ):
        auxiliary(auxiliary),
        get_configuration_impl_trampoline(get_documentation_impl_trampoline),
        get_documentation_impl_trampoline(get_documentation_impl_trampoline),
        get_options_impl_trampoline(get_options_impl_trampoline),
        set_options_impl_trampoline(set_options_impl_trampoline),
        compress_impl_trampoline(compress_impl_trampoline),
        decompress_impl_trampoline(decompress_impl_trampoline),
        major_version_trampoline(major_version_trampoline),
        minor_version_trampoline(minor_version_trampoline),
        patch_version_trampoline(patch_version_trampoline),
        revision_version_trampoline(revision_version_trampoline),
        version_trampoline(version_trampoline),
        prefix_trampoline(prefix_trampoline),
        get_metrics_results_impl_trampoline(get_metrics_results_impl_trampoline),
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

  int set_options_impl(struct pressio_options const& options) override {
      return set_options_impl_trampoline(auxiliary, &options);
  }

  int compress_impl(const pressio_data *input, struct pressio_data* output) override {
      return compress_impl_trampoline(auxiliary, input, output);
  }

  int decompress_impl(const pressio_data *input, struct pressio_data* output) override {
      return decompress_impl_trampoline(auxiliary, input, output);
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
        set_options_impl_trampoline,
        compress_impl_trampoline,
        decompress_impl_trampoline,
        major_version_trampoline,
        minor_version_trampoline,
        patch_version_trampoline,
        revision_version_trampoline,
        version_trampoline.c_str(),
        prefix_trampoline.c_str(),
        get_metrics_results_impl_trampoline,
        clone_trampoline,
        release_trampoline
    );
  }

  ~trampoline_plugin(){
      release_trampoline(auxiliary);
  }

  void* auxiliary;
  struct pressio_options* (*get_configuration_impl_trampoline)(void*);
  struct pressio_options* (*get_documentation_impl_trampoline)(void*);
  struct pressio_options* (*get_options_impl_trampoline)(void*);
  int (*set_options_impl_trampoline)(void*, struct pressio_options const *);
  int (*compress_impl_trampoline)(void*, const struct pressio_data*, struct pressio_data*);
  int (*decompress_impl_trampoline)(void*, const struct pressio_data*, struct pressio_data*);
  int major_version_trampoline;
  int minor_version_trampoline;
  int patch_version_trampoline;
  int revision_version_trampoline;
  std::string version_trampoline;
  std::string prefix_trampoline;
  struct pressio_options* (*get_metrics_results_impl_trampoline)(void*);
  void* (*clone_trampoline)(void*);
  void (*release_trampoline)(void*);
};

} } }

extern "C" {
    bool pressio_register_compressor(
        struct pressio* library,
        void* auxiliary,
        struct pressio_options* (*get_configuration_impl_trampoline)(void*),
        struct pressio_options* (*get_documentation_impl_trampoline)(void*),
        struct pressio_options* (*get_options_impl_trampoline)(void*),
        int (*set_options_impl_trampoline)(void*, struct pressio_options const *),
        int (*compress_impl_trampoline)(void*, const struct pressio_data*, struct pressio_data*),
        int (*decompress_impl_trampoline)(void*, const struct pressio_data*, struct pressio_data*),
        int major_version_trampoline,
        int minor_version_trampoline,
        int patch_version_trampoline,
        int revision_version_trampoline,
        const char* version_trampoline,
        const char* prefix_trampoline,
        struct pressio_options* (*get_metrics_results_impl_trampoline)(void*),
        void* (*clone_trampoline)(void*),
        void (*release_trampoline)(void*)
    ) {
        return libpressio::compressor_plugins().regsiter_factory(prefix_trampoline, [=](){
            return compat::make_unique<libpressio::compressors::trampoline_ns::trampoline_plugin>(
                clone_trampoline(auxiliary),
                get_configuration_impl_trampoline,
                get_documentation_impl_trampoline,
                get_options_impl_trampoline,
                set_options_impl_trampoline,
                compress_impl_trampoline,
                decompress_impl_trampoline,
                major_version_trampoline,
                minor_version_trampoline,
                patch_version_trampoline,
                revision_version_trampoline,
                version_trampoline,
                prefix_trampoline,
                get_metrics_results_impl_trampoline,
                clone_trampoline,
                release_trampoline
            );
        });
    }
}
