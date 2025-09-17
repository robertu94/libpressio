#include <libpressio_ext/launch/external_launch_metrics.h>
#include <std_compat/memory.h>

namespace libpressio {
pressio_registry<std::unique_ptr<libpressio::launch_metrics::libpressio_launch_metrics_plugin>>& launch_metrics_plugins() {
  static pressio_registry<std::unique_ptr<libpressio::launch_metrics::libpressio_launch_metrics_plugin>> registry;
  return registry;
}
}

namespace libpressio { namespace launch_metrics { namespace noop_ns {
struct libpressio_launch_metrics_noop_plugin : public libpressio_launch_metrics_plugin {
  virtual void launch_begin(std::vector<std::string> const&) const {
      return;
  }
  virtual void launch_end(std::vector<std::string> const&, extern_proc_results const&) const {
      return;
  }
  virtual std::unique_ptr<libpressio_launch_metrics_plugin> clone() const  {
      return compat::make_unique<libpressio_launch_metrics_noop_plugin>(*this);
  }

  const char* prefix() const {
      return "noop";
  }
};

pressio_register registration(launch_metrics_plugins(), "noop", [](){ return compat::make_unique<libpressio_launch_metrics_noop_plugin>();});

}}}
