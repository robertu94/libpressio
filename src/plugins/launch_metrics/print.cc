#include <libpressio_ext/launch/external_launch_metrics.h>
#include <std_compat/memory.h>
#include <iostream>
#include <iomanip>


namespace libpressio { namespace launch_metrics_print {
struct libpressio_launch_metrics_print_plugin : public libpressio_launch_metrics_plugin {
  virtual void launch_begin(std::vector<std::string> const& args) const {
      for (auto const& i : args) {
          std::cout << i << ' ';
      }
      std::cout << std::endl;
      return;
  }
  virtual void launch_end(std::vector<std::string> const&, extern_proc_results const& ret) const {
      std::cout << "stdout " <<  std::quoted(ret.proc_stdout) << std::endl;
      std::cout << "stderr " << std::quoted(ret.proc_stderr) << std::endl;
      std::cout << "error " << ret.error_code << std::endl;
      std::cout << "return " << ret.return_code << std::endl;
      return;
  }
  virtual std::unique_ptr<libpressio_launch_metrics_plugin> clone() const  {
      return compat::make_unique<libpressio_launch_metrics_print_plugin>(*this);
  }

  const char* prefix() const {
      return "print";
  }
};

static pressio_register launch_metrics_print_plugin(launch_metrics_plugins(), "print", [](){ return compat::make_unique<libpressio_launch_metrics_print_plugin>();});

}}

