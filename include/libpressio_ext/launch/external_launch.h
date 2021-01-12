#ifndef LIBPRESSIO_EXTERNAL_LAUNCH_H
#define LIBPRESSIO_EXTERNAL_LAUNCH_H



#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/configurable.h"
#include "libpressio_ext/cpp/errorable.h"
#include "libpressio_ext/cpp/pressio.h"
#include <memory>
#include <string>

enum extern_proc_error_codes {
  success=0,
  pipe_error=1,
  fork_error=2,
  exec_error=3,
  format_error=4
};

struct extern_proc_results {
  std::string proc_stdout; //stdout from the command
  std::string proc_stderr; //stdin from the command
  int return_code = 0; //the return code from the external process
  int error_code = success; //used to report errors with run_command
};

struct libpressio_launch_plugin: public pressio_configurable, public pressio_errorable {
  virtual ~libpressio_launch_plugin()=default;
  virtual extern_proc_results launch(std::vector<std::string> const&) const =0;
  virtual std::unique_ptr<libpressio_launch_plugin> clone() const = 0;
};

struct pressio_launcher {

  pressio_launcher()=default;

  pressio_launcher(std::unique_ptr<libpressio_launch_plugin>&& ptr): plugin(std::move(ptr)) {}

  pressio_launcher(pressio_launcher const& launcher): plugin(launcher.plugin->clone()) {}

  pressio_launcher(pressio_launcher&& compressor)=default;

  pressio_launcher& operator=(pressio_launcher&& launcher) noexcept=default;

  operator bool() const {
    return bool(plugin);
  }

  libpressio_launch_plugin& operator*() const noexcept {
    return *plugin;
  }

  libpressio_launch_plugin* operator->() const noexcept {
    return plugin.get();
  }

  std::unique_ptr<libpressio_launch_plugin> plugin;
};

pressio_registry<std::unique_ptr<libpressio_launch_plugin>>& launch_plugins();

#endif /* end of include guard: LIBPRESSIO_EXTERNAL_LAUNCH_H */
