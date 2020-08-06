#ifndef LIBPRESSIO_EXTERNAL_LAUNCH_H
#define LIBPRESSIO_EXTERNAL_LAUNCH_H



#include "libpressio_ext/cpp/configurable.h"
#include "libpressio_ext/cpp/errorable.h"
#include "libpressio_ext/cpp/pressio.h"
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
  virtual extern_proc_results launch(std::string const&, std::string const&) const =0;
};

pressio_registry<std::unique_ptr<libpressio_launch_plugin>>& launch_plugins();

#endif /* end of include guard: LIBPRESSIO_EXTERNAL_LAUNCH_H */
