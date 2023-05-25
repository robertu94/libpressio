#ifndef LIBPRESSIO_EXTERNAL_LAUNCH_METRICS_H_DPRI7MVH
#define LIBPRESSIO_EXTERNAL_LAUNCH_METRICS_H_DPRI7MVH
#include "libpressio_ext/cpp/configurable.h"
#include "libpressio_ext/cpp/pressio.h"
#include <memory>


/**
 * error codes for extern_proc_results
 */
enum extern_proc_error_codes {
  /**the launch was successful */
  success=0, 
  /** there was a failure to create the pipe */
  pipe_error=1, 
  /** there was a failure to fork process */
  fork_error=2,
  /** there was a failure to exec the process */
  exec_error=3,
  /** there was a failure parsing the format */
  format_error=4
};

/**
 * results from launching a process
 */
struct extern_proc_results {
  /** stdout from the command */
  std::string proc_stdout; 
  /** stderr from the command */
  std::string proc_stderr;
  /** the return code from the external process */
  int return_code = 0; 
  /** used to report errors with run_command */
  int error_code = success;
};

struct libpressio_launch_metrics_plugin : public pressio_configurable {
  virtual void launch_begin(std::vector<std::string> const&) const {
      return;
  }
  virtual void launch_end(std::vector<std::string> const&, extern_proc_results const&) const {
      return;
  }
  virtual std::unique_ptr<libpressio_launch_metrics_plugin> clone() const = 0;

  std::string type() const final {
	  return "launchmetric";
  }
};

/**
 * wrapper for launching processes
 */
struct pressio_launcher_metrics {

  /**
   * launch methods are default constructible
   */
  pressio_launcher_metrics()=default;

  /**
   * launch methods are constructible from a unique_ptr to a plugin
   *
   * \param[in] ptr the pointer to move from
   */
  pressio_launcher_metrics(std::unique_ptr<libpressio_launch_metrics_plugin>&& ptr): plugin(std::move(ptr)) {}

  /**
   * launch methods are copy constructible and have the effect of cloning the plugin
   *
   * \param[in] launcher the launcher to clone
   */
  pressio_launcher_metrics(pressio_launcher_metrics const& launcher): plugin(launcher.plugin->clone()) {}

  /**
   * launch methods are move constructible
   *
   * \param[in] compressor the launcher to clone
   */
  pressio_launcher_metrics(pressio_launcher_metrics&& compressor)=default;

  /**
   * launch methods are move assignable
   *
   * \param[in] launcher the launcher to clone
   */
  pressio_launcher_metrics& operator=(pressio_launcher_metrics&& launcher) noexcept=default;

  /**
   * \returns true if the plugin is not a nullptr
   */
  operator bool() const {
    return bool(plugin);
  }

  /**
   * pressio_launcher are dereference-able
   */
  libpressio_launch_metrics_plugin& operator*() const noexcept {
    return *plugin;
  }

  /**
   * pressio_launcher are dereference-able
   */
  libpressio_launch_metrics_plugin* operator->() const noexcept {
    return plugin.get();
  }

  /**
   * the underlying plugin
   */
  std::unique_ptr<libpressio_launch_metrics_plugin> plugin;
};

pressio_registry<std::unique_ptr<libpressio_launch_metrics_plugin>>& launch_metrics_plugins();

#endif /* end of include guard: EXTERNAL_LAUNCH_METRICS_H_DPRI7MVH */
