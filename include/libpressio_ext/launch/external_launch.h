#ifndef LIBPRESSIO_EXTERNAL_LAUNCH_H
#define LIBPRESSIO_EXTERNAL_LAUNCH_H


/**
 * \file
 * \brief interface for external launch methods
 */


#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/configurable.h"
#include "libpressio_ext/cpp/errorable.h"
#include "libpressio_ext/cpp/pressio.h"
#include <memory>
#include <string>

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

/**
 * an extension point for launching processes
 */
struct libpressio_launch_plugin: public pressio_configurable, public pressio_errorable {
  virtual ~libpressio_launch_plugin()=default;
  /**
   * launch the process
   *
   * \param[in] args the arguments to launch the process
   * \returns results of running the process
   */
  virtual extern_proc_results launch(std::vector<std::string> const& args) const =0;
  /**
   * clones the launch method
   *
   * \returns a clone of the launcher
   */
  virtual std::unique_ptr<libpressio_launch_plugin> clone() const = 0;
};

/**
 * wrapper for launching processes
 */
struct pressio_launcher {

  /**
   * launch methods are default constructible
   */
  pressio_launcher()=default;

  /**
   * launch methods are constructible from a unique_ptr to a plugin
   *
   * \param[in] ptr the pointer to move from
   */
  pressio_launcher(std::unique_ptr<libpressio_launch_plugin>&& ptr): plugin(std::move(ptr)) {}

  /**
   * launch methods are copy constructible and have the effect of cloning the plugin
   *
   * \param[in] launcher the launcher to clone
   */
  pressio_launcher(pressio_launcher const& launcher): plugin(launcher.plugin->clone()) {}

  /**
   * launch methods are move constructible
   *
   * \param[in] compressor the launcher to clone
   */
  pressio_launcher(pressio_launcher&& compressor)=default;

  /**
   * launch methods are move assignable
   *
   * \param[in] launcher the launcher to clone
   */
  pressio_launcher& operator=(pressio_launcher&& launcher) noexcept=default;

  /**
   * \returns true if the plugin is not a nullptr
   */
  operator bool() const {
    return bool(plugin);
  }

  /**
   * pressio_launcher are dereference-able
   */
  libpressio_launch_plugin& operator*() const noexcept {
    return *plugin;
  }

  /**
   * pressio_launcher are dereference-able
   */
  libpressio_launch_plugin* operator->() const noexcept {
    return plugin.get();
  }

  /**
   * the underlying plugin
   */
  std::unique_ptr<libpressio_launch_plugin> plugin;
};

/**
 * the registry for launch plugins
 */
pressio_registry<std::unique_ptr<libpressio_launch_plugin>>& launch_plugins();

#endif /* end of include guard: LIBPRESSIO_EXTERNAL_LAUNCH_H */
