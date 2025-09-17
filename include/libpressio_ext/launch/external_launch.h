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
#include "external_launch_metrics.h"


namespace libpressio { namespace launch {

/**
 * an extension point for launching processes
 */
struct libpressio_launch_plugin: public pressio_configurable {
  virtual std::string type() const final {
      return "launch";
  }

  virtual ~libpressio_launch_plugin()=default;

  /**
   * launch the external command
   * \param[in] args the arguments to pass to the command
   */
  virtual extern_proc_results launch(std::vector<std::string> const& args) const final {
      metrics_plugin->launch_begin(args);
      auto ret = launch_impl(args);
      metrics_plugin->launch_end(args, ret);
      return ret;
  }

  void view_command(std::vector<std::string> const& args) const {
      metrics_plugin->view_command(args);
  }

  /**
   * launch the process
   *
   * \param[in] args the arguments to launch the process
   * \returns results of running the process
   */
  virtual extern_proc_results launch_impl(std::vector<std::string> const& args) const =0;

  /**
   * default set_options_impl implementation that has no options
   */
  virtual int set_options_impl(pressio_options const&) {
      return 0;
  }
  /**
   * default get_options_impl implementation that has no options
   */
  virtual pressio_options get_options_impl() const {
      return {};
  }
  /**
   * default get_configuratoin_impl implementation that has no options
   */
  virtual pressio_options get_configuration_impl() const {
      return {};
  }
  /**
   * return the name of the launch metric
   */
  std::string get_metrics_key_name() const {
    return std::string(prefix()) + ":launch_metric";
  }
  /**
   * common get_configuration options
   */
  pressio_options get_configuration() const final {
      pressio_options opts;
      set_meta_configuration(opts, "external:launch_metric", launch_metrics_plugins(), metrics_plugin);
      set_meta_configuration(opts, get_metrics_key_name(), launch_metrics_plugins(), metrics_plugin);
      opts.copy_from(get_configuration_impl());
      set(opts, "pressio:children", children());
      set(opts, "pressio:type", type());
      set(opts, "pressio:prefix", prefix());
      return opts;
  }
  /**
   * common get_options options
   */
  pressio_options get_options() const final {
      pressio_options opts;
      set_meta(opts, "external:launch_metric",  metrics_id, metrics_plugin);
      set_meta(opts, get_metrics_key_name(),  metrics_id, metrics_plugin);
      opts.copy_from(get_options_impl());
      return opts;
  }
  /**
   * common set_options options
   */
  int set_options(pressio_options const& opts) final {
      get_meta(opts, "external:launch_metric", launch_metrics_plugins(), metrics_id, metrics_plugin);
      get_meta(opts, get_metrics_key_name(), launch_metrics_plugins(), metrics_id, metrics_plugin);
      auto ret = set_options_impl(opts);
      return ret;
  }

  /**
   * common get documentation docs
   */
  pressio_options get_documentation() const final {
    pressio_options opts;
    set_meta_docs(opts, "external:launch_metric", "metrics to collect while launching an external process", metrics_plugin);
    set_meta_docs(opts, get_metrics_key_name(), "metrics to collect while launching an external process", metrics_plugin);
    opts.copy_from(get_documentation_impl());
    set(opts, "external:runtime", "the time the command takes to run");
    set(opts, "external:stderr", "the stderr from the external process, used for error reporting");
    set(opts, "external:return_code", "the return code from the external process if it was launched");
    set(opts, "external:error_code", "error code, indicates problems with launching processes");
    set(opts, "pressio:thread_safe", "level of thread safety provided by the compressor");
    set(opts, "pressio:stability", "level of stablity provided by the compressor; see the README for libpressio");
    set(opts, "pressio:type", R"(type of the libpressio meta object)");
    set(opts, "pressio:children", R"(children of this libpressio meta object)");
    set(opts, "pressio:prefix", R"(prefix of this libpressio meta object)");
    return opts;
  }

  /**
   * base set_name to set the name of the launch metric
   */
  void set_name(std::string const& new_name) final {
      pressio_configurable::set_name(new_name);
      if(new_name != "") {
          metrics_plugin->set_name(new_name +'/'+metrics_plugin->prefix());
      } else {
          metrics_plugin->set_name(new_name);
      }
  };

  /**
   * default method to return the children of this plugin
   */
  std::vector<std::string> children() const final {
	  return {
		  metrics_plugin->get_name()
	  };
  }

  /**
   * \returns documentation for this launch plugin
   */
  virtual pressio_options get_documentation_impl() const=0;

  /**
   * clones the launch method
   *
   * \returns a clone of the launcher
   */
  virtual std::unique_ptr<libpressio_launch_plugin> clone() const = 0;

  private:
  /**
   * ID of the launch metrics plugin in use
   */
  std::string metrics_id = "noop";
  /**
   * the launch metrics plugin
   */
  pressio_launcher_metrics metrics_plugin = launch_metrics_plugins().build("noop");
};
}
/**
 * the registry for launch plugins
 */
pressio_registry<std::unique_ptr<launch::libpressio_launch_plugin>>& launch_plugins();
}

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
  pressio_launcher(std::unique_ptr<libpressio::launch::libpressio_launch_plugin>&& ptr): plugin(std::move(ptr)) {}

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
  libpressio::launch::libpressio_launch_plugin& operator*() const noexcept {
    return *plugin;
  }

  /**
   * pressio_launcher are dereference-able
   */
  libpressio::launch::libpressio_launch_plugin* operator->() const noexcept {
    return plugin.get();
  }

  /**
   * the underlying plugin
   */
  std::unique_ptr<libpressio::launch::libpressio_launch_plugin> plugin;
};


#endif /* end of include guard: LIBPRESSIO_EXTERNAL_LAUNCH_H */
