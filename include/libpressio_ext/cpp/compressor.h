#ifndef LIBPRESSIO_COMPRESSOR_IMPL_H
#define LIBPRESSIO_COMPRESSOR_IMPL_H
#include <string>
#include <memory>
#include "options.h"
#include "metrics.h"
#include "configurable.h"
#include "versionable.h"
#include "errorable.h"

/*!\file 
 * \brief an extension header for adding compressor plugins to libpressio
 */

struct pressio_data;
struct pressio_options;
class libpressio_metrics_plugin;

/**
 * plugin to provide a new compressor
 */
class libpressio_compressor_plugin :public pressio_configurable, public pressio_versionable, public pressio_errorable {
  public:

  libpressio_compressor_plugin() noexcept;
  /**
   * copy construct a compressor plugin by cloning the plugin
   * \param[in] plugin the plugin to clone
   */
  libpressio_compressor_plugin(libpressio_compressor_plugin const& plugin):
    pressio_configurable(plugin),
    pressio_errorable(plugin),
    metrics_plugin(plugin.metrics_plugin->clone())
  {}
  /**
   * copy assign a compressor plugin by cloning the plugin
   * \param[in] plugin the plugin to clone
   */
  libpressio_compressor_plugin& operator=(libpressio_compressor_plugin const& plugin)
  {
    pressio_configurable::operator=(plugin);
    pressio_errorable::operator=(plugin);
    metrics_plugin = plugin.metrics_plugin->clone();
    return *this;
  }
  /**
   * move construct a compressor plugin by cloning the plugin
   * \param[in] plugin the plugin to clone
   */
  libpressio_compressor_plugin(libpressio_compressor_plugin&& plugin) noexcept:
    pressio_configurable(plugin),
    pressio_errorable(plugin),
    metrics_plugin(std::move(plugin.metrics_plugin))
    {}
  /**
   * move assign a compressor plugin by cloning the plugin
   * \param[in] plugin the plugin to clone
   */
  libpressio_compressor_plugin& operator=(libpressio_compressor_plugin&& plugin) noexcept
  {
    pressio_configurable::operator=(plugin);
    pressio_errorable::operator=(plugin);
    metrics_plugin = std::move(plugin.metrics_plugin);
    return *this;
  }


  /** compressor should free their global memory in the destructor */
  virtual ~libpressio_compressor_plugin();

  /** get a set of metrics options available for the compressor.
   *
   * \see pressio_metrics_set_options for metrics options
   */
  struct pressio_options get_metrics_options() const;

  /** get a set of options available for the configurable object.
   *
   * The compressor should set a value if they have been set as default
   * The compressor should set a "reset" value if they are required to be set, but don't have a meaningful default
   *
   * \see pressio_compressor_get_options for the semantics this function should obey
   * \see pressio_options_clear to set a "reset" value
   * \see pressio_options_set_integer to set an integer value
   * \see pressio_options_set_double to set an double value
   * \see pressio_options_set_userptr to set an data value, include an \c include/ext/\<my_plugin\>.h to define the structure used
   * \see pressio_options_set_string to set a string value
   */
  struct pressio_options get_options() const override final;

  /**
   * checks the options for a compresor, handles metrics calls
   *
   * \see check_options_impl for the actual functions to call
   */
  int check_options(struct pressio_options const& options) override final;

  /** get the compile time configuration of a compressor, handles metrics calls
   *
   * \see pressio_compressor_get_configuration for the semantics this function should obey
   */
  struct pressio_options get_configuration() const override final;
  /** sets a set of options for the compressor, handles metrics calls
   * \param[in] options to set for configuration of the compressor
   * \see pressio_compressor_set_options for the semantics this function should obey
   */
  int set_options(struct pressio_options const& options) override final;

  /** sets a set of metrics options for the compressor 
   * \param[in] options to set for configuration of the metrics
   * \see pressio_metrics_set_options for the semantics this function should obey
   */
  int set_metrics_options(struct pressio_options const& options);

  /** compresses a pressio_data buffer
   * \see pressio_compressor_compress for the semantics this function should obey
   */
  int compress(struct pressio_data const*input, struct pressio_data* output);
  /** decompress a pressio_data buffer
   * \see pressio_compressor_decompress for the semantics this function should obey
   */
  int decompress(struct pressio_data const*input, struct pressio_data* output);
  /**
   * \returns a pressio_options structure containing the metrics returned by the provided metrics plugin
   * \see libpressio_metricsplugin for how to compute results
   */
  struct pressio_options get_metrics_results() const;
  /**
   * \returns the configured libpressio_metricsplugin plugin
   */
  struct pressio_metrics get_metrics() const;
  /**
   * \param[in] plugin the configured libpressio_metricsplugin plugin to use
   */
  void set_metrics(struct pressio_metrics& plugin);


  /**
   * \returns a copy of each a compressor and its configuration.  If the
   * compressor is not thread-safe and indicates such via its configuration, it
   * may return a new shared pointer to the same object.  For this reason, this
   * function may not be const.
   */
  virtual std::shared_ptr<libpressio_compressor_plugin> clone()=0;




  protected:
  /** get a set of options available for the compressor.
   *
   * The compressor should set a value if they have been set as default
   * The compressor should set a "reset" value if they are required to be set, but don't have a meaningful default
   *
   * \see pressio_compressor_get_options for the semantics this function should obey
   * \see pressio_options_clear to set a "reset" value
   * \see pressio_options_set_integer to set an integer value
   * \see pressio_options_set_double to set an double value
   * \see pressio_options_set_userptr to set an data value, include an \c include/ext/\<my_plugin\>.h to define the structure used
   * \see pressio_options_set_string to set a string value
   */
  virtual struct pressio_options get_options_impl() const=0;
  /** get a set of compile-time configurations for the compressor.
   *
   * \see pressio_compressor_get_configuration for the semantics this function should obey
   */
  virtual struct pressio_options get_configuration_impl() const=0;
  /** sets a set of options for the compressor 
   * \param[in] options to set for configuration of the compressor
   * \see pressio_compressor_set_options for the semantics this function should obey
   */
  virtual int set_options_impl(struct pressio_options const& options)=0;
  /** compresses a pressio_data buffer
   * \see pressio_compressor_compress for the semantics this function should obey
   */
  virtual int compress_impl(const pressio_data *input, struct pressio_data* output)=0;
  /** decompress a pressio_data buffer
   * \see pressio_compressor_decompress for the semantics this function should obey
   */
  virtual int decompress_impl(const pressio_data *input, struct pressio_data* output)=0;

  /** checks for extra arguments set for the compressor.
   * Unlike other functions, this option is NOT required
   *
   * \see pressio_compressor_check_options for semantics this function obeys
   * */
  virtual int check_options_impl(struct pressio_options const&);

  private:
  pressio_metrics metrics_plugin;
};

/**
 * wrapper for the compressor to use in C
 */
struct pressio_compressor {
  /**
   * \param[in] impl a newly constructed compressor plugin
   */
  pressio_compressor(std::shared_ptr<libpressio_compressor_plugin>&& impl): plugin(std::forward<std::shared_ptr<libpressio_compressor_plugin>>(impl)) {}
  /**
   * defaults constructs a compressor with a nullptr
   */
  pressio_compressor()=default;
  /**
   * copy constructs a compressor from another pointer by cloning
   */
  pressio_compressor(pressio_compressor const& compressor):
    plugin(compressor->clone()) {}
  /**
   * copy assigns a compressor from another pointer by cloning
   */
  pressio_compressor& operator=(pressio_compressor const& compressor) {
    if(&compressor == this) return *this;
    plugin = compressor->clone();
    return *this;
  }
  /**
   * move assigns a compressor from another pointer
   */
  pressio_compressor& operator=(pressio_compressor&& compressor) noexcept=default;
  /**
   * move constructs a compressor from another pointer
   */
  pressio_compressor(pressio_compressor&& compressor)=default;

  /** \returns true if the plugin is set */
  operator bool() const {
    return bool(plugin);
  }

  /** make libpressio_compressor_plugin behave like a shared_ptr */
  libpressio_compressor_plugin& operator*() const noexcept {
    return *plugin;
  }

  /** make libpressio_compressor_plugin behave like a shared_ptr */
  libpressio_compressor_plugin* operator->() const noexcept {
    return plugin.get();
  }

  /**
   * pointer to the implementation
   */
  std::shared_ptr<libpressio_compressor_plugin> plugin;
};

#endif
