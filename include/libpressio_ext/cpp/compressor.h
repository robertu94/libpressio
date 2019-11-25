#ifndef LIBPRESSIO_COMPRESSOR_IMPL_H
#define LIBPRESSIO_COMPRESSOR_IMPL_H
#include <string>
#include <memory>
#include "metrics.h"

/*!\file 
 * \brief an extension header for adding compressor plugins to libpressio
 */

struct pressio_data;
struct pressio_options;
class libpressio_metrics_plugin;

/**
 * plugin to provide a new compressor
 */
class libpressio_compressor_plugin {
  public:

  libpressio_compressor_plugin() noexcept;

  /** compressor should free their global memory in the destructor */
  virtual ~libpressio_compressor_plugin();
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
  struct pressio_options get_options() const;

  /** get a set of metrics options available for the compressor.
   *
   * \see pressio_metrics_set_options for metrics options
   */
  struct pressio_options get_metrics_options() const;

  /** get the compile time configuration of a compressor
   *
   * \see pressio_compressor_get_configuration for the semantics this function should obey
   */
  struct pressio_options get_configuration() const;
  /** sets a set of options for the compressor 
   * \param[in] options to set for configuration of the compressor
   * \see pressio_compressor_set_options for the semantics this function should obey
   */
  int set_options(struct pressio_options const& options);

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

  /** get the prefix used by this compressor for options */
  virtual const char* prefix() const=0;
  /** get a version string for the compressor
   * \see pressio_compressor_version for the semantics this function should obey
   */
  virtual const char* version() const=0;
  /** get the major version, default version returns 0
   * \see pressio_compressor_major_version for the semantics this function should obey
   */
  virtual int major_version() const;
  /** get the minor version, default version returns 0
   * \see pressio_compressor_minor_version for the semantics this function should obey
   */
  virtual int minor_version() const;
  /** get the patch version, default version returns 0
   * \see pressio_compressor_patch_version for the semantics this function should obey
   */
  virtual int patch_version() const;

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

  /** get the error message for the last error
   * \returns an implementation specific c-style error message for the last error
   */
  const char* error_msg() const;
  /** get the error code for the last error
   * \returns an implementation specific integer error code for the last error, 0 is reserved for no error
   */
  int error_code() const;

  /** checks for extra arguments set for the compressor.
   * the default verison just checks for unknown options passed in.
   *
   * \see pressio_compressor_check_options for semantics this function obeys
   * */
  int check_options(struct pressio_options const&);

  protected:
  /**
   * Should be used by implementing plug-ins to provide error codes
   * \param[in] code a implementation specific code for the last error
   * \param[in] msg a implementation specific message for the last error
   * \returns the value passed to code
   */
  int set_error(int code, std::string const& msg);

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
  struct {
    int code;
    std::string msg;
  } error;
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
   * move constructs a compressor from another pointer
   */
  pressio_compressor(pressio_compressor&& compressor)=default;
  /**
   * move assigns a compressor from another pointer
   */
  pressio_compressor& operator=(pressio_compressor&& compressor)=default;

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
