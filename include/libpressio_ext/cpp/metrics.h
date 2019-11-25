#ifndef PRESIO_METRIC_PLUGIN
#define PRESIO_METRIC_PLUGIN

#include <memory>
#include <vector>

struct pressio_options;
struct pressio_data;

/*!\file 
 * \brief an extension header for adding metrics plugins to libpressio
 */

/**
 * plugin to collect metrics about compressors
 */
class libpressio_metrics_plugin {
  public:
  /**
   * destructor for inheritance
   */
  virtual ~libpressio_metrics_plugin()=default;
  /**
   * called at the beginning of check_options 
   * \param [in] options the value passed in to check_options
   */
  virtual void begin_check_options(struct pressio_options const* options);
  /**
   * called at the end of check_options 
   * \param [in] options the value passed in to check_options
   * \param [in] rc the return value from the underlying compressor check_options command
   */
  virtual void end_check_options(struct pressio_options const* options, int rc);
  /**
   * called at the beginning of get_options 
   */
  virtual void begin_get_options();
  /**
   * called at the end of get_options 
   * \param [in] ret the return value from the underlying compressor get_options command
   */
  virtual void end_get_options(struct pressio_options const* ret);

  /**
   * called at the beginning of get_configuration 
   */
  virtual void begin_get_configuration();
  /**
   * called at the end of get_configuration 
   * \param [in] ret the return value from the underlying compressor get_options command
   */
  virtual void end_get_configuration(struct pressio_options const& ret);
  /**
   * called at the beginning of set_options 
   * \param [in] options the value passed in to set_options
   */
  virtual void begin_set_options(struct pressio_options const& options);
  /**
   * called at the end of set_options 
   * \param [in] options the value passed in to set_options
   * \param [in] rc the return value from the underlying compressor set_options command
   */
  virtual void end_set_options(struct pressio_options const& options, int rc);

  /**
   * called at the beginning of compress 
   * \param [in] input the value passed in to compress
   * \param [in] output the value passed in to compress
   */
  virtual void begin_compress(const struct pressio_data * input, struct pressio_data const * output);
  /**
   * called at the end of compress 
   * \param [in] input the value passed in to compress
   * \param [in] output the value passed in to compress
   * \param [in] rc the return value from the underlying compressor compress command
   */
  virtual void end_compress(struct pressio_data const* input, pressio_data const * output, int rc);
  /**
   * called at the beginning of decompress 
   * \param [in] input the value passed in to decompress
   * \param [in] output the value passed in to decompress
   */
  virtual void begin_decompress(struct pressio_data const* input, pressio_data const* output);
  /**
   * called at the end of decompress 
   * \param [in] input the value passed in to decompress
   * \param [in] output the value passed in to decompress
   * \param [in] rc the return value from the underlying compressor decompress command
   */
  virtual void end_decompress(struct pressio_data const* input, pressio_data const* output, int rc);

  /**
   * \returns a pressio_options structure containing the metrics returned by the provided metrics plugin
   */
  virtual struct pressio_options get_metrics_results() const=0;

  /**
   * \returns a pressio_options structure containing the options for the provided metrics plugin
   */
  virtual struct pressio_options get_metrics_options() const;

  /**
   * \param[in] options a pressio_options structure containing the options for the provided metrics plugin
   */
  virtual int set_metrics_options(struct pressio_options const& options);
};

/**
 * C compatible pointer to metrics_plugins
 */
struct pressio_metrics {

  /** construct a metrics wrapper*/
  pressio_metrics(std::unique_ptr<libpressio_metrics_plugin>&& metrics): plugin(std::move(metrics)) {}

  /** allow access to underlying plugin*/
  libpressio_metrics_plugin* operator->() const noexcept {return plugin.get();}
  /** allow access to underlying plugin*/
  libpressio_metrics_plugin& operator*() const noexcept {return *plugin;}
  /** returns true if the pointer is not nullptr */
  operator bool() const { return plugin.get() != nullptr; }

  private:
  std::shared_ptr<libpressio_metrics_plugin> plugin;
};

/**
 * returns a composite metrics plugin from a vector of metrics_plugins
 */
std::unique_ptr<libpressio_metrics_plugin> make_m_composite(std::vector<std::unique_ptr<libpressio_metrics_plugin>>&& plugins); 


#endif
