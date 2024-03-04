#ifndef PRESIO_METRIC_PLUGIN
#define PRESIO_METRIC_PLUGIN

#include <memory>
#include <vector>
#include "configurable.h"
#include <std_compat/span.h>


struct pressio_options;
struct pressio_data;

/*!\file 
 * \brief an extension header for adding metrics plugins to libpressio
 */

/**
 * plugin to collect metrics about compressors
 */
class libpressio_metrics_plugin : public pressio_configurable {
  public:
  std::string type() const final {
      return "metric";
  }

  /**
   * base configuration implementation provides type, and children
   */
  pressio_options get_configuration() const final;

  /**
   * default constructor
   */
  libpressio_metrics_plugin();
  /**
   * destructor for inheritance
   */
  virtual ~libpressio_metrics_plugin()=default;
  /**
 * called at the beginning of check_options
 * \param [in] options the value passed in to check_options
 */
  int begin_check_options(struct pressio_options const* options);
  /**
   * called at the end of check_options
   * \param [in] options the value passed in to check_options
   * \param [in] rc the return value from the underlying compressor check_options command
   */
  int end_check_options(struct pressio_options const* options, int rc);
  /**
   * called at the beginning of get_options
   */
  int begin_get_options();
  /**
   * called at the end of get_options
   * \param [in] ret the return value from the underlying compressor get_options command
   */
  int end_get_options(struct pressio_options const* ret);

  /**
   * called at the beginning of get_configuration
   */
  int begin_get_documentation();
  /**
   * called at the end of get_configuration
   * \param [in] ret the return value from the underlying compressor get_options command
   */
  int end_get_documentation(struct pressio_options const& ret);
  /**
   * called at the beginning of get_configuration
   */
  int begin_get_configuration();
  /**
   * called at the end of get_configuration
   * \param [in] ret the return value from the underlying compressor get_options command
   */
  int end_get_configuration(struct pressio_options const& ret);
  /**
   * called at the beginning of set_options
   * \param [in] options the value passed in to set_options
   */
  int begin_set_options(struct pressio_options const& options);
  /**
   * called at the end of set_options
   * \param [in] options the value passed in to set_options
   * \param [in] rc the return value from the underlying compressor set_options command
   */
  int end_set_options(struct pressio_options const& options, int rc);

  /**
   * called at the beginning of compress
   * \param [in] input the value passed in to compress
   * \param [in] output the value passed in to compress
   */
  int begin_compress(const struct pressio_data * input, struct pressio_data const * output);
  /**
   * called at the end of compress
   * \param [in] input the value passed in to compress
   * \param [in] output the value passed in to compress
   * \param [in] rc the return value from the underlying compressor compress command
   */
  int end_compress(struct pressio_data const* input, pressio_data const * output, int rc);
  /**
   * called at the beginning of decompress
   * \param [in] input the value passed in to decompress
   * \param [in] output the value passed in to decompress
   */
  int begin_decompress(struct pressio_data const* input, pressio_data const* output);
  /**
   * called at the end of decompress
   * \param [in] input the value passed in to decompress
   * \param [in] output the value passed in to decompress
   * \param [in] rc the return value from the underlying compressor decompress command
   */
  int end_decompress(struct pressio_data const* input, pressio_data const* output, int rc);

  /**
   * called at the beginning of compress_many
   */
  int begin_compress_many(compat::span<const pressio_data* const> const& inputs,
                                       compat::span<const pressio_data* const> const& outputs);

  /**
   * called at the end of compress_many
   */
  int end_compress_many(compat::span<const pressio_data* const> const& inputs,
                                     compat::span<const pressio_data* const> const& outputs, int rc);

  /**
   * called at the beginning of decompress_many
   */
  int begin_decompress_many(compat::span<const pressio_data* const> const& inputs,
                                         compat::span<const pressio_data* const> const& outputs);

  /**
   * called at the end of decompress_many
   */
  int end_decompress_many(compat::span<const pressio_data* const> const& inputs,
                                       compat::span<const pressio_data* const> const& outputs, int rc);

  /**
   * compressors that produce several distinct segments of data
   * may call this function to allow metrics libraries to present
   * this data to 3rd parties
   *
   * \param[in] data the data to expose
   * \param[in] segment_id an identifier for this segment
   */
  int view_segment(pressio_data const* data, const char* segment_id);

  /**
   * called by metrics implementations to retrieve documentation
   */
  virtual pressio_options get_documentation_impl() const=0;

  /**
   * get documentation for the metrics module
   */
  virtual pressio_options get_documentation() const final;

  /**
   * prohibit overriding set_name from child classes, override set_name_impl instead
   */
  void set_name(std::string const& new_name) override final;

  /**
   * \param options the metrics from the compressor plugin
   * \returns a pressio_options structure containing the metrics returned by the provided metrics plugin
   */
  virtual pressio_options get_metrics_results(pressio_options const &options)=0;

  /**
   * \returns a clone of the metric
   */
  virtual std::unique_ptr<libpressio_metrics_plugin> clone()=0;

protected:
  /**
   * called at the beginning of check_options 
   * \param [in] options the value passed in to check_options
   */
  virtual int begin_check_options_impl(struct pressio_options const* options);
  /**
   * called at the end of check_options 
   * \param [in] options the value passed in to check_options
   * \param [in] rc the return value from the underlying compressor check_options command
   */
  virtual int end_check_options_impl(struct pressio_options const* options, int rc);
  /**
   * called at the beginning of get_options 
   */
  virtual int begin_get_options_impl();
  /**
   * called at the end of get_options 
   * \param [in] ret the return value from the underlying compressor get_options command
   */
  virtual int end_get_options_impl(struct pressio_options const* ret);

  /**
   * called at the beginning of get_configuration 
   */
  virtual int begin_get_documentation_impl();
  /**
   * called at the end of get_configuration 
   * \param [in] ret the return value from the underlying compressor get_options command
   */
  virtual int end_get_documentation_impl(struct pressio_options const& ret);
  /**
   * called at the beginning of get_configuration 
   */
  virtual int begin_get_configuration_impl();
  /**
   * called at the end of get_configuration 
   * \param [in] ret the return value from the underlying compressor get_options command
   */
  virtual int end_get_configuration_impl(struct pressio_options const& ret);
  /**
   * called at the beginning of set_options 
   * \param [in] options the value passed in to set_options
   */
  virtual int begin_set_options_impl(struct pressio_options const& options);
  /**
   * called at the end of set_options 
   * \param [in] options the value passed in to set_options
   * \param [in] rc the return value from the underlying compressor set_options command
   */
  virtual int end_set_options_impl(struct pressio_options const& options, int rc);

  /**
   * called at the beginning of compress 
   * \param [in] input the value passed in to compress
   * \param [in] output the value passed in to compress
   */
  virtual int begin_compress_impl(const struct pressio_data * input, struct pressio_data const * output);
  /**
   * called at the end of compress 
   * \param [in] input the value passed in to compress
   * \param [in] output the value passed in to compress
   * \param [in] rc the return value from the underlying compressor compress command
   */
  virtual int end_compress_impl(struct pressio_data const* input, pressio_data const * output, int rc);
  /**
   * called at the beginning of decompress 
   * \param [in] input the value passed in to decompress
   * \param [in] output the value passed in to decompress
   */
  virtual int begin_decompress_impl(struct pressio_data const* input, pressio_data const* output);
  /**
   * called at the end of decompress 
   * \param [in] input the value passed in to decompress
   * \param [in] output the value passed in to decompress
   * \param [in] rc the return value from the underlying compressor decompress command
   */
  virtual int end_decompress_impl(struct pressio_data const* input, pressio_data const* output, int rc);

  /**
   * called at the beginning of compress_many
   */
  virtual int begin_compress_many_impl(compat::span<const pressio_data* const> const& inputs,
                                   compat::span<const pressio_data* const> const& outputs);

  /**
   * called at the end of compress_many
   */
  virtual int end_compress_many_impl(compat::span<const pressio_data* const> const& inputs,
                                   compat::span<const pressio_data* const> const& outputs, int rc);

  /**
   * called at the beginning of decompress_many
   */
  virtual int begin_decompress_many_impl(compat::span<const pressio_data* const> const& inputs,
                                   compat::span<const pressio_data* const> const& outputs);

  /**
   * called at the end of decompress_many
   */
  virtual int end_decompress_many_impl(compat::span<const pressio_data* const> const& inputs,
                                   compat::span<const pressio_data* const> const& outputs, int rc);

  /**
   * compressors that produce several distinct segments of data
   * may call this function to allow metrics libraries to present
   * this data to 3rd parties
   *
   * \param[in] data the data to expose
   * \param[in] segment_id an identifier for this segment
   */
  virtual int view_segment_impl(pressio_data const* data, const char* segment_id);

  /**
   * base no-op routine that can be overwritten to provide configuration options
   */
  virtual pressio_options get_configuration_impl() const;

//  virtual pressio_options get_metrics_results_impl(pressio_options const &options)=0;
};

/**
 * C compatible pointer to metrics_plugins
 */
struct pressio_metrics {

  /** construct a metrics wrapper*/
  pressio_metrics(std::unique_ptr<libpressio_metrics_plugin>&& metrics): plugin(std::move(metrics)) {}
  /** construct a metrics wrapper*/
  pressio_metrics(std::shared_ptr<libpressio_metrics_plugin>&& metrics): plugin(std::move(metrics)) {}

  /** allow default construction*/
  pressio_metrics()=default;
  /**
   * copy construct a metric from another pointer
   */
  pressio_metrics(pressio_metrics const& metrics): plugin(metrics->clone()) {}
  /**
   * move assigns a metric from another pointer
   */
  pressio_metrics& operator=(pressio_metrics const& metrics) {
    if(&metrics == this) return *this;
    plugin = metrics->clone();
    return *this;
  }
  /**
   * move construct a metric from another pointer
   */
  pressio_metrics(pressio_metrics&& metrics)=default;
  /**
   * move assigns a metric from another pointer
   */
  pressio_metrics& operator=(pressio_metrics&& metrics)=default;


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
std::unique_ptr<libpressio_metrics_plugin> make_m_composite(std::vector<pressio_metrics>&& plugins); 


#endif
