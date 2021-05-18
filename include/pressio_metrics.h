/*!\file 
 * \brief Metrics facilities to introspect compressor functions, input, and output
 */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef PRESSIO_METRICS
#define PRESSIO_METRICS

/** \file */

struct pressio_metrics;
struct pressio_options;


/**
 * frees a metrics structure
 * \param[in] metrics the metrics structure to free
 */
void pressio_metrics_free(struct pressio_metrics* metrics);

/**
 * Gets the results from a metrics structure
 * \param[in] metrics the metrics structure to get results from
 * \returns a new pressio_options structure the metrics structure to get results from
 */
struct pressio_options* pressio_metrics_get_results(struct pressio_metrics const* metrics);

/**
 * Gets the options for a metrics structure
 * \param[in] metrics the metrics structure to get options for
 * \returns a new pressio_options structure with the options for the metrics
 */
struct pressio_options* pressio_metrics_get_options(struct pressio_metrics const* metrics);

/**
 * Gets the configuration for a metrics structure
 * \param[in] metrics the metrics structure to get configuration for
 * \returns a new pressio_options structure with the configuration for the metrics
 */
struct pressio_options* pressio_metrics_get_configuration(struct pressio_metrics const* metrics);

/**
 * Gets the documentation for a metrics structure
 * \param[in] metrics the metrics structure to get documentation for
 * \returns a new pressio_options structure with the documentation for the metrics
 */
struct pressio_options* pressio_metrics_get_documentation(struct pressio_metrics const* metrics);

/**
 * Gets the options for a metrics structure
 * \param[in] metrics the metrics structure to get options for
 * \param[in] options the to set
 * \returns 0 if successful, positive values on errors, negative values on warnings
 */
int pressio_metrics_set_options(struct pressio_metrics const* metrics, struct pressio_options const* options);

/**
 * Clones a pressio_metrics object and its configuration
 * \param[in] metrics the metrics object to clone
 * \returns a new reference to a pressio metrics object
 */
struct pressio_metrics* pressio_metrics_clone(struct pressio_metrics* metrics);


/**
 * Assign a new name to a metrics.  Names are used to prefix options in meta-metrics.
 *
 * sub-metrics will be renamed either by the of the sub-metricss prefix
 * or by the $prefix:name configuration option
 *
 * i.e. for some new_name and a metrics with prefix foo and submetricss
 * with prefixs "abc", "def", "ghi" respectively
 *
 * - if foo:names = ['one', 'two', 'three'], then the names will be `$new_name/one, $new_name/two $new_name/three
 * - otherwise the names will be $new_name/abc, $new_name/def, $new_name/ghi
 *
 * \param[in] metrics the metrics to get the name of
 * \param[in] new_name the name to set
 */
void pressio_metrics_set_name(struct pressio_metrics* metrics, const char* new_name);

/**
 * Get the name of a metrics
 * \param[in] metrics the metrics to get the name of
 * \returns a string with the metrics name. The string becomes invalid if
 *          the name is set_name is called.
 */
const char* pressio_metrics_get_name(struct pressio_metrics const* metrics);

/**
 * Evaluate a metric that operates on a data buffer and return a corresponding options structure
 *
 * If the metric does not support being called on data buffers, the returned object is undefined.
 *
 * \param[in] metrics the metrics object to invoke
 * \param[in] uncompressed the data before compression, if nullptr, begin/end_compress_impl will not be called
 * \param[in] compressed the data after compression, if nullptr, the metrics object MAY choose not to compute some or all metrics, but SHOULD compute as many as possible
 * \param[in] decompressed  the data after decompression, if nullptr, begin/end_decompress_impl will not be called
 *
 * \returns a new pressio_options as if pressio_metrics_get_results was called.
 */
struct pressio_options* pressio_metrics_evaluate(
    struct pressio_metrics* metrics,
    struct pressio_data const* uncompressed,
    struct pressio_data const* compressed,
    struct pressio_data const* decompressed
    );

#endif

#ifdef __cplusplus
}
#endif
