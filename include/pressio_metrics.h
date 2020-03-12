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

#endif

#ifdef __cplusplus
}
#endif
