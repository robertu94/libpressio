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

#endif

#ifdef __cplusplus
}
#endif
