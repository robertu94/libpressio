/** \file
 *  \brief provide constructor factory functions for built-in plugin types
 */
#ifndef LIBPRESSIO_PLUGINS_PUBLIC
#define LIBPRESSIO_PLUGINS_PUBLIC
#include <memory>
#include <vector>
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/metrics.h"
/** construct the sz plugin */
std::unique_ptr<libpressio_compressor_plugin> make_c_sz();
/** construct the zfp plugin */
std::unique_ptr<libpressio_compressor_plugin> make_c_zfp();
/** construct the time metrics plugin */
std::unique_ptr<libpressio_metrics_plugin> make_m_time();
/** construct the size metrics plugin */
std::unique_ptr<libpressio_metrics_plugin> make_m_size();
/** construct a composite metrics plugin 
 * \param[in] plugins the plugins to wrap
 * */
std::unique_ptr<libpressio_metrics_plugin> make_m_composite(std::vector<std::unique_ptr<libpressio_metrics_plugin>>&& plugins);
#endif

