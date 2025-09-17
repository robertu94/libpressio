/**
 * \file
 * \brief C++ interface to the compressor loader
 */
#ifndef LIBPRESSIO_PRESSIO_IMPL_H
#define LIBPRESSIO_PRESSIO_IMPL_H
#include <memory>
#include "metrics.h"
#include "std_compat/language.h"

namespace libpressio {
    namespace compressors {
        class libpressio_compressor_plugin;
    }
    namespace io {
        struct libpressio_io_plugin;
    }
}

/**
 * forward declearation of pressio_register_all() to prevent circular dependency
 * between C and C++
 */
extern "C" void pressio_register_all();

namespace libpressio {
/**
 * the registry for compressor plugins
 */
pressio_registry<std::shared_ptr<libpressio::compressors::libpressio_compressor_plugin>>& compressor_plugins();
/**
 * the registry for metrics plugins
 */
pressio_registry<std::unique_ptr<libpressio::metrics::libpressio_metrics_plugin>>& metrics_plugins();

/**
 * the registry for metrics plugins
 */
pressio_registry<std::unique_ptr<libpressio::io::libpressio_io_plugin>>& io_plugins();
}

/**
 * the libraries basic state
 */
struct pressio {
  public:

  pressio() {
      pressio_register_all();
  }
  /**
   * sets an error code and message
   * \param[in] code non-zero represents an error
   * \param[in] msg a human readable description of the error
   */
  void set_error(int code, std::string const& msg) {
    error.code = code;
    error.msg = msg;
  }

  /**
   * \returns the last error code for this library object
   */
  int err_code() const { return error.code; }

  /**
   * \returns the last error message for this library object
   */
  std::string const& err_msg() const { return error.msg; }

  /**
   * Returns an instance of a compressor
   * \param[in] compressor_id name of the compressor to request
   * \returns an instance of compressor registered at name, or nullptr on error
   */
  std::shared_ptr<libpressio::compressors::libpressio_compressor_plugin> get_compressor(std::string const& compressor_id);

  /**
   * Returns an io module
   * \param[in] io_module_id name of the compressor to request
   * \returns an instance of compressor registered at name, or nullptr on error
   */
  std::shared_ptr<libpressio::io::libpressio_io_plugin> get_io(std::string const& io_module_id);
  

  /**
   * Returns an metrics module
   * \param[in] id name of the compressor to request
   * \returns an instance of compressor registered at name, or nullptr on error
   */
  template <class Str>
  std::unique_ptr<libpressio::metrics::libpressio_metrics_plugin> get_metric(Str id) {
    auto ret = libpressio::metrics_plugins().build(id);
    if(!ret) {
        set_error(2, std::string("failed to construct metrics plugin: ") + id);
        return nullptr;
    }
    return ret;
  }

  /**
   * Returns a composite metric for all the metric_ids requested
   * \param[in] first iterator to the first metric_id requested
   * \param[in] last iterator to the last metric_id requested
   * \returns an instance of a metrics module regsitered at a name wrapping it in a composite if nessisary
   */
  template <class ForwardIt>
  std::unique_ptr<libpressio::metrics::libpressio_metrics_plugin> get_metrics(ForwardIt first, ForwardIt last) {
    std::vector<pressio_metrics> plugins;

    for (auto metric = first; metric != last; ++metric) {
      plugins.emplace_back(libpressio::metrics_plugins().build(*metric));
      if(not plugins.back()) {
        set_error(2, std::string("failed to construct metrics plugin: ") + *metric);
        return nullptr;
      }
    }

    auto metrics = libpressio::metrics::make_m_composite(std::move(plugins));

    if(metrics) return metrics;
    else {
      set_error(3, "failed to construct composite metric");
      return nullptr;
    };
  }

  /**
   * \returns the version string for this version of libpressio
   *
   * \see pressio_version
   */
  static const char* version();

  /**
   * \returns the features string for this version of libpressio
   *
   * \see pressio_features
   */
  static const char* features();

  /**
   * \returns the supported compressors list for this version of libpressio
   *
   * \see pressio_supported_compressors
   */
  static const char* supported_compressors();

  /**
   * \returns the supported metrics list for this version of libpressio
   *
   * \see pressio_supported_metrics
   */
  static const char* supported_metrics();

  /**
   * \returns the supported io modules list for this version of libpressio
   *
   * \see pressio_supported_io
   */
  static const char* supported_io();

  /**
   * \returns the major version number of libpressio
   */
  static unsigned int major_version();

  /**
   * \returns the minor version number of libpressio
   */
  static unsigned int minor_version();

  /**
   * \returns the patch version number of libpressio
   */
  static unsigned int patch_version();

  private:
  struct {
    int code;
    std::string msg;
  } error;
};
#endif
