/**
 * \file
 * \brief C++ interface to the compressor loader
 */
#ifndef LIBPRESSIO_PRESSIO_IMPL_H
#define LIBPRESSIO_PRESSIO_IMPL_H
#include <map>
#include <memory>
#include <functional>
#include "compressor.h"
#include "metrics.h"
#include "io.h"
#include "std_compat/language.h"

/**
 * a type that registers constructor functions
 */
template <class T>
struct pressio_registry {
  /**
   * construct a element of the registered type
   *
   * \param[in] name the item to construct
   * \returns the result of the factory function
   */
  T build(std::string const& name) const {
    auto factory = factories.find(name);
    if ( factory != factories.end()) {
      return factory->second();
    } else {
      return nullptr;
    }
  }

  /**
   * register a factory function with the registry at the provided name
   *
   * \param[in] name the name to register
   * \param[in] factory the constructor function which takes 0 arguments
   */
  template <class Name, class Factory>
  void regsiter_factory(Name&& name, Factory&& factory) {
    factories.emplace(std::forward<Name>(name), std::forward<Factory>(factory));
  }

  private:
  std::map<std::string, std::function<T()>> factories;

  public:
  /** the value type the registry constructs*/
  using value_type = T;
  /** the reference type the registry constructs*/
  using reference = T&;
  /** the const reference type the registry constructs*/
  using const_reference = T const;
  /**
   * \returns an begin iterator over the registered types
   */
  auto begin() const -> decltype(factories.begin()) { return std::begin(factories); }
  /**
   * \returns an end iterator over the registered types
   */
  auto end() const -> decltype(factories.end()) { return std::end(factories); }

  /**
   * checks if the name is registered
   *
   * \param[in] key the key to search for
   * \returns true if present
   */
  bool contains(std::string const& key) const {
    return factories.find(key) != factories.end();
  }

  /**
   * checks if the name is registered
   *
   * \param[in] key the key to search for
   * \returns an iterator if the entry is found; else end()
   */
  auto find(std::string const& key) const -> decltype(factories.find(key)) {
    return factories.find(key);
  }

  /**
   * checks if the name is registered
   *
   * \param[in] key the key to search for
   * \returns an iterator if the entry is found; else end()
   */
  auto find(std::string const& key) -> decltype(factories.find(key)) {
    return factories.find(key);
  }

};

/**
 * a class that registers a type on construction, using a type over a function
 * to force it to be called at static construction time
 */
class pressio_register{
  public:
  /**
   * Registers a new factory with a name in a registry.  Designed to be used as a static variable
   *
   * \param[in] registry the registry to use
   * \param[in] name the name to register
   * \param[in] factory the factory to register
   */
  template <class RegistryType, class NameType, class Factory>
  pressio_register(pressio_registry<RegistryType>& registry, NameType&& name, Factory&& factory) {
    registry.regsiter_factory(name, factory);
  }
};

/**
 * the registry for compressor plugins
 */
pressio_registry<std::shared_ptr<libpressio_compressor_plugin>>& compressor_plugins();
/**
 * the registry for metrics plugins
 */
pressio_registry<std::unique_ptr<libpressio_metrics_plugin>>& metrics_plugins();

/**
 * the registry for metrics plugins
 */
pressio_registry<std::unique_ptr<libpressio_io_plugin>>& io_plugins();

/**
 * the libraries basic state
 */
struct pressio {
  public:

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
  std::shared_ptr<libpressio_compressor_plugin> get_compressor(std::string const& compressor_id) {
    auto compressor = compressor_plugins().build(compressor_id);
    if (compressor) return compressor;
    else {
      set_error(1, std::string("invalid compressor id ") + compressor_id);
      return nullptr;
    }
  }

  /**
   * Returns an io module
   * \param[in] io_module_id name of the compressor to request
   * \returns an instance of compressor registered at name, or nullptr on error
   */
  std::shared_ptr<libpressio_io_plugin> get_io(std::string const& io_module_id)
  {
    auto io_module = io_plugins().build(io_module_id);
    if (io_module)
      return RVO_MOVE(io_module);
    else {
      set_error(1, std::string("invalid io_plugin id ") + io_module_id);
      return nullptr;
    }
  }

  /**
   * Returns an metrics module
   * \param[in] id name of the compressor to request
   * \returns an instance of compressor registered at name, or nullptr on error
   */
  template <class Str>
  std::unique_ptr<libpressio_metrics_plugin> get_metric(Str id) {
    auto ret = metrics_plugins().build(id);
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
  std::unique_ptr<libpressio_metrics_plugin> get_metrics(ForwardIt first, ForwardIt last) {
    std::vector<pressio_metrics> plugins;

    for (auto metric = first; metric != last; ++metric) {
      plugins.emplace_back(metrics_plugins().build(*metric));
      if(not plugins.back()) {
        set_error(2, std::string("failed to construct metrics plugin: ") + *metric);
        return nullptr;
      }
    }

    auto metrics = make_m_composite(std::move(plugins));

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
