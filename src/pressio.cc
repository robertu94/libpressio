#include <vector>
#include <map>
#include <memory>
#include <string>
#include <sstream>
#include <functional>
#include "pressio.h"
#include "pressio_version.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/compressor.h"



pressio_registry<std::shared_ptr<libpressio_compressor_plugin>>& compressor_plugins() {
  static pressio_registry<std::shared_ptr<libpressio_compressor_plugin>> registry;
  return registry;
}

pressio_registry<std::unique_ptr<libpressio_metrics_plugin>>& metrics_plugins() {
  static pressio_registry<std::unique_ptr<libpressio_metrics_plugin>> registry;
  return registry;
}

extern "C" {

struct pressio* pressio_instance() {
  return new pressio;
}

//IMPLEMENTATION NOTE this function exists to preserve the option of releasing the memory for the library in the future.
//currently this is undesirable because some libraries such as SZ don't handle this well, but may be possible
//after the planned C++ rewrite.
//
//Therefore, it intentionally does not release the memory
void pressio_release(struct pressio* library) {
  delete library;
}

int pressio_error_code(struct pressio* library) {
  return library->err_code();
}

const char* pressio_error_msg(struct pressio* library) {
  return library->err_msg().c_str();
}

void pressio_set_error(pressio* library, int code, const char* msg) {
  library->set_error(code, msg);
}


struct pressio_compressor* pressio_get_compressor(struct pressio* library, const char* compressor_id) {
  auto compressor = library->get_compressor(compressor_id);
  if(compressor != nullptr) return new pressio_compressor(std::move(compressor));
  else return nullptr;
}

struct pressio_metrics* pressio_new_metrics(struct pressio* library, const char* metrics_ids[], int num_metrics) {
  auto metrics = library->get_metrics(metrics_ids, metrics_ids+num_metrics);
  if (metrics != nullptr) return new pressio_metrics(std::move(metrics));
  else return nullptr;
}

const char* pressio_version() {
  return pressio::version();
}

const char* pressio_features() {
  return pressio::features();
}

const char* pressio_supported_compressors() {
  return pressio::supported_compressors();
}

const char* pressio_supported_metrics() {
  return pressio::supported_metrics();
}

unsigned int pressio_major_version() {
  return pressio::major_version();
}

unsigned int pressio_minor_version() {
  return pressio::minor_version();
}

unsigned int pressio_patch_version() {
  return pressio::patch_version();
}

}


const char* pressio::version() {
  return LIBPRESSIO_VERSION;
}

const char* pressio::features() {
  return LIBPRESSIO_FEATURES;
}

template <class T>
static std::string build_from(T const& plugins) {
  std::ostringstream os;
  for (auto const& it : plugins) {
    os << it.first << " "; 
  }
  return os.str();
}

const char* pressio::supported_compressors() {
  static std::string modules = build_from(compressor_plugins());
  return modules.c_str();
}

const char* pressio::supported_metrics() {
  static std::string modules = build_from(metrics_plugins());
  return modules.c_str();
}

const char* pressio::supported_io() {
  static std::string modules = build_from(io_plugins());
  return modules.c_str();
}

unsigned int pressio::major_version() {
  return LIBPRESSIO_MAJOR_VERSION;
}

unsigned int pressio::minor_version() {
  return LIBPRESSIO_MINOR_VERSION;
}

unsigned int pressio::patch_version() {
  return LIBPRESSIO_PATCH_VERSION;
}

