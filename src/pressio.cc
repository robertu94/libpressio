#include <vector>
#include <map>
#include <memory>
#include <string>
#include <functional>
#include "pressio.h"
#include "pressio_version.h"
#include <libpressio_ext/cpp/plugins.h>
#include "libpressio_ext/cpp/metrics.h"

#include "pressio_compressor_impl.h"

namespace {

  std::map compressor_constructors{
    std::pair(std::string("sz"), std::function(make_c_sz)),
    std::pair(std::string("zfp"), std::function(make_c_zfp)),
  };
  std::map metrics_constructor{
    std::pair(std::string("time"), std::function(make_m_time)),
    std::pair(std::string("size"), std::function(make_m_size)),
  };

  template <class MapType>
  typename MapType::mapped_type get_or(MapType map, typename MapType::key_type const& key) {
    if(auto it = map.find(key); it != map.end()) {
      return it->second;
    } else {
      return nullptr;
    }
  }
}

extern "C" {

struct pressio {
  public:
  std::map<std::string, pressio_compressor> compressors;
  struct {
    int code;
    std::string msg;
  } error;
};

struct pressio* pressio_instance() {
  static struct pressio library;
  return &library;
}


//IMPLEMENTATION NOTE this function exists to preserve the option of releasing the memory for the library in the future.
//currently this is undesirable because some libraries such as SZ don't handle this well, but may be possible
//after the planned C++ rewrite.
//
//Therefore, it intentionally does not release the memory
void pressio_release(struct pressio** library) {
  *library = nullptr;
}

int pressio_error_code(struct pressio* library) {
  return library->error.code;
}

const char* pressio_error_msg(struct pressio* library) {
  return library->error.msg.c_str();
}

struct pressio_compressor* pressio_get_compressor(struct pressio* library, const char* const compressor_id) {
  if(auto compressor = library->compressors.find(compressor_id); compressor != library->compressors.end())
  {
    return &compressor->second;
  } else {
    auto constructor = get_or(compressor_constructors, compressor_id);
    if(constructor) {
      library->compressors[compressor_id] = constructor();
      return &library->compressors[compressor_id];
    } else {
      return nullptr;
    }
  }
}


struct pressio_metrics* pressio_new_metrics(struct pressio* library, const char* metrics_ids[], int num_metrics) {
  (void)library;
  try {
    std::vector<std::unique_ptr<libpressio_metrics_plugin>> plugins;
    for (int i = 0; i < num_metrics; ++i) {
      auto constructor = get_or(metrics_constructor, metrics_ids[i]);
      if(constructor) {
        plugins.emplace_back(constructor());
      } else {
        throw std::runtime_error("failed to locate metrics");
      }
    }

    auto metrics = make_m_composite(std::move(plugins));
    return new pressio_metrics{std::move(metrics)};
  } catch (std::runtime_error const&) {
    return nullptr;
  }
}

const char* pressio_version() {
  return LIBPRESSIO_VERSION;
}
const char* pressio_features() {
  return LIBPRESSIO_FEATURES;
}
unsigned int pressio_major_version() {
  return LIBPRESSIO_MAJOR_VERSION;
}
unsigned int pressio_minor_version() {
  return LIBPRESSIO_MINOR_VERSION;
}
unsigned int pressio_patch_version() {
  return LIBPRESSIO_PATCH_VERSION;
}

}
