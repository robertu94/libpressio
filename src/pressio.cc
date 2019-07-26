#include <map>
#include <memory>
#include <string>
#include "pressio.h"
#include "pressio_version.h"
#include "plugins.h"

#include "pressio_compressor_impl.h"

struct pressio {
  public:
  std::map<std::string, pressio_compressor> compressors;
  struct {
    int code;
    std::string msg;
  } error;
};

extern "C" {
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
    std::string compressor_id_s = compressor_id;
    if(compressor_id_s == "sz") {
      library->compressors.emplace("sz",pressio_compressor(make_sz())); 
    } else if (compressor_id_s == "zfp") {
      library->compressors.emplace("zfp",pressio_compressor(make_zfp())); 
    } else {
      return nullptr;
    }
    return &library->compressors[compressor_id];
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
