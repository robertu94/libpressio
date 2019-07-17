#include <map>
#include <memory>
#include <string>
#include "lossy.h"
#include "lossy_version.h"
#include "plugins.h"

#include "lossy_compressor_impl.h"

struct lossy {
  public:
  std::map<std::string, lossy_compressor> compressors;
  struct {
    int code;
    std::string msg;
  } error;
};

extern "C" {
struct lossy* lossy_instance() {
  static struct lossy library;
  return &library;
}


//IMPLEMENTATION NOTE this function exists to preserve the option of releasing the memory for the library in the future.
//currently this is undesirable because some libraries such as SZ don't handle this well, but may be possible
//after the planned C++ rewrite.
//
//Therefore, it intentionally does not release the memory
void lossy_release(struct lossy** library) {
  *library = nullptr;
}

int lossy_error_code(struct lossy* library) {
  return library->error.code;
}

const char* lossy_error_msg(struct lossy* library) {
  return library->error.msg.c_str();
}


struct lossy_compressor* lossy_get_compressor(struct lossy* library, const char* const compressor_id) {
  if(auto compressor = library->compressors.find(compressor_id); compressor != library->compressors.end())
  {
    return &compressor->second;
  } else {
    std::string compressor_id_s = compressor_id;
    if(compressor_id_s == "sz") {
      library->compressors.emplace("sz",lossy_compressor(make_sz())); 
    } else if (compressor_id_s == "zfp") {
      library->compressors.emplace("zfp",lossy_compressor(make_zfp())); 
    } else {
      return nullptr;
    }
    return &library->compressors[compressor_id];
  }
}

const char* lossy_version() {
  return LIBLOSSY_VERSION;
}
const char* lossy_features() {
  return LIBLOSSY_FEATURES;
}
unsigned int lossy_major_version() {
  return LIBLOSSY_MAJOR_VERSION;
}
unsigned int lossy_minor_version() {
  return LIBLOSSY_MINOR_VERSION;
}
unsigned int lossy_patch_version() {
  return LIBLOSSY_PATCH_VERSION;
}

}
