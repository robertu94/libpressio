#ifndef LIBLOSSY_COMPRESSOR_IMPL
#define LIBLOSSY_COMPRESSOR_IMPL
#include <memory>
#include "plugins/liblossy_plugin.h"

struct lossy_compressor {
  lossy_compressor(std::unique_ptr<liblossy_plugin>&& impl): plugin(std::forward<std::unique_ptr<liblossy_plugin>>(impl)) {}
  lossy_compressor()=default;
  lossy_compressor(lossy_compressor&& compressor)=default;
  lossy_compressor& operator=(lossy_compressor&& compressor)=default;
  std::unique_ptr<liblossy_plugin> plugin;
};
#endif
