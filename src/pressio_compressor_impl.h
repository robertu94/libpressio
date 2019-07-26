#ifndef LIBPRESSIO_COMPRESSOR_IMPL
#define LIBPRESSIO_COMPRESSOR_IMPL
#include <memory>
#include "plugins/libpressio_plugin.h"

struct pressio_compressor {
  pressio_compressor(std::unique_ptr<libpressio_plugin>&& impl): plugin(std::forward<std::unique_ptr<libpressio_plugin>>(impl)) {}
  pressio_compressor()=default;
  pressio_compressor(pressio_compressor&& compressor)=default;
  pressio_compressor& operator=(pressio_compressor&& compressor)=default;
  std::unique_ptr<libpressio_plugin> plugin;
};
#endif
