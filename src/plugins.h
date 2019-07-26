#ifndef LIBPRESSIO_PLUGINS_PUBLIC
#define LIBPRESSIO_PLUGINS_PUBLIC
#include <memory>
#include "plugins/libpressio_plugin.h"
std::unique_ptr<libpressio_plugin> make_sz();
std::unique_ptr<libpressio_plugin> make_zfp();
#endif

