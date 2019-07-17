#ifndef LIBLOSSY_PLUGINS_PUBLIC
#define LIBLOSSY_PLUGINS_PUBLIC
#include <memory>
#include "plugins/liblossy_plugin.h"
std::unique_ptr<liblossy_plugin> make_sz();
std::unique_ptr<liblossy_plugin> make_zfp();
#endif

