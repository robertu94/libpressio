#include "libpressio_ext/compat/span.h"

#if !LIBPRESSIO_COMPAT_HAS_SPAN
namespace compat {
const size_t dynamic_extent = compat_dynamic_extent;
}
#endif
