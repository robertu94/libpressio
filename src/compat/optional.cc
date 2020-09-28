#include "libpressio_ext/compat/optional.h"

#if !(LIBPRESSIO_COMPAT_HAS_OPTIONAL)
const boost::none_t& nullopt = boost::none;
#endif
