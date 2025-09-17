#include "libpressio_ext/cpp/versionable.h"

namespace libpressio {
uint64_t pressio_versionable::epoch_version() const { return 0; }
int pressio_versionable::major_version() const { return 0; }
int pressio_versionable::minor_version() const { return 0; }
int pressio_versionable::patch_version() const { return 0; }
}
