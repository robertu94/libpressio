#include <libpressio_ext/cpp/domain.h>

namespace libpressio {
    pressio_registry<std::shared_ptr<domains::pressio_domain>>& domain_plugins() {
    static pressio_registry<std::shared_ptr<domains::pressio_domain>> registry;
    return registry;
}
}
